import os
import datetime
import pytz


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import reduce
from time import time
from os.path import join
import numpy as np
import pandas as pd
import click

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import norm
import jax.tree_util as jtu
from tqdm import trange

import numpy as np
from pyDOE2.doe_lhs import lhs

import jaxopt

import gpax.kernels as gpk
import gpax.likelihoods as gpl
import gpax.means as gpm
from gpax.models import (
    ExactGPRegression,
    LatentGPHeinonen,
    LatentGPDeltaInducing,
    LatentGPPlagemann,
    LatentGPGaussianBasis,
    SparseGPRegression,
)
from gpax.utils import index_pytree, DataScaler
from gpax.core import set_positive_bijector, set_default_jitter

jax.config.update("jax_enable_x64", True)

set_default_jitter(1e-3)


def get_ts():
    # Get the current local time and date
    now = datetime.datetime.now()

    # Get the timezone information
    tz = pytz.timezone("Asia/Kolkata")  # GMT+5:30

    # Convert the datetime object to the specified timezone
    local_time = tz.localize(now)

    # Format the date and time
    date_time = local_time.strftime("%m/%d/%Y, %H:%M:%S %Z")

    return date_time


def print(*args, **kwargs):
    prefix = f"[{get_ts()}] "
    # Call the original print function with the modified arguments
    return __builtins__.print(prefix, *args, **kwargs)


def load_data(args, debug=True):
    root = "/home/anonymous"
    train_data = pd.read_csv(f"{root}/AAAI22/data/time_feature/fold{args.fold}/train_data_mar_nsgp.csv.gz")
    if args.mode == "test":
        test_data = pd.read_csv(f"{root}/AAAI22/data/time_feature/fold{args.fold}/test_data_mar_nsgp.csv.gz")
    X_cat_fet = ["weather", "wind_direction"]
    X_noncat_fet = [
        "latitude",
        "longitude",
        "temperature",
        "humidity",
        "wind_speed",
        "delta_t",
    ]
    all_fet = X_noncat_fet + X_cat_fet
    y_feature = "PM25_Concentration"

    start = "2015-03-01"
    end = "2015-03-31"

    def preprocess(data, scaler=None):
        data["time"] = pd.to_datetime(data["time"])
        data = data.set_index("time")
        data = data[start:end]
        # print(data.shape)

        X = data[args.features].values
        y = data[y_feature].values
        active_dims = [all_fet.index(fet) for fet in args.features if fet in X_noncat_fet]
        if debug:
            print(X.shape, y.shape)

        if scaler is None:
            scaler = DataScaler(X, y, active_dims=active_dims)
        X, y = scaler.transform(jnp.asarray(X), jnp.asarray(y))
        return X, y, scaler

    cut_days = 0
    train_prefix = cut_days * 24 * 20
    test_prefix = cut_days * 24 * 10

    X_train, y_train, scaler = preprocess(train_data.iloc[: train_data.shape[0] - train_prefix])
    if args.mode == "test":
        X_test, y_test, _ = preprocess(test_data.iloc[: test_data.shape[0] - test_prefix], scaler)
    else:
        X_test, y_test = None, None
    return X_train, y_train, X_test, y_test, scaler


def get_x_inducing(args, X, debug=True):
    # X_inducing = lhs(X.shape[1], samples=args.n_inducing)
    # X_inducing = jax.vmap(lambda min_val, max_val, x_inducing: min_val + (max_val - min_val) * x_inducing)(
    #     X.min(axis=0), X.max(axis=0), X_inducing.T
    # ).T
    # key = jr.PRNGKey(args.seed + 1234)
    # X_inducing_ind = jax.random.choice(key, X.shape[0], (args.n_inducing,), replace=False)
    # X_inducing = X[X_inducing_ind]

    np.random.seed(args.seed + 1234)
    X_inducing = lhs(X.shape[1], samples=args.n_inducing)
    if debug:
        print(X_inducing.shape)

    def scale_x(x_inducing, x):
        return x_inducing * (x.max() - x.min()) + x.min()

    X_inducing = jax.vmap(scale_x, in_axes=(1, 1), out_axes=1)(X_inducing, X)

    if "weather" in args.features:
        weather_unique = jnp.unique(X[:, args.features.index("weather")])
        weather_key = jr.PRNGKey(args.seed + 12345)
        weather_inducing = jr.choice(weather_key, weather_unique, (args.n_inducing,), replace=True)
        X_inducing = X_inducing.at[:, args.features.index("weather")].set(weather_inducing)

    if "wind_direction" in args.features:
        wind_direction_key = jr.PRNGKey(args.seed + 123456)
        wind_direction_unique = jnp.unique(X[:, args.features.index("wind_direction")])
        wind_direction_inducing = jr.choice(wind_direction_key, wind_direction_unique, (args.n_inducing,), replace=True)
        X_inducing = X_inducing.at[:, args.features.index("wind_direction")].set(wind_direction_inducing)

    # skip_ts = 5
    # n_stations = 20
    # X_inducing = jnp.concatenate([X[i : i + n_stations] for i in range(0, X.shape[0], skip_ts * n_stations)])

    return X_inducing


def build_model(args, X, debug=True):
    GP = ExactGPRegression if args.gp == "exact" else SparseGPRegression if args.gp == "sparse" else None
    Kernel = gpk.RBF if args.kernel == "rbf" else gpk.Matern32 if args.kernel == "matern32" else None
    TimeKernel = gpk.RBF if args.time_kernel == "rbf" else gpk.Matern32 if args.time_kernel == "matern32" else None

    # Build kernel
    kernels = []
    cont_features = args.features.copy()
    for cat_fet in ["weather", "wind_direction"]:
        if cat_fet in args.features:
            wind_dir_idx = args.features.index(cat_fet)
            wind_kernel = gpk.Hamming(X, active_dims=[wind_dir_idx])
            kernels.append(wind_kernel)
            cont_features.remove(cat_fet)

    if "delta_t" in args.features:
        time_idx = args.features.index("delta_t")
        time_kernel = TimeKernel(X, active_dims=[time_idx]) * gpk.Periodic(X, active_dims=[time_idx])
        kernels.append(time_kernel)
        cont_features.remove("delta_t")

    cont_idx = [args.features.index(fet) for fet in cont_features]

    latent_models = {
        "heinonen": LatentGPHeinonen,
        "delta": LatentGPDeltaInducing,
        "plagemann": LatentGPPlagemann,
        "gaussian": LatentGPGaussianBasis,
    }
    LatentModel = latent_models[args.latent_model]

    def get_latent_kwargs(ls):
        if args.latent_model == "gaussian":
            kwargs = {"grid_size": args.grid_size, "active_dims": lat_long_time_idx}
        else:
            kwargs = {"kernel": (1.0 * gpk.RBF(X, active_dims=lat_long_time_idx, lengthscale=ls)).trainable(False)}

        kwargs["sparse"] = True if args.gp == "sparse" else False

        return kwargs

    lat_long_time_idx = [args.features.index(fet) for fet in ["latitude", "longitude", "delta_t"]]
    for idx in lat_long_time_idx:
        if idx in cont_idx:
            cont_idx.remove(idx)
    if args.latent_model == "heinonen" and args.gp == "exact":
        X_inducing = X
    else:
        X_inducing = get_x_inducing(args, X, debug)

    if debug:
        print("X_inducing.shape", X_inducing.shape)

    if "l" in args.model_type:
        latent_model = LatentModel(X_inducing[:, lat_long_time_idx], vmap=True, **get_latent_kwargs(ls=0.2))
        lat_long_time_kernel = gpk.Gibbs(X, latent_model, active_dims=lat_long_time_idx)
        met_kernel = Kernel(X, active_dims=cont_idx)
        cont_kernel = lat_long_time_kernel * met_kernel
    else:
        cont_kernel = Kernel(X, active_dims=lat_long_time_idx + cont_idx)

    kernels.append(cont_kernel)

    base_kernel = reduce(lambda x, y: x * y, kernels)

    if "s" in args.model_type:
        kwargs = get_latent_kwargs(ls=0.2)
        # kwargs["kernel"].trainable(True)
        latent_model = LatentModel(X_inducing[:, lat_long_time_idx], **kwargs)
        kernel = gpk.InputDependentScale(X_inducing, base_kernel, latent_model)
    else:
        kernel = gpk.Scale(X, base_kernel)

    # Build likelihood
    if "o" in args.model_type:
        latent_model = LatentModel(X_inducing[:, lat_long_time_idx], **get_latent_kwargs(ls=0.3))
        likelihood = gpl.Heteroscedastic(latent_model)
    else:
        likelihood = gpl.Gaussian()

    model = GP(kernel, likelihood, gpm.Average(), X_inducing)
    return model


def train(args):
    init = time()
    print("fold", args.fold, "seed", args.seed, "features", args.features)
    X_train, y_train, _, _, _ = load_data(args)

    model = build_model(args, X_train)

    def find_gibbs(kernel):
        if isinstance(kernel, gpk.Gibbs):
            return kernel
        elif isinstance(kernel, (gpk.Product, gpk.Sum)):
            res = find_gibbs(kernel.k1)
            if res is not None:
                return res
            else:
                return find_gibbs(kernel.k2)
        else:
            return None

    def customize_fn(model):
        if "o" in args.model_type:
            model.likelihood.latent_model.reverse_init(jnp.array(0.05))
        if "s" in args.model_type:
            model.kernel.latent_model.reverse_init(jnp.array(0.3))
        if "l" in args.model_type:
            find_gibbs(model.kernel.base_kernel).latent_model.reverse_init(jnp.array(0.05))
            model.likelihood.scale.set_value(jnp.array(0.05))

    train_key = jr.PRNGKey(args.seed)
    res = model.fit(
        train_key,
        X_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        customize_fn=customize_fn,
        optimizer_name=args.optimizer,
    )

    pd.to_pickle(res, join(args.root, f"seed-{args.seed}_fet-{'_'.join(sorted(args.features))}.pkl"))
    print(f"Training took {time() - init:.2f} seconds")
    return res


def test(args):
    print("fold", args.fold, "features", args.features)
    init = time()
    X_train, y_train, X_test, y_test, scaler = load_data(args)

    model = build_model(args, X_train)
    res = {
        seed: pd.read_pickle(join(args.root, f"seed-{seed}_fet-{'_'.join(sorted(args.features))}.pkl"))
        for seed in range(4)
    }
    print([value.keys() for value in res.values()])
    seed_and_loss = sorted([(key, value["loss_history"][-1]) for key, value in res.items()], key=lambda x: x[1])
    print("seed and loss", seed_and_loss)
    best_seed = seed_and_loss[0][0]
    best_res = res[best_seed]
    model.set_raw_parameters(best_res["raw_params"])
    predict_fn = model.condition(X_train, y_train)

    pred_mean, pred_var = predict_fn(X_test)
    pred_mean_train, pred_var_train = predict_fn(X_train)

    (X_train, X_test), (y_train, y_test, pred_mean, pred_mean_train) = scaler.inverse_transform(
        (X_train, X_test), (y_train, y_test, pred_mean, pred_mean_train)
    )
    pred_std = jnp.sqrt(pred_var) * scaler.y_scale
    pred_std_train = jnp.sqrt(pred_var_train) * scaler.y_scale

    save_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_mean_train": pred_mean_train,
        "pred_std_train": pred_std_train,
    }

    res.update(save_dict)

    pd.to_pickle(res, join(args.root, f"fet-{'_'.join(sorted(args.features))}.pkl"))

    print(f"Testing took {time() - init:.2f} seconds")


@click.command()
@click.option("--mode", help="`train` or `test`")
@click.option("--fold", help="Fold number (0, 1, 2)")
@click.option("--lr", help="Learning rate")
@click.option("--epochs", help="Number of epochs")
@click.option("--n_inducing", help="Number of inducing points")
@click.option("--kernel", help="Kernel type (rbf, matern32)")
@click.option("--time_kernel", help="Time kernel type (rbf, matern32)")
@click.option("--optimizer", help="Optimizer type (adam, lbfgsb, bfgs)")
@click.option("--gp", help="GP type (exact, sparse)")
@click.option("--model_type", help="Type of experiment (e, l, ls, lso)")
@click.option("--latent_model", help="Latent model (heinonen, delta, plagemann, gaussian)")
@click.option("--grid_size", help="Grid size for gaussian latent model")
@click.option("--features", help="Features")
@click.option("--seed", help="Seed")
@click.option("--rootdir", help="Root directory")
def run(
    mode,
    fold,
    lr,
    epochs,
    n_inducing,
    kernel,
    time_kernel,
    optimizer,
    gp,
    model_type,
    latent_model,
    grid_size,
    features,
    seed,
    rootdir,
):
    # store all arguments in a dictionary like structure but should be accessible by dot notation
    args = type(
        "args",
        (),
        {
            "mode": mode,
            "fold": int(fold),
            "lr": float(lr),
            "epochs": int(epochs),
            "n_inducing": int(n_inducing),
            "kernel": kernel,
            "time_kernel": time_kernel,
            "optimizer": optimizer,
            "gp": gp,
            "model_type": model_type,
            "latent_model": latent_model,
            "grid_size": int(grid_size),
            "features": eval(features),
            "seed": int(seed),
            "root": f"./{rootdir}/{model_type}-{gp}-{latent_model}-grid-{grid_size}-ind-{n_inducing}-{kernel}-time-{time_kernel}/fold-{fold}/opt-{optimizer}-epochs-{epochs}-lr-{lr}",
        },
    )

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if mode == "train":
        train(args)
    elif mode == "test":
        test(args)


if __name__ == "__main__":
    run()


####################### Previous Code #######################
# import sys
# import os
# from parsing import unparse, parse
# import subprocess
# import random
# import toml
# from time import sleep
# import logging

# all_gpus = [0, 1, 2, 3]
# max_wait_time = 30 * 60  # 30 minutes
# sleep_time = 5  # check every 5 seconds
# log_time = 30  # log every 30 seconds
# divider = log_time // sleep_time
# max_retries = max_wait_time // sleep_time


# def find_empty_gpu(retry=0):
#     if retry > max_retries:
#         logging.info(f"No GPUs available, retry:{retry}/{max_retries}, exiting...")
#         raise Exception("No GPUs available")

#     try:
#         with open("logs/global_gpu.log", "r") as f:
#             lines = f.read().strip().split("\n")
#     except FileNotFoundError:
#         lines = []
#     gpus = [int(line.strip()) for line in lines if line.strip() != ""]
#     available_gpus = set(all_gpus) - set(gpus)
#     if len(available_gpus) == 0:
#         if retry % divider == 0:
#             logging.info(f"No GPUs available, retry:{retry}/{max_retries}, waiting...")
#         sleep(sleep_time)
#         return find_empty_gpu(retry + 1)
#     gpu_id = random.choice(list(available_gpus))
#     return gpu_id


# ######## Set config
# config_name = sys.argv[1]
# mode = sys.argv[2]  # train or test
# fold = int(sys.argv[3])  # 0, 1, 2

# config = toml.load(f"configs/{config_name}.toml")
# config.update({"name": config_name, "fold": fold})
# config_path = "configs"
# models = {"egp": "exact_gp"}
# str_config = unparse(config)

# ######## Set Logging
# logging.basicConfig(
#     format="%(asctime)s %(levelname)s %(message)s",
#     filename=f"logs/runtime_{str_config}.log",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S %Z",
# )
# config = parse(str_config)  # sort config
# logging.info(f"Initialized {config}")

# ######## Set GPs
# gpu_id = find_empty_gpu(retry=0)
# logging.info(f"GPU {gpu_id} found, launching {mode}...")
# config.update({"gpu_id": gpu_id})
# str_config = unparse(config)

# ######## Run
# pid = subprocess.Popen(["python", f"{mode}.py", str_config], start_new_session=True).pid

# ######## Log
# with open(f"logs/global_gpu.log", "a") as f:
#     print(f"{gpu_id}", file=f)

# with open(f"logs/pid_{config_name}.log", "a") as f:
#     print("PID:", pid, file=f)
