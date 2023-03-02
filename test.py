def free_up_gpu(gpu_id):
    print("Releasing GPU", gpu_id)
    with open("logs/global_gpu.log", "r") as f:
        lines = f.read().strip().split("\n")
    gpus = [int(line.strip()) for line in lines if line.strip() != ""]
    available_gpus = set(gpus) - set([int(gpu_id)])
    with open("logs/global_gpu.log", "w") as f:
        for gpu in available_gpus:
            print(f"{gpu}", file=f)


import sys
from parsing import parse, unparse

str_config = sys.argv[1]
config = parse(str_config)
gpu_id = config.pop("gpu_id")
str_config = unparse(config)

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    filename=f"logs/runtime_{str_config}.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)
logging.info(f"Testing {str_config}")

try:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    import numpy as np
    import pandas as pd

    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import jax.tree_util as jtu
    from tqdm import trange

    import tensorflow_probability.substrates.jax as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors

    import arviz as az
    import regdata as rd

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    from time import sleep, time

    from common import get_data, get_model

    jax.config.update("jax_enable_x64", True)
    start_time = time()

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        cont_idx_list,
        non_cont_idx_list,
    ) = get_data(config, test=True)
    model = get_model(config, X_train)

    res = pd.read_pickle(f"results/train_{str_config}.pkl")

    # np.random.seed(0)

    X_inducing = None

    # # X_inducing = X_train[::10]
    # # print(f"{X_inducing.shape[0]} inducing points")
    # n_inducing = 2100

    # df = pd.DataFrame(X_train, columns=all_fet)
    # for col in df:
    #     print(col, df[col].unique().shape)

    # # X_inducing_ind = jax.random.choice(jr.PRNGKey(100), X_train.shape[0], (n_inducing, ), replace=False)
    # # X_inducing = X_train[X_inducing_ind]
    # # df = pd.DataFrame(X_inducing, columns=all_fet)
    # # for col in df:
    # #     print(col, df[col].unique().shape)

    # ################## Unique method
    # def choose_x(x):
    #     return jnp.linspace(x.min(), x.max(), n_inducing)

    # X_inducing = jax.vmap(choose_x, in_axes=1, out_axes=1)(X_train)

    # ################## LHS method
    # X_inducing = lhs(X_train.shape[1], samples=n_inducing)
    # print(X_inducing.shape)
    # def scale_x(x_inducing, x):
    #     return x_inducing * (x.max() - x.min()) + x.min()

    # X_inducing = jax.vmap(scale_x, in_axes=(1, 1), out_axes=1)(X_inducing, X_train)

    # weather_unique = jnp.unique(X_train[:, weather_idx])
    # weather_inducing = weather_unique.repeat(n_inducing // weather_unique.shape[0])
    # wind_direction_unique = jnp.unique(X_train[:, wind_dir_idx])
    # wind_direction_inducing = wind_direction_unique.repeat(n_inducing // wind_direction_unique.shape[0])

    # X_inducing = X_inducing.at[:, weather_idx].set(weather_inducing)
    # X_inducing = X_inducing.at[:, wind_dir_idx].set(wind_direction_inducing)

    # df = pd.DataFrame(X_inducing, columns=all_fet)
    # for col in df:
    #     print(col, df[col].unique().shape)

    # print(X_inducing.shape)
    # model = SparseGPRegression(kernel, gpl.Gaussian(), gpm.Average(), X_inducing=X_inducing)

    # jax.pmap(lambda key: model.log_probability(X_train, y_train))(keys)

    # def init_fn(key):
    #     model.initialize(key)
    #     return model.get_raw_parameters()
    #     # return model.log_probability(X_train, y_train)

    # def log_prob_fn(params):
    #     model.set_raw_parameters(params)
    #     return -model.log_probability(X_train, y_train)

    # loss_fn(keys[0]), loss_fn(keys[1])
    # inits = jax.vmap(init_fn)(keys)
    # print(inits)
    # losses = jax.vmap(jax.jit(log_prob_fn))(inits)
    # print(losses)

    str_config = unparse(config)

    model.set_raw_parameters(res["raw_params"])

    pred_mean, pred_var = model.predict(X_train, y_train, X_test)
    pred_std = jnp.sqrt(pred_var)
    test_res = {"pred_mean": pred_mean, "pred_std": pred_std}

    if config["test_history"] == "true" and config["optimizer"] != "lbfgs":

        @jax.jit
        def predict_i(i):
            raw_params = index_pytree(res["raw_params_history"], i)
            model.set_raw_parameters(raw_params)
            pred_fn = model.condition(X_train, y_train)
            pred_mean, pred_var = pred_fn(X_test)
            pred_std = jnp.sqrt(pred_var)
            pred_mean_train, pred_var_train = pred_fn(X_train)
            pred_std_train = jnp.sqrt(pred_var_train)

            return (
                pred_mean,
                pred_mean_train,
                pred_std,
                pred_std_train,
            )

        epochs = int(config["epochs"])
        pred_mean_history = np.empty((epochs, X_test.shape[0]))
        pred_mean_train_history = np.empty((epochs, X_train.shape[0]))
        pred_std_history = np.empty((epochs, X_test.shape[0]))
        pred_std_train_history = np.empty((epochs, X_train.shape[0]))
        for i in range(epochs):
            (
                pred_mean_history[i],
                pred_mean_train_history[i],
                pred_std_history[i],
                pred_std_train_history[i],
            ) = predict_i(i)

        test_res["pred_mean_history"] = pred_mean_history
        test_res["pred_mean_train_history"] = pred_mean_train_history
        test_res["pred_std_history"] = pred_std_history
        test_res["pred_std_train_history"] = pred_std_train_history

    pd.to_pickle(test_res, f"results/test_{str_config}.pkl")
    end_time = time()
    logging.info(f"Testing took {end_time - start_time:.2f} seconds")
    logging.info(f"Saved results to {str_config}")

    logging.info("Freeing up GPU")
    free_up_gpu(gpu_id)

except KeyboardInterrupt:
    logging.info("Keyboard interrupt")
finally:
    logging.info("Freeing up GPU")
    free_up_gpu(gpu_id)
