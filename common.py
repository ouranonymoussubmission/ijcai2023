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

jax.config.update("jax_enable_x64", True)


############## root path
root = "/home/anonymous"


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
time_idx = all_fet.index("delta_t") if "delta_t" in all_fet else -1
weather_idx = all_fet.index("weather") if "weather" in all_fet else -1
wind_dir_idx = all_fet.index("wind_direction") if "wind_direction" in all_fet else -1


def preprocess(data, scaler=None, scale=True):
    data["time"] = pd.to_datetime(data["time"])
    data = data.set_index("time")
    data = data[start:end]
    # print(data.shape)

    X_cat = data[X_cat_fet].values
    X_noncat = data[X_noncat_fet].values
    y = data[y_feature].values
    # print(X_cat.shape, X_noncat.shape, y.shape)

    if scaler is None:
        scaler = DataScaler(X_noncat, y)
    if scale:
        X_noncat, y = scaler.transform(X_noncat, y)
    X = jnp.concatenate([X_noncat, X_cat], axis=1)
    return X, y, scaler


def get_data(config, test=False, scale=True):
    cont_idx_list = list(range(8))
    non_cont_idx_list = [wind_dir_idx, weather_idx]
    for i in non_cont_idx_list:
        cont_idx_list.remove(i)

    train_data = pd.read_csv(
        f"{root}/AAAI22/data/time_feature/fold{config['fold']}/train_data_mar_nsgp.csv.gz"
    )
    X_train, y_train, scaler = preprocess(train_data, scale=scale)
    if test:
        test_data = pd.read_csv(
            f"{root}/AAAI22/data/time_feature/fold{config['fold']}/test_data_mar_nsgp.csv.gz"
        )
        X_test, y_test, _ = preprocess(test_data, scaler, scale=scale)
        return (
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            cont_idx_list,
            non_cont_idx_list,
        )
    return X_train, y_train


def get_model(config, X_train):
    wind_dir_kernel = gpk.Hamming(X_train, active_dims=[wind_dir_idx])
    weather_kernel = gpk.Hamming(X_train, active_dims=[weather_idx])
    if config["name"].startswith("egp"):
        time_kernel = gpk.Matern32(X_train, active_dims=[time_idx]) * gpk.Periodic(
            X_train, active_dims=[time_idx]
        )
        cont_idx_list = list(range(8))
        for i in [wind_dir_idx, weather_idx, time_idx]:
            cont_idx_list.remove(i)

        cont_kernel = gpk.Matern32(X_train, active_dims=cont_idx_list)

        kernel = gpk.Scale(
            X_train, cont_kernel * wind_dir_kernel * weather_kernel * time_kernel
        )

        # latent_kernel = gpk.Scale(X_inducing, gpk.RBF(X_inducing, lengthscale=0.3), variance=1.0).trainable(False)
        # latent_model = LatentModel(X_inducing, latent_kernel)
        model = ExactGPRegression(kernel, gpl.Gaussian(), gpm.Average())
        return model

    if config["name"] == "nsegp":
        latent_model_names = {
            "heinonen": LatentGPHeinonen,
            "plagemann": LatentGPPlagemann,
            "delta": LatentGPDeltaInducing,
            "gaussian": LatentGPGaussianBasis,
        }

        LatentModel = latent_model_names[config["latent_model"]]
        X_inducing = X_train if LatentModel is LatentGPHeinonen else X_inducing

        latent_kernel = gpk.Scale(
            X_inducing,
            gpk.RBF(
                X_inducing, lengthscale=0.2, active_dims=cont_idx_list + [time_idx]
            ),
            variance=1.0,
        ).trainable(False)
        latent_model = LatentModel(X_inducing, latent_kernel)

        kernel = gpk.InputDependentScale(
            X_inducing,
            cont_kernel * wind_dir_kernel * weather_kernel * time_kernel,
            latent_model,
        )

        latent_kernel = gpk.Scale(
            X_inducing, gpk.RBF(X_inducing, lengthscale=0.3), variance=1.0
        ).trainable(False)
        latent_model = LatentModel(X_inducing, latent_kernel)
