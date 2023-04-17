#!/usr/bin/env python

import tensorflow as tf
from Compute_Jacobian import jacobian # Please download 'Compute_Jacobian.py' in the repository 
import numpy as np
import timeit
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from pinn import PINN
from util import compute_ntk_eigenvalues
from plot import *



# Define solution and its Laplace
a = 4

def u(x, a):
  return np.sin(np.pi * a * x)

def u_xx(x, a):
  return -(np.pi * a)**2 * np.sin(np.pi * a * x)

model_dict ={}


def plot_all(model_dict, fig_folder="data"):
    """Computes eigenvalues and plots all figures
    slightly inneficient since most figures would allready be generated in 
    previous calls to plot_model.
    """
    eigen_dict = {}
    for name, model in model_dict.items():
        print(name)
        eigen_dict[name] = plot_model(name, model, fig_folder=fig_folder)
    plot_all_combined_eigenvalues(eigen_dict)
    savefig(f"{fig_folder}/5")
    plt.show()

def plot_model(name, model, fig_folder="data"):
    os.makedirs(f"{fig_folder}/{name}", exist_ok=True)
    plot_loss(model.loss_bcs_log, model.loss_res_log)
    # plt.title(name)
    savefig(f"{fig_folder}/{name}/0")


    nn = 1000
    X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    u_star = u(X_star, a)
    r_star = u_xx(X_star, a)

    # Predictions
    u_pred = model.predict_u(X_star)
    r_pred = model.predict_r(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_r: {:.2e}'.format(error_r))



    plot_resulting_func(X_star, u_star, u_pred)
    # plt.title(name)
    savefig(f"{fig_folder}/{name}/1")


    lambda_K_log, lambda_K_uu_log, lambda_K_rr_log, K_list = compute_ntk_eigenvalues(model)


    plot_eigenvalues(lambda_K_log, lambda_K_uu_log, lambda_K_rr_log)
    # plt.title(name)
    savefig(f"{fig_folder}/{name}/2")



    plot_ntk_changes(K_list)
    # plt.title(name)
    savefig(f"{fig_folder}/{name}/3")



    # Change of the weights and biases
    def compute_weights_diff(weights_1, weights_2):
        weights = []
        N = len(weights_1)
        for k in range(N):
            weight = weights_1[k] - weights_2[k]
            weights.append(weight)
        return weights

    def compute_weights_norm(weights, biases):
        norm = 0
        for w in weights:
            norm = norm + np.sum(np.square(w))
        for b in biases:
            norm = norm + np.sum(np.square(b))
        norm = np.sqrt(norm)
        return norm

    # Restore the list weights and biases
    weights_log = model.weights_log
    biases_log = model.biases_log

    weights_0 = weights_log[0]
    biases_0 = biases_log[0]

    # Norm of the weights at initialization
    weights_init_norm = compute_weights_norm(weights_0, biases_0)

    weights_change_list = []

    N = len(weights_log)
    for k in range(N):
        weights_diff = compute_weights_diff(weights_log[k], weights_log[0])
        biases_diff = compute_weights_diff(biases_log[k], biases_log[0])
        
        weights_diff_norm = compute_weights_norm(weights_diff, biases_diff)
        weights_change = weights_diff_norm / weights_init_norm
        weights_change_list.append(weights_change)


    plot_weights_change(weights_change_list)
    # plt.title(name)
    savefig(f"{fig_folder}/{name}/4")
    plt.close("all")
    return lambda_K_log, lambda_K_uu_log, lambda_K_rr_log


for name, activation_function in zip(["tanh"    , "sigmoid"    , "softmax"    , "ReLU"    , "ReLU6"    ], \
                                     [tf.nn.tanh, tf.nn.sigmoid, tf.nn.softmax, tf.nn.relu, tf.nn.relu6]):
    os.makedirs(name, exist_ok=True)
    print("current used activation function:", name)
    # Define computional domain
    bc1_coords = np.array([[0.0],
                        [0.0]])

    bc2_coords = np.array([[1.0],
                        [1.0]])

    dom_coords = np.array([[0.0],
                        [1.0]])

    # Training data on u(x) -- Dirichlet boundary conditions

    nn  = 100

    X_bc1 = dom_coords[0, 0] * np.ones((nn // 2, 1))
    X_bc2 = dom_coords[1, 0] * np.ones((nn // 2, 1))
    X_u = np.vstack([X_bc1, X_bc2])
    Y_u = u(X_u, a)

    X_r = np.linspace(dom_coords[0, 0],
                    dom_coords[1, 0], nn)[:, None]
    Y_r = u_xx(X_r, a)




    # Define model
    layers = [1, 512, 1]  
    # layers = [1, 512, 512, 512, 1]  
    model = PINN(layers, X_u, Y_u, X_r, Y_r, activation_function=activation_function)    



    # Train model
    model.train(nIter=40001, batch_size=100, log_NTK=True, log_weights=True)
    model.activation_func = tf.nn.tanh
    model_dict[name] = model
    plot_model(name, model)


plot_all(model_dict)