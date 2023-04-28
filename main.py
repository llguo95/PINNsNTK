
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


def u(x, a):
    return np.sin(np.pi * a * x)


def u_xx(x, a):
    return -(np.pi * a)**2 * np.sin(np.pi * a * x)


def main(
        train_algo=None, 
        visualize=None, 
        SMOKE_TEST=False, 
        iteration=None,
        noisy_data=None,
        regularization=None,
        df=None,
        df_errors=None,
        seed=None,
        ):
    # Define solution and its Laplace
    a = 4

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

    
    if noisy_data:
        np.random.seed(seed)
        Y_r += np.random.randn(*Y_r.shape)

    model = PINN(layers, X_u, Y_u, X_r, Y_r, train_algo=train_algo, regularization=regularization)    


    # Train model
    if SMOKE_TEST:
        nIter = 401
    else:
        nIter = 40001

    model.train(nIter=nIter, batch_size=100, log_NTK=True, log_weights=True)

    
    # **Training Loss**


    loss_bcs = model.loss_bcs_log
    loss_res = model.loss_res_log

    if "losses" in visualize:
        fig = plt.figure(figsize=(6,5))
        plt.plot(loss_res, label='$\mathcal{L}_{r}$')
        plt.plot(loss_bcs, label='$\mathcal{L}_{b}$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

    
    # **Model Prediction**


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

    if df_errors is not None:
        # df_errors[train_algo._name] = [error_u, error_r]
        df_errors[regularization] = [error_u, error_r]

    if "prediction" in visualize:
        fig = plt.figure(num="prediction", figsize=(12, 5))
        plt.subplot(1,2,1)
        if iteration == 0:
            plt.plot(X_star, u_star, 'k')
        plt.plot(X_star, u_pred, '--', label='Predicted')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        # plt.legend(loc='upper right')

        plt.subplot(1,2,2)
        plt.plot(X_star, np.abs(u_star - u_pred), label='Error')
        plt.yscale('log')
        plt.xlabel('$x$')
        plt.ylabel('Point-wise error')
        plt.tight_layout()

    
    # **NTK Eigenvalues**


    lambda_K_log, lambda_K_uu_log, lambda_K_rr_log, K_list = compute_ntk_eigenvalues(model)

    if "eigenvalues" in visualize:
        fig = plt.figure(num="eigenvalues", figsize=(18, 5))
        plt.subplot(1,3,1)
        # for i in range(1, len(lambda_K_log), 10):
        plt.plot(lambda_K_log[-1], '--')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r'Eigenvalues of ${K}$')
        plt.xlabel('Eigenvalue no.')
        plt.ylabel('Magnitude')
        plt.tight_layout()

        plt.subplot(1,3,2)
        # for i in range(1, len(lambda_K_uu_log), 10):
        plt.plot(lambda_K_uu_log[-1], '--')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r'Eigenvalues of ${K}_{uu}$')
        plt.xlabel('Eigenvalue no.')
        plt.ylabel('Magnitude')
        plt.tight_layout()

        plt.subplot(1,3,3)
        # for i in range(1, len(lambda_K_log), 10):
        plt.plot(lambda_K_rr_log[-1], '--')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r'Eigenvalues of ${K}_{rr}$')
        plt.xlabel('Eigenvalue no.')
        plt.ylabel('Magnitude')
        plt.tight_layout()

    
    # **Change of NTK**


    # Change of the NTK
    NTK_change_list = []
    K0 = K_list[0]
    for K in K_list:
        diff = np.linalg.norm(K - K0) / np.linalg.norm(K0) * 100
        NTK_change_list.append(diff)

    if df is not None:
        df[noisy_data] = NTK_change_list

    if "convergence" in visualize:
        fig = plt.figure(num='convergence', figsize=(6,5))
        plt.plot(NTK_change_list)
        plt.xlabel('Iteration no. (x100)')
        plt.ylabel('Relative change in magnitude (%)')

    
    # 
    # **Change of NN Params**


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

    if "convergence_weights" in visualize:
        fig = plt.figure(figsize=(6,5))
        plt.plot(weights_change_list)

    return df

if __name__ == '__main__':
    main()

