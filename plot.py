import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_bcs, loss_res):
    fig = plt.figure(figsize=(6,5))
    plt.plot(loss_res, label='$\mathcal{L}_{r}$')
    plt.plot(loss_bcs, label='$\mathcal{L}_{b}$')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

def plot_resulting_func(X_star, u_star, u_pred):
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(X_star, u_star, label='Exact')
    plt.plot(X_star, u_pred, '--', label='Predicted')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(X_star, np.abs(u_star - u_pred), label='Error')
    plt.yscale('log')
    plt.xlabel('$x$')
    plt.ylabel('Point-wise error')
    plt.tight_layout()
    plt.savefig("test.png")

def plot_eigenvalues(lambda_K_log, lambda_K_uu_log, lambda_K_rr_log):
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1,3,1)
    for i in range(1, len(lambda_K_log), 10):
        plt.plot(lambda_K_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${K}$')
    plt.tight_layout()

    plt.subplot(1,3,2)
    for i in range(1, len(lambda_K_uu_log), 10):
        plt.plot(lambda_K_uu_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${K}_{uu}$')
    plt.tight_layout()

    plt.subplot(1,3,3)
    for i in range(1, len(lambda_K_log), 10):
        plt.plot(lambda_K_rr_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${K}_{rr}$')
    plt.tight_layout()
    plt.savefig("test_3.png")

def plot_ntk_changes(K_list):
    NTK_change_list = []
    K0 = K_list[0]
    for K in K_list:
        diff = np.linalg.norm(K - K0) / np.linalg.norm(K0) 
        NTK_change_list.append(diff)

    fig = plt.figure(figsize=(6,5))
    plt.plot(NTK_change_list)

def plot_weights_change(weights_change_list):
    fig = plt.figure(figsize=(6,5))
    plt.title("Change of NN Params")
    plt.plot(weights_change_list)
    plt.savefig("test_2.png")

