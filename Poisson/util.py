import numpy as np

def compute_ntk_eigenvalues(model):
    # Create empty lists for storing the eigenvalues of NTK
    lambda_K_log = []
    lambda_K_uu_log = []
    lambda_K_ur_log = []
    lambda_K_rr_log = []

    # Restore the NTK
    K_uu_list = model.K_uu_log
    K_ur_list = model.K_ur_log
    K_rr_list = model.K_rr_log
    K_list = []
        
    for k in range(len(K_uu_list)):
        K_uu = K_uu_list[k]
        K_ur = K_ur_list[k]
        K_rr = K_rr_list[k]
        
        K = np.concatenate([np.concatenate([K_uu, K_ur], axis = 1),
                            np.concatenate([K_ur.T, K_rr], axis = 1)], axis = 0)
        K_list.append(K)

        # Compute eigenvalues
        lambda_K, _ = np.linalg.eig(K)
        lambda_K_uu, _ = np.linalg.eig(K_uu)
        lambda_K_rr, _ = np.linalg.eig(K_rr)
        
        # Sort in descresing order
        lambda_K = np.sort(np.real(lambda_K))[::-1]
        lambda_K_uu = np.sort(np.real(lambda_K_uu))[::-1]
        lambda_K_rr = np.sort(np.real(lambda_K_rr))[::-1]
        
        # Store eigenvalues
        lambda_K_log.append(lambda_K)
        lambda_K_uu_log.append(lambda_K_uu)
        lambda_K_rr_log.append(lambda_K_rr)

    return lambda_K_log, lambda_K_uu_log, lambda_K_rr_log, K_list