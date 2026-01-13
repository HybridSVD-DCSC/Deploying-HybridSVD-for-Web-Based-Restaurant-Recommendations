import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import eye, save_npz
from scipy.sparse.linalg import svds, spsolve_triangular
import os
import time

def HybridSVD(k, alpha, Z, R, beta):
    t0=time.time()
    n = R.shape[0]  # users

    m = R.shape[1]  # items

    # Creating sparse item-item similarity matrix
    # Z = eye(m, format='csc')
    I = eye(m, format='csc')
    S = (1 - alpha) * I + alpha * Z  # both are sparse CSC

    # # sparse user-user similarity matrix
    Y = eye(n, format='csc')
    I = eye(n, format='csc')
    K = (1 - beta) * I + beta * Y  # both are sparse CSC

    # Cholesky decomposition
    factor = cholesky(S)
    Ls = factor.L()

    factor = cholesky(K)
    Lk = factor.L()

    # Hybrid SVD training
    R_tilde = Lk.T @ R @ Ls

    # Running ordinary SVD on our fold in
    U_tilde, Sigma, Vt_tilde = svds(R_tilde, k=k)

    train_time = time.time() - t0
    return Sigma, U_tilde, Vt_tilde, Ls, Lk, train_time

def Save_HyrbidSVD_output(R_train1, R_test, Ls, Vt_tilde, U_tilde, output_dir):
    """
    Prints shapes of key matrices and saves Ls (.npz) and Vt_tilde (.npy).

    Parameters
    ----------
    R_train1 : sparse matrix
        Final training interaction matrix
    R_test : sparse matrix
        Test matrix
    Ls : sparse matrix
        Cholesky factor of item-item similarity matrix
    Vt_tilde : ndarray
        Right singular vectors (tilde) matrix from HybridSVD
    U_tilde : ndarray
        Left singular vectors (tilde)
    output_dir : str
        Directory where files are saved
    """

    # --- Ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Print shapes ---
    print("R_train1 shape:", R_train1.shape)
    print("R_test shape:", R_test.shape)
    print("p shape:", R_train1[1].shape)

    print("Ls shape:", Ls.shape)
    print("Ls.T shape:", Ls.T.shape)

    print("Vt_tilde shape:", Vt_tilde.shape)
    print("U_tilde shape:", U_tilde.shape)

    # --- Saving files ---
    ls_path = os.path.join(output_dir, "Ls.npz")
    vt_path = os.path.join(output_dir, "Vt_tilde.npy")

    save_npz(ls_path, Ls)                 # sparse => .npz
    np.save(vt_path, Vt_tilde)            # dense => .npy

    print(f"Saved Ls to: {ls_path}")
    print(f"Saved Vt_tilde to: {vt_path}")
    print(f"All matrices saved successfully to {output_dir}.")

def Recommend_topN(R_train, Vt_tilde, Ls, N):
    t0 = time.time()

    num_users, num_items = R_train.shape
    V = Vt_tilde.T          # (m, k)
    Ls = Ls.tocsc()         # sparse, required

    topN_matrix = np.zeros((num_users, N), dtype=np.int32)

    for user in range(num_users):
        p = R_train[user].toarray().ravel()

        # HybridSVD scoring
        x = Ls.T @ p
        scores = V @ (V.T @ x)
        scores = spsolve_triangular(Ls.T, scores, lower=False)

        # mask seen items
        scores[p > 0] = -np.inf

        # Get top-N (ordered)
        topN = np.argsort(-scores)[:N]
        topN_matrix[user] = topN

    rec_time = time.time() - t0
    return topN_matrix, rec_time
