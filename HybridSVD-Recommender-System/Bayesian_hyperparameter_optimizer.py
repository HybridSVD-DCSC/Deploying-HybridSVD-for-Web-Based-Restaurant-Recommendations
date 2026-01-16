import os
import random
import numpy as np
import pandas as pd
import optuna
from scipy.sparse import load_npz
import pickle
import matplotlib.pyplot as plt
from Data import scale_sparse_matrix, scale_R_for_popularity  # your original scaling functions
from HybridSVD import HybridSVD, Recommend_topN
from Benchmark import hitrate, compute_coverage, compute_NDCG

def run_optuna_hybridsvd(
    N,
    Dataset,
    Similarity,
    n_trials,
    seed,
    output_dir):
    """
    Bayesian optimization of HybridSVD hyperparameters (k, alpha, d) to maximize NDCG.
    d affects popularity scaling of R_train and R_val.
    """

    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Construct paths
    Z_path = os.path.join("Datasets", Dataset, f"Item-Item Similarity Matrices Z/{Similarity} Similarity Matrix Z.npz")
    R_train_path = os.path.join("Datasets", Dataset, "User-Item Interaction Matrices R/R train user-item interactions.npz")
    R_val_path = os.path.join("Datasets", Dataset, "User-Item Interaction Matrices R/R validate user-item interactions.npz")
    R_train1_path = os.path.join("Datasets", Dataset, "User-Item Interaction Matrices R/R train1 user-item interactions.npz")  # TRAIN+VAL
    R_test_path = os.path.join("Datasets", Dataset, "User-Item Interaction Matrices R/R test user-item interactions.npz")

    # Load Z once
    Z = load_npz(Z_path)

    # Scaling R matrices from [0,5] to [0,1]
    R_train = scale_sparse_matrix(load_npz(R_train_path))
    R_val = scale_sparse_matrix(load_npz(R_val_path))
    R_train1 = scale_sparse_matrix(load_npz(R_train1_path))
    R_test = scale_sparse_matrix(load_npz(R_test_path))


    # ------------------------------
    # Inner evaluation function
    # ------------------------------
    def evaluate(k, alpha, R_train_scaled, R_val_scaled):

        # Train model on scaled R_train
        Sigma, U_tilde, Vt_tilde, Ls, Lk, train_time = HybridSVD(k, alpha, Z, R=R_train_scaled, beta=0)

        # Generate Top-N recommendations
        topN_matrix, rec_time = Recommend_topN(R_train_scaled, Vt_tilde, Ls, N)
        hit = hitrate(topN_matrix, R_val_scaled)
        ndcg = compute_NDCG(topN_matrix, R_val_scaled)
        print(f"NDCG: {ndcg}")
        return hit, ndcg, train_time, rec_time

    # ------------------------------
    # Optuna objective
    # ------------------------------
    def objective(trial):
        k = trial.suggest_int("k", 5, 300)
        alpha = trial.suggest_float("alpha", 0.0, 1.0, step=None)
        d = trial.suggest_float("d", 0, 1.0)

        # Scaling R matrices for popularity according to scaling factor d
        R_train_scaled, _ = scale_R_for_popularity(R_train, d=d)
        R_val_scaled, _ = scale_R_for_popularity(R_val, d=d)

        hit, ndcg, train_time, rec_time = evaluate(k, alpha, R_train_scaled, R_val_scaled)
        trial.set_user_attr("ndcg", ndcg)
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("rec_time", rec_time)
        return ndcg  # maximize NDCG

    # ------------------------------
    # Run Optuna optimization
    # ------------------------------
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    results_df = study.trials_dataframe()
    results_df["Dataset"] = Dataset
    results_df["Similarity"] = Similarity
    results_df["# of users"] = R_train.shape[0]
    results_df["N"] = N
    results_df.to_csv(os.path.join(output_dir, f"{Dataset}_{Similarity}_OPTUNA_Gridsearch_results.csv"), index=False)

    # ------------------------------
    # Final evaluation on TEST using R_train1 and R_test
    # ------------------------------
    # Apply best d scaling to R_train1 and R_test
    R_train1_scaled, _ = scale_R_for_popularity(R_train1, d=best_params["d"])
    R_test_scaled, _ = scale_R_for_popularity(R_test, d=best_params["d"])

    # Training optimal HybridSVD model on R_train1
    Sigma, U_tilde, Vt_tilde, Ls, Lk, train_time = HybridSVD(
        k=best_params["k"],
        alpha=best_params["alpha"],
        Z=Z,
        R=R_train1_scaled,
        beta=0
    )

    topN_matrix, rec_time = Recommend_topN(R_train1_scaled, Vt_tilde, Ls, N)
    hit_test, hits, num_users = hitrate(topN_matrix, R_test_scaled)
    ndcg_test = compute_NDCG(topN_matrix, R_test_scaled)
    coverage_test, num_rec_items = compute_coverage(topN_matrix, R_test.shape[1])

    test_results = {
        f"Hit@{N}": hit_test,
        f"Hits": hits,
        f"# of users": num_users,
        f"NDCG@{N}": ndcg_test,
        f"Coverage": coverage_test,
        f"# of unique recommended items": num_rec_items,
        "Train_time_s": train_time,
        "Rec_time_s": rec_time,
    }
    # Saving results
    best_params_path = os.path.join(output_dir, f"{Dataset} {Similarity} best params.pkl")
    results_df_path = os.path.join(output_dir, f"{Dataset} {Similarity} optuna results_df.pkl")
    test_results_path = os.path.join(output_dir, f"{Dataset} {Similarity} test results.pkl")

    with open(best_params_path, "wb") as f:
        pickle.dump(best_params, f)

    with open(results_df_path, "wb") as f:
        pickle.dump(results_df, f)

    with open(test_results_path, "wb") as f:
        pickle.dump(test_results, f)

    topN_matrix_path = os.path.join(output_dir, f"{Dataset} {Similarity} TopN matrix.csv")
    pd.DataFrame(topN_matrix).to_csv(topN_matrix_path, index=False)
    print("Best Hyperparameters:")
    print(best_params)
    print("Results on TEST set:")
    print(test_results)
    print(f"Saved results to:{output_dir}")
    return

def plot_optuna(Dataset, Similarity, N, output_dir):
    # Loading gridsearch results
    best_params_path = os.path.join(output_dir, f"{Dataset} {Similarity} best params.pkl")
    results_df_path = os.path.join(output_dir, f"{Dataset} {Similarity} optuna results_df.pkl")
    test_results_path = os.path.join(output_dir, f"{Dataset} {Similarity} test results.pkl")

    # Load
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    with open(results_df_path, "rb") as f:
        Gridsearch_df = pickle.load(f)

    with open(test_results_path, "rb") as f:
        test_results = pickle.load(f)

    print("Best Hyperparameters:")
    print(best_params)
    print("Test Results:")
    print(test_results)

    # Plotting Gridsearch Results
    plt.plot(Gridsearch_df.iloc[:, 0], Gridsearch_df.iloc[:, 1])
    plt.title(f"Bayesian optimization of NDCG@{N} over trials for {Dataset} with {Similarity} similarity")
    plt.xlabel(f"Trial #")
    plt.ylabel(f"NDCG@{N}")
    plt.savefig(f"{output_dir}/PLOT Bayesian Hyperparameter Optimization {Dataset} {Similarity}.png", dpi=300,
                bbox_inches="tight")
    plt.show()

    return

