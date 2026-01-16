import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from scipy.sparse import csr_matrix
import pickle
from scipy.sparse import load_npz
from HybridSVD import HybridSVD, Recommend_topN
from Data import scale_sparse_matrix, scale_R_for_popularity
from Benchmark import hitrate, compute_coverage, compute_NDCG
import matplotlib.pyplot as plt
import os

def baselines(output_dir, Dataset, Similarity, N, seed):
    Z_path = os.path.join("Datasets", Dataset, f"Item-Item Similarity Matrices Z/{Similarity} Similarity Matrix Z.npz")
    R_train1_path = os.path.join("Datasets", Dataset,
                                 "User-Item Interaction Matrices R/R train1 user-item interactions.npz")
    R_test_path = os.path.join("Datasets", Dataset,
                               "User-Item Interaction Matrices R/R test user-item interactions.npz")

    Z = load_npz(Z_path)

    R_train1 = scale_sparse_matrix(load_npz(R_train1_path))
    R_test = scale_sparse_matrix(load_npz(R_test_path))

    #### GENERATING BASELINES #####
    topN_random = Recommend_topN_random(R_train1, N, seed)
    topN_MostPopular = Recommend_topN_MostPopular(R_train1, N)

    # PureSVD
    PureSVD_Optimal_Parameters_Path = os.path.join("Datasets", Dataset,
                                                   "PureSVD Optimal Parameters/PureSVD best params.pkl")
    with open(PureSVD_Optimal_Parameters_Path, "rb") as f:
        PureSVD_Optimal_Parameters_Path = pickle.load(f)
    d_PureSVD = PureSVD_Optimal_Parameters_Path["d"]
    alpha_PureSVD = PureSVD_Optimal_Parameters_Path["alpha"]
    k_PureSVD = PureSVD_Optimal_Parameters_Path["k"]

    R_train1_scaled_PureSVD, _ = scale_R_for_popularity(R_train1, d_PureSVD)
    R_test_scaled_PureSVD, _ = scale_R_for_popularity(R_test, d_PureSVD)

    Sigma_PureSVD, U_tilde_PureSVD, Vt_tilde_PureSVD, Ls_PureSVD, Lk_PureSVD, train_timePureSVD = HybridSVD(k=k_PureSVD,
                                                                                                            alpha=alpha_PureSVD,
                                                                                                            beta=0, Z=Z,
                                                                                                            R=R_train1_scaled_PureSVD)
    topN_PureSVD, rec_timePureSVD = Recommend_topN(R_train1_scaled_PureSVD, Vt_tilde_PureSVD, Ls_PureSVD, N)

    #### EVALUATING BASELINES #####
    hit_at_N_random, hits_random, num_users_random = hitrate(topN_random, R_test)
    coverage_random, num_rec_items_random = compute_coverage(topN_random, num_items=R_train1.shape[1])
    NDCG_at_N_random = compute_NDCG(topN_random, R_test)

    print(f'Random Metrics: Hit@{N}: {hit_at_N_random}, NDCG@{N}: {NDCG_at_N_random}, Coverage: {coverage_random}')

    # 2. CHANGED: Save Random to pickle instead of returning
    random_results = {
        "Hit": hit_at_N_random,
        "NDCG": NDCG_at_N_random,
        "Coverage": coverage_random
    }
    with open(os.path.join(output_dir, "Random_results.pkl"), "wb") as f:
        pickle.dump(random_results, f)

    hit_at_N_MostPop, hits_MostPop, num_users_MostPop = hitrate(topN_MostPopular, R_test)
    coverage_MostPop, num_rec_items_MostPop = compute_coverage(topN_MostPopular, num_items=R_train1.shape[1])
    NDCG_at_N_MostPop = compute_NDCG(topN_MostPopular, R_test)

    print(
        f'MostPopular Metrics: Hit@{N}: {hit_at_N_MostPop}, NDCG@{N}: {NDCG_at_N_MostPop}, Coverage: {coverage_MostPop}')

    # 3. CHANGED: Save MostPopular to pickle
    pop_results = {
        "Hit": hit_at_N_MostPop,
        "NDCG": NDCG_at_N_MostPop,
        "Coverage": coverage_MostPop
    }
    with open(os.path.join(output_dir, "MostPopular_results.pkl"), "wb") as f:
        pickle.dump(pop_results, f)

    hit_at_N_PureSVD, hits_PureSVD, num_users_PureSVD = hitrate(topN_PureSVD, R_test)
    coverage_PureSVD, num_rec_items_PureSVD = compute_coverage(topN_PureSVD, num_items=R_train1.shape[1])
    NDCG_at_N_PureSVD = compute_NDCG(topN_PureSVD, R_test)

    print(f'PureSVD Metrics: Hit@{N}: {hit_at_N_PureSVD}, NDCG@{N}: {NDCG_at_N_PureSVD}, Coverage: {coverage_PureSVD}')

    # 4. CHANGED: Save PureSVD to pickle
    puresvd_results = {
        "Hit": hit_at_N_PureSVD,
        "NDCG": NDCG_at_N_PureSVD,
        "Coverage": coverage_PureSVD,
        "k": k_PureSVD,
        "alpha": alpha_PureSVD,
        "d": d_PureSVD
    }
    with open(os.path.join(output_dir, "PureSVD_results.pkl"), "wb") as f:
        pickle.dump(puresvd_results, f)

    print("Saved baseline results to pickle files in", output_dir)
    return

def Recommend_topN_MostPopular(R_train, N):
    """
    Generate Top-N most popular recommendations for each user.
    Ensures items already interacted with are excluded.

    Parameters:
    -----------
    R_train : scipy sparse matrix (user × item)
        The training interaction matrix.
    N : int
        Number of items to recommend.

    Returns:
    --------
    all_topN : numpy array, shape (num_users, N)
        Top-N recommended item IDs for each user.
    """

    num_users, num_items = R_train.shape

    # ---- 1. Compute item popularity ----
    # popularity[i] = number of users who interacted with item i
    popularity = np.asarray(R_train.sum(axis=0)).ravel()

    # Sort items by decreasing popularity
    sorted_items = np.argsort(-popularity)

    # ---- 2. Recommend most popular unseen items per user ----
    TopN_MostPopular = np.zeros((num_users, N), dtype=int)

    for u in range(num_users):

        # Items user has already interacted with
        seen_items = set(R_train[u].nonzero()[1])

        # Filter out seen items while preserving sorted order
        candidates = [item for item in sorted_items if item not in seen_items]

        # Select the top-N unseen popular items
        TopN_MostPopular[u] = candidates[:N]

    return TopN_MostPopular

import numpy as np

# Random Top-N Recommender System
def Recommend_topN_random(R_train, N, seed):
    """
    Generate random Top-N recommendations for each user.
    Ensures items already interacted with are excluded.

    Parameters:
    -----------
    R_train : scipy sparse matrix (user × item)
        The training interaction matrix.
    N : int
        Number of items to recommend.
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    all_topN : numpy array, shape (num_users, N)
        Top-N recommended item IDs for each user.
    """

    np.random.seed(seed)

    num_users, num_items = R_train.shape
    topN_random = np.zeros((num_users, N), dtype=int)

    for u in range(num_users):

        # Items already seen by user (non-zero entries)
        seen_items = R_train[u].nonzero()[1]

        # Pool of candidate items = all items except seen ones
        candidates = np.setdiff1d(np.arange(num_items), seen_items, assume_unique=True)

        # Sample N random unseen items
        chosen = np.random.choice(candidates, size=N, replace=False)

        topN_random[u] = chosen

    return topN_random

#############################################################################################



def generate_baseline_plot(baseline_results, N, Dataset, output_dir):
    """
    Plots baseline comparison bar charts for Hit@N, NDCG@N, and Coverage.
    The order of bars on the X-axis is determined by the order of keys in baseline_results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data (Respects the insertion order of the dictionary)
    models = list(baseline_results.keys())
    hits = [baseline_results[m]["Hit"] for m in models]
    ndcgs = [baseline_results[m]["NDCG"] for m in models]
    coverages = [baseline_results[m]["Coverage"] for m in models]

    # -------------------------------
    # Plot Hit@N
    # -------------------------------
    plt.figure(figsize=(8, 5))  # Slightly wider to accommodate labels
    plt.bar(models, hits)
    plt.ylabel(f"Hit@{N}")
    plt.title(f"Baseline Comparison {Dataset} — Hit@{N}")
    plt.ylim(0, max(hits) * 1.2)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    hit_path = os.path.join(output_dir, f"All Baselines Hit@{N}.jpg")
    plt.savefig(hit_path, dpi=300)
    plt.show()

    # -------------------------------
    # Plot NDCG@N
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(models, ndcgs, color='orange')
    plt.ylabel(f"NDCG@{N}")
    plt.title(f"Baseline Comparison {Dataset} — NDCG@{N}")
    plt.ylim(0, max(ndcgs) * 1.2)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    ndcg_path = os.path.join(output_dir, f"All Baselines NDCG@{N}.jpg")
    plt.savefig(ndcg_path, dpi=300)
    plt.show()

    # -------------------------------
    # Plot Coverage
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(models, coverages, color='green')
    plt.ylabel("Coverage")
    plt.title(f"Baseline Comparison {Dataset} — Coverage")
    plt.ylim(0, max(coverages) * 1.2)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    cov_path = os.path.join(output_dir, "All Baselines Coverage.jpg")
    plt.savefig(cov_path, dpi=300)
    plt.show()

    print("Saved baseline plots:")
    print(" -", hit_path)
    print(" -", ndcg_path)
    print(" -", cov_path)


def save_and_print_metrics_table(baseline_results, Dataset, output_dir):
    """
    Converts the results dictionary to a Pandas DataFrame,
    prints it to the console, and saves it as a CSV.
    """
    # 1. Convert Dictionary to DataFrame
    df = pd.DataFrame.from_dict(baseline_results, orient='index')

    # 2. Clean up the DataFrame
    df.index.name = 'Model'
    df.reset_index(inplace=True)

    # 3. Print the Table
    print("\n" + "=" * 50)
    print(f"METRICS TABLE: {Dataset}")
    print("=" * 50)
    # Display with 4 decimal places for readability
    print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("=" * 50 + "\n")

    # 4. Save to CSV
    csv_filename = f"{Dataset}_metrics_comparison.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    df.to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")


def baselines_results(Dataset, Similarity, N, output_dir):
    """
    Loads pre-calculated metrics from pickle files, generates comparison plots,
    and saves a summary table.
    """

    results_dir = output_dir

    # --- 1. Load Standard Baselines ---
    def load_metrics(filename):
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: Could not find {filename} at {path}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    random_data = load_metrics("Random_results.pkl")
    pop_data = load_metrics("MostPopular_results.pkl")
    svd_data = load_metrics("PureSVD_results.pkl")

    if not (random_data and pop_data and svd_data):
        print(f"Error: Standard baseline files not found in {results_dir}.")
        return

    # Dictionary Insertion Order determines Plot Order
    baseline_results = {
        "Random": {
            "Hit": random_data["Hit"],
            "NDCG": random_data["NDCG"],
            "Coverage": random_data["Coverage"]
        },
        "MostPopular": {
            "Hit": pop_data["Hit"],
            "NDCG": pop_data["NDCG"],
            "Coverage": pop_data["Coverage"]
        },
        f"PureSVD {svd_data['d']:.2f}d {svd_data['alpha']:.2f}α {svd_data['k']}k": {
            "Hit": svd_data["Hit"],
            "NDCG": svd_data["NDCG"],
            "Coverage": svd_data["Coverage"]
        }
    }

    # --- 2. Load HybridSVD Results ---
    if Dataset == "DC Core 20":
        # CHANGED: Order is now Category -> Description -> Geolocation
        sim_types = ["Category", "Description", "Geolocation"]

        for sim in sim_types:
            base_folder = f"outputs/{Dataset} {sim}"
            params_path = os.path.join(base_folder, f"{Dataset} {sim} best params.pkl")
            results_path = os.path.join(base_folder, f"{Dataset} {sim} test results.pkl")

            if os.path.exists(params_path) and os.path.exists(results_path):
                with open(params_path, "rb") as f:
                    params = pickle.load(f)
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                label = f"HybridSVD {sim} {params['d']:.2f}d {params['alpha']:.2f}α {params['k']}k"
                baseline_results[label] = {
                    "Hit": results[f"Hit@{N}"],
                    "NDCG": results[f"NDCG@{N}"],
                    "Coverage": results["Coverage"]
                }

    elif Dataset == "ml-100k":
        params_filename = f"{Dataset} {Similarity} best params.pkl"
        results_filename = f"{Dataset} {Similarity} test results.pkl"
        params_path = os.path.join(output_dir, params_filename)
        results_path = os.path.join(output_dir, results_filename)

        if os.path.exists(params_path) and os.path.exists(results_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            with open(results_path, "rb") as f:
                results = pickle.load(f)

            label = f"HybridSVD {Similarity} {params['d']:.2f}d {params['alpha']:.2f}α {params['k']}k"
            baseline_results[label] = {
                "Hit": results[f"Hit@{N}"],
                "NDCG": results[f"NDCG@{N}"],
                "Coverage": results["Coverage"]
            }

    # --- 3. Generate Output ---

    # A. Generate the Bar Charts (Will follow the order defined above)
    generate_baseline_plot(baseline_results, N, Dataset, output_dir)

    # B. Generate the CSV Table
    save_and_print_metrics_table(baseline_results, Dataset, output_dir)
