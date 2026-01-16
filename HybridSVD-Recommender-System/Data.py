from scipy.sparse import csr_matrix, load_npz, save_npz
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Map_Washington_DC import create_dc_map_from_df

def loading_raw_data(Dataset, output_dir):
    if Dataset == "DC Core 20":
        return load_dc_google_dataset(Dataset, output_dir)
    elif Dataset == "ml-100k":
        return load_movielens_100k_dataset(Dataset, output_dir)
    else:
        raise ValueError(f"Dataset {Dataset} not recognized.")

def load_movielens_100k_dataset(Dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")
    print(f"Loading {Dataset} dataset...")
    # Loading Movielens 100k dataset
    # Ratings
    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    reviews_path = os.path.join("Datasets", Dataset, "u.data")
    reviews = pd.read_csv(
        reviews_path,
        sep="\t",
        names=ratings_cols
    )

    # Items (movies + genres)
    item_cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    meta_path = os.path.join("Datasets", Dataset, "u.item")
    meta = pd.read_csv(
        meta_path,
        sep="|",
        names=item_cols,
        encoding="latin-1"
    )
    # --------------------------------------
    # Save CSV exports
    # --------------------------------------

    reviews_csv = os.path.join(output_dir, "reviews.csv")
    meta_csv = os.path.join(output_dir, "item_metadata.csv")

    reviews.to_csv(reviews_csv, index=False, encoding="utf-8")
    meta.to_csv(meta_csv, index=False, encoding="utf-8")

    print(f"Saved reviews → {output_dir}/{reviews_csv}")
    print(f"Saved metadata → {output_dir}/{meta_csv}")

    return reviews, meta, None, None, None


def load_dc_google_dataset(Dataset, output_dir):
    """
    Loads Google Local Reviews (DC Core 20) dataset, saves CSVs, loads user/item maps,
    and builds a gmap_id → restaurant name lookup dictionary.

    Parameters
    ----------
    Dataset : str
        Name of the processed dataset directory (e.g., "DC_core20").

    Returns
    -------
    reviews : DataFrame
    meta : DataFrame
    user_map : dict
    item_map : dict
    gmap_to_name : dict   # mapping gmap_id → restaurant name
    """
    os.makedirs(output_dir, exist_ok=True)
    # --------------------------------------
    # Load reviews JSONL
    # --------------------------------------
    reviews_path = os.path.join("Datasets", Dataset, "review-District_of_Columbia.json")
    reviews = pd.read_json(reviews_path, lines=True)

    # Load metadata
    meta_path = os.path.join("Datasets", Dataset, "meta-District_of_Columbia.json")
    meta = pd.read_json(meta_path, lines=True)

    # --------------------------------------
    # Save CSV exports
    # --------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    reviews_csv = os.path.join(output_dir, "reviews.csv")
    meta_csv = os.path.join(output_dir, "item_metadata.csv")

    reviews.to_csv(reviews_csv, index=False, encoding="utf-8")
    meta.to_csv(meta_csv, index=False, encoding="utf-8")

    print(f"Saved reviews → {output_dir}/{reviews_csv}")
    print(f"Saved metadata → {output_dir}/{meta_csv}")

    # --------------------------------------
    # Load mapping dictionaries
    # --------------------------------------
    user_map_path = os.path.join("Datasets", Dataset, "Mappings/User Map DCcore20.json")
    item_map_path = os.path.join("Datasets", Dataset, "Mappings/Restaurant Map DCcore20.json")

    with open(user_map_path, "r", encoding="utf-8") as f:
        user_map = json.load(f)

    with open(item_map_path, "r", encoding="utf-8") as f:
        item_map = json.load(f)

    # --------------------------------------
    # Build gmap_id → name mapping
    # --------------------------------------
    meta["gmap_id"] = meta["gmap_id"].astype(str)
    gmap_to_name = dict(zip(meta["gmap_id"], meta["name"]))

    print("\nFirst 3 restaurant mappings:")
    for g in list(gmap_to_name.keys())[:3]:
        print(f"{g} → {gmap_to_name[g]}")

    m=create_dc_map_from_df(meta, output_dir)
    display(m)
    print(f"{Dataset} data loaded succesfully.\n")
    return reviews, meta, user_map, item_map, gmap_to_name


def scale_sparse_matrix(R):
    """
    Scale sparse ratings from [0,5] to [-1,1] without densifying.
    """
    # R = R.tocsr(copy=True)          # ensure CSR format
    # R.data = R.data / 5
    #R.data = 2 * (R.data / 5) - 1   # apply scaling only to stored values

    # Binarization
    R = R.tocsr(copy=True)  # ensure CSR format
    R.data[:] = 1.0  # binarize: any interaction = 1
    return R


def scale_R_for_popularity(R, d, eps=1e-12):
    """
    Column-scale a sparse user–item interaction matrix R using
    R <- R D^{-d}, where D_jj = ||R[:, j]||_2.

    Parameters
    ----------
    R : scipy.sparse.csr_matrix
        User–item interaction matrix (users × items)
    d : float, optional (default=0.4)
        Scaling exponent (0 = full scaling, 1 = no scaling)
    eps : float, optional
        Small constant to avoid division by zero

    Returns
    -------
    R_scaled : scipy.sparse.csr_matrix
        Scaled interaction matrix
    col_norms : np.ndarray
        Original column L2 norms (length = #items)
    """

    # Compute column L2 norms (D_jj = ||R[:, j]||_2)
    col_norms = np.sqrt(R.power(2).sum(axis=0)).A1
    col_norms[col_norms < eps] = eps

    # Paper notation: R <- R D^{d-1}
    scale_factors = col_norms ** (d - 1)

    # Apply scaling without forming dense D
    R_scaled = R @ csr_matrix(np.diag(scale_factors))

    return R_scaled, col_norms

def translate_sampled_users_to_user_ids_DC(sampled_users, user_map):

    """
    Translate internal user indices to original user IDs.

    Parameters
    ----------
    sampled_users : array-like
        Internal user indices.
    user_map : dict
        Mapping {internal_idx (str or int) -> original_user_id}.

    Returns
    -------
    translated_users : list
        Original user IDs corresponding to sampled_users.
    """
    # Ensure integer keys
    user_map_int = {int(k): v for k, v in user_map.items()}

    translated_users = [user_map_int[u] for u in sampled_users]

    #print(f"Sampled user indices preview:{sampled_users[:3]} ")
    #print(f"Translated original user IDs preview: {translated_users[:3]}")

    return translated_users

def translate_item_indices_to_gmapIDs_DC(topN_matrix, item_map, gmap_to_name):
    """
    Translate item indices to Google Maps IDs and restaurant names (DC Core 20 dataset).

    Parameters
    ----------
    topN_matrix : np.ndarray (num_users × N)
        Matrix with item indices.
    item_map : dict
        Mapping {item_idx (str/int) -> gmap_id}.
    gmap_to_name : dict
        Mapping {gmap_id -> restaurant name}.

    Returns
    -------
    topN_restaurant_names : np.ndarray
        Matrix with restaurant names.
    """

    # Convert item_map keys to int
    item_map_int = {int(k): v for k, v in item_map.items()}

    # item_idx → gmap_id
    topN_restaurant_gmapID = np.vectorize(
        lambda idx: item_map_int.get(idx, None)
    )(topN_matrix)

    # gmap_id → restaurant name
    topN_restaurant_names = np.vectorize(
        lambda gmap: gmap_to_name.get(gmap, "UNKNOWN_ITEM")
    )(topN_restaurant_gmapID)
    #print(f"Top-N restaurant gmapID matrix shape and preview:{topN_restaurant_gmapID.shape}{topN_restaurant_gmapID[0,:3]}")
    #print(f"Top-N restaurant name matrix shape and preview:{topN_restaurant_names.shape}{topN_restaurant_names[0,:3]}")

    return topN_restaurant_names, topN_restaurant_gmapID

def create_topN_dataframes_with_userIDs(
    topN_restaurant_names,
    topN_restaurant_gmapID,
    translated_users,
    N,
    output_dir,
    Dataset,
    Similarity
):
    """
    Create and save Top-N DataFrames with user IDs for restaurant names and gmap IDs.

    Parameters
    ----------
    topN_restaurant_names : np.ndarray
        Matrix of restaurant names (num_users × N).
    topN_restaurant_gmapID : np.ndarray
        Matrix of gmap IDs (num_users × N).
    translated_users : list
        List of translated user IDs.
    N : int
        Number of recommendations per user.
    output_dir : str
        Output directory.
    Dataset : str
        Dataset name.
    Similarity : str
        Similarity / model label.
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")
    print(f"Loading {Dataset} dataset...")
    # Column names: Restaurant 1, Restaurant 2, ..., Restaurant N
    name_cols = [f"Restaurant {i+1}" for i in range(N)]

    # Create DataFrames (no index)
    topN_names = pd.DataFrame(topN_restaurant_names, columns=name_cols)
    topN_gmap = pd.DataFrame(topN_restaurant_gmapID, columns=name_cols)

    # Insert user IDs as first column
    topN_names.insert(0, "user_id", translated_users)
    topN_gmap.insert(0, "user_id", translated_users)

    # Save CSVs
    names_path = os.path.join(
        output_dir, f"{Dataset}_{Similarity}_topN_names.csv"
    )
    gmap_path = os.path.join(
        output_dir, f"{Dataset}_{Similarity}_topN_gmap.csv"
    )

    topN_names.to_csv(names_path, index=False, encoding="utf-8")
    topN_gmap.to_csv(gmap_path, index=False, encoding="utf-8")

    print("Saved Top-N files:")
    print(names_path)
    print(gmap_path)

    return topN_names, topN_gmap

def checking_matrix_dimension(Dataset, Similarity):
    Z_path = os.path.join("Datasets", Dataset, f"Item-Item Similarity Matrices Z/{Similarity} Similarity Matrix Z.npz")
    R_train_path = os.path.join("Datasets", Dataset,
                                "User-Item Interaction Matrices R/R train user-item interactions.npz")
    R_val_path = os.path.join("Datasets", Dataset,
                              "User-Item Interaction Matrices R/R validate user-item interactions.npz")
    R_train1_path = os.path.join("Datasets", Dataset,
                                 "User-Item Interaction Matrices R/R train1 user-item interactions.npz")  # TRAIN+VAL
    R_test_path = os.path.join("Datasets", Dataset,
                               "User-Item Interaction Matrices R/R test user-item interactions.npz")

    # Load Z once
    Z = load_npz(Z_path)

    # Scaling R matrices from [0,5] to [0,1]
    R_train = scale_sparse_matrix(load_npz(R_train_path))
    R_val = scale_sparse_matrix(load_npz(R_val_path))
    R_train1 = scale_sparse_matrix(load_npz(R_train1_path))
    R_test = scale_sparse_matrix(load_npz(R_test_path))

    print(f"R_train shape: {R_train.shape}, R_val shape: {R_val.shape}, R_train1 shape: {R_train1.shape}, R_test shape: {R_test.shape}")
    print(f"R_train nnz: {R_train.nnz}, R_val nnz: {R_val.nnz}, R_train1 nnz: {R_train1.nnz}, R_test nnz: {R_test.nnz}")
    print(f"Z shape: {Z.shape}")
    print(f"Matrix Z: {Z}")
    print(f"Matrix R_train1: {R_train1}")
    print(f"Matrix R_test: {R_test}")
    return


def analyze_dataset(Dataset, Similarity, output_dir):
    """
    Loads dataset matrices, calculates key statistics (Sparsity, Density),
    plots the Long Tail distribution of interactions, and saves the plot.
    """

    # 1. Load Matrices using the helper function
    Z_path = os.path.join("Datasets", Dataset, f"Item-Item Similarity Matrices Z/{Similarity} Similarity Matrix Z.npz")
    R_train_path = os.path.join("Datasets", Dataset,
                                "User-Item Interaction Matrices R/R train user-item interactions.npz")
    R_val_path = os.path.join("Datasets", Dataset,
                              "User-Item Interaction Matrices R/R validate user-item interactions.npz")
    R_train1_path = os.path.join("Datasets", Dataset,
                                 "User-Item Interaction Matrices R/R train1 user-item interactions.npz")  # TRAIN+VAL
    R_test_path = os.path.join("Datasets", Dataset,
                               "User-Item Interaction Matrices R/R test user-item interactions.npz")

    # Load Z once
    Z = load_npz(Z_path)

    # Scaling R matrices from [0,5] to [0,1]
    R_train = scale_sparse_matrix(load_npz(R_train_path))
    R_val = scale_sparse_matrix(load_npz(R_val_path))
    R_train1 = scale_sparse_matrix(load_npz(R_train1_path))
    R_test = scale_sparse_matrix(load_npz(R_test_path))

    # 2. Analyze Statistics
    # Note: R_train1 is usually Train + Val, so total interactions = R_train1 + R_test
    N_of_users = R_train1.shape[0]
    N_of_items = R_train1.shape[1]

    # Total interactions across the entire dataset (Train + Test)
    N_of_interactions = R_train1.nnz + R_test.nnz

    # Density calculation
    # Density = nnz / (M * N)
    Density = N_of_interactions / (N_of_users * N_of_items)

    # Sparsity calculation (Percentage of empty cells)
    Sparsity_percent = (1 - Density) * 100

    print("=" * 40)
    print(f"ANALYSIS REPORT: {Dataset}")
    print("=" * 40)
    print(f"Number of users       : {N_of_users}")
    print(f"Number of items       : {N_of_items}")
    print(f"Total interactions    : {N_of_interactions}")
    print(f"Density               : {Density:.6f}")
    print(f"Sparsity              : {Sparsity_percent:.4f}%")
    print("-" * 40)

    # 3. Plot Long Tail Distribution
    # ADDED: Passing output_dir so the plot can be saved
    plot_long_tail_distribution(R_train1, Dataset, output_dir)

    return


def plot_long_tail_distribution(R, Dataset, output_dir):
    """
    Generates a Long Tail (Rank-Frequency) plot for a sparse interaction matrix
    and saves it to the output directory.
    """

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Calculate Popularity (Column Sums)
    # axis=0 sums down the columns (items). .A1 flattens to 1D array.
    item_popularity = np.array(R.sum(axis=0)).ravel()

    # 2. Sort Data (Descending Order)
    sorted_popularity = np.sort(item_popularity)[::-1]

    # 3. Create Plots (Linear and Log-Log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Linear Scale Plot ---
    ax1.plot(sorted_popularity, color='blue', linewidth=2)
    ax1.set_title(f"{Dataset}: Long Tail (Linear Scale)")
    ax1.set_xlabel("Item Rank")
    ax1.set_ylabel("Number of Interactions")
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(range(len(sorted_popularity)), sorted_popularity, color='blue', alpha=0.1)

    # --- Log-Log Scale Plot ---
    ax2.plot(sorted_popularity, color='red', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f"{Dataset}: Long Tail (Log-Log Scale)")
    ax2.set_xlabel("Item Rank (Log)")
    ax2.set_ylabel("Number of Interactions (Log)")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # ADDED: Save Plot logic
    save_path = os.path.join(output_dir, f"{Dataset}_Long_Tail_Distribution.jpg")
    plt.savefig(save_path, dpi=300)
    print(f"Long Tail plot saved to: {save_path}")

    plt.show()

    # 4. Print Statistics
    total_interactions = sorted_popularity.sum()
    if total_interactions > 0:
        top_20_percent = int(len(sorted_popularity) * 0.2)
        head_interactions = sorted_popularity[:top_20_percent].sum()
        pareto_share = head_interactions / total_interactions
    else:
        pareto_share = 0.0

    print(f"--- Popularity Statistics for {Dataset} ---")
    print(f"Total Items: {len(sorted_popularity)}")
    print(f"Top 1% Item Interactions: {sorted_popularity[0] if len(sorted_popularity) > 0 else 0}")
    print(f"Median Item Interactions: {np.median(sorted_popularity)}")
    print(f"Pareto Principle: Top 20% of items account for {pareto_share:.4%} of all interactions.")
    print("=" * 40)


