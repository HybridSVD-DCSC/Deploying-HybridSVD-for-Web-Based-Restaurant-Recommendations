import numpy as np
import pandas as pd
from scipy import sparse
import json
import os

# --- CONFIG ---
DATA_DIR = 'Data'
MIN_HISTORY_LEN = 5
NUM_REPRESENTATIVE_USERS = 35


def load_data():
    print("Loading data...")
    path_R = os.path.join(DATA_DIR, 'R_retrain_rest_20.npz')
    path_map = os.path.join(DATA_DIR, 'restaurant_item_map_rest_20.json')
    path_cats = os.path.join(DATA_DIR, 'categorized_restaurants.csv')

    R_matrix = sparse.load_npz(path_R)

    with open(path_map, 'r') as f:
        item_map = json.load(f)
    gmap_to_idx = {v: int(k) for k, v in item_map.items()}

    cat_df = pd.read_csv(path_cats)

    return R_matrix, gmap_to_idx, cat_df


def generate_lift_based_personas():
    R_matrix, gmap_to_idx, cat_df = load_data()
    n_users, n_items = R_matrix.shape

    persona_vectors = {}

    print(f"\nScanning {n_users} users using LIFT metric...")

    # 1. PRE-CALCULATION
    R_csr = R_matrix.tocsr()

    # A. User Totals (for Lift Calculation)
    user_total_history = np.array(R_csr.sum(axis=1)).flatten()
    global_total_interactions = user_total_history.sum()
    user_total_history[user_total_history == 0] = 1

    # --- NEW: Calculate Popularity Weights (IDF) ---
    # We want to punish items that everyone visits (Popularity Bias)
    # Count how many users visited each item (Column Sums of binary matrix)
    R_binary = (R_csr > 0).astype(int)
    item_user_counts = np.array(R_binary.sum(axis=0)).flatten()

    # Avoid div by zero
    item_user_counts[item_user_counts == 0] = 1

    # IDF Formula: log(Total_Users / Item_Popularity)
    # Items visited by everyone -> log(1) -> 0 weight
    # Items visited by few -> log(Big) -> High weight
    idf_weights = np.log10(n_users / item_user_counts)

    print("    Popularity weighting (IDF) calculated.")
    # -------------------------------------------------------

    for category in cat_df.columns:
        print(f"--> Processing: {category}")

        # 2. Identify Target Items
        target_gmaps = cat_df[category].dropna().astype(str).tolist()
        target_indices = [gmap_to_idx[gid] for gid in target_gmaps if gid in gmap_to_idx]

        if not target_indices:
            print(f"    (Skipping {category}: No matching items found)")
            continue

        # 3. Global Probability
        cat_interactions_global = R_csr[:, target_indices].sum()
        prob_category_global = cat_interactions_global / global_total_interactions

        if prob_category_global == 0: continue

        # 4. User Probability
        user_cat_counts = np.array(R_csr[:, target_indices].sum(axis=1)).flatten()
        prob_category_given_user = user_cat_counts / user_total_history

        # 5. Calculate LIFT
        lift_scores = prob_category_given_user / prob_category_global

        # 6. Select Top Representatives
        valid_mask = (user_total_history >= MIN_HISTORY_LEN) & (user_cat_counts > 0)

        if not np.any(valid_mask):
            print("    No valid users found.")
            continue

        valid_user_indices = np.where(valid_mask)[0]
        valid_lifts = lift_scores[valid_user_indices]

        # Sort descending
        sorted_indices_local = np.argsort(valid_lifts)[::-1]
        top_n_local = sorted_indices_local[:NUM_REPRESENTATIVE_USERS]
        top_user_indices = valid_user_indices[top_n_local]

        # Stats
        highest_lift = valid_lifts[top_n_local[0]]
        lowest_lift_in_set = valid_lifts[top_n_local[-1]]
        most_representative_user_id = top_user_indices[0]

        print(f"    Selected {len(top_user_indices)} representatives.")
        print(f"    Max Lift: {highest_lift:.2f} | Min Lift: {lowest_lift_in_set:.2f}")
        print(f"    🏆 Most Representative User ID: {most_representative_user_id}")

        # --- UPDATED: Compute Weighted Centroid ---
        # Get the raw interactions of the super fans
        fan_vectors = R_csr[top_user_indices]

        # Average them to get the "Fan Profile"
        avg_vector = np.array(fan_vectors.mean(axis=0)).flatten()

        # APPLY BIAS CORRECTION:
        # Multiply the Fan Profile by the Scarcity (IDF) of the items.
        # This reduces the signal of "Global Pop" items and boosts "Niche" items.
        weighted_vector = avg_vector * idf_weights

        persona_vectors[category] = weighted_vector
        # --------------------------------------------------

    # 8. Save (Kept filename exactly as requested)
    output_path = os.path.join(DATA_DIR, 'persona_vectors_lift.npy')
    np.save(output_path, persona_vectors)
    print(f"\n✅ Saved {len(persona_vectors)} de-biased persona vectors to {output_path}")


if __name__ == "__main__":
    generate_lift_based_personas()