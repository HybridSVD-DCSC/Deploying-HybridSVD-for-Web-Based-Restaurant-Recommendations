import numpy as np

def hitrate(topN_matrix, R_test):
    num_users = topN_matrix.shape[0]   # = sample_size
    hits = 0

    for i in range(num_users):
        # Get true held-out item for user
        true_item = R_test[i].nonzero()[1][0]

        # Top-N recommendations (titles or IDs)
        rec_items = topN_matrix[i]

        # Hit?
        if true_item in rec_items:
            hits += 1

    hit_at_N = hits / num_users

    return hit_at_N, hits, num_users

def compute_coverage(topN_matrix, num_items):
    # Flatten all Top-N recommendations into one long list
    all_recs = topN_matrix.flatten()

    # Count unique recommended items
    unique_recs = np.unique(all_recs)

    coverage = len(unique_recs) / num_items
    num_rec_items = len(unique_recs)
    return coverage, num_rec_items


def compute_NDCG(topN_matrix, R_test):
    num_users = topN_matrix.shape[0]
    ndcg_sum = 0.0

    for i in range(num_users):
        # Get true held-out item for user
        true_item = R_test[i].nonzero()[1][0]

        # Get top-N list
        rec_items = topN_matrix[i]

        # Check if true item is in top-N
        if true_item in rec_items:
            rank = np.where(rec_items == true_item)[0][0]  # index 0..N-1

            # DCG = 1 / log2(rank + 2)
            # "+2" because rank starts at 0 → position = rank+1 → DCG uses log2(pos+1)
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0

        ndcg_sum += ndcg

    NDCG_at_N = ndcg_sum / num_users
    return NDCG_at_N
