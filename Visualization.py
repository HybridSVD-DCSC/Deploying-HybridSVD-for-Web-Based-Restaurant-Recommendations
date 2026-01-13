import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. Helper: Get Color based on Model Name
# ---------------------------------------------------------
def get_model_color(name):
    name_lower = name.lower()
    if "random" in name_lower:
        return "silver"
    elif "mostpopular" in name_lower:
        return "#ff7f0e"
    elif "puresvd" in name_lower:
        return "#1f77b4"
    elif "category" in name_lower:
        return "#2ca02c"
    elif "description" in name_lower:
        return "#9467bd"
    elif "geolocation" in name_lower:
        return "#d62728"
    elif "genres" in name_lower:
        return "#17becf"
    else:
        return "gray"


# ---------------------------------------------------------
# 2. Helper: Clean Labels
# ---------------------------------------------------------
def clean_label(label):
    """
    Simplifies labels: "HybridSVD Category 0.48d..." -> "Category"
    """
    if "Random" in label: return "Random"
    if "MostPopular" in label: return "Most Popular"
    if "PureSVD" in label: return "PureSVD"
    if "HybridSVD" in label:
        parts = label.split()
        if len(parts) > 1: return parts[1]  # Returns Category, Geolocation, etc.
    return label


# ---------------------------------------------------------
# 3. Helper: Data Collector
# ---------------------------------------------------------
def get_dataset_metrics(Dataset, N, base_output_dir="outputs"):
    baseline_results = {}

    if Dataset == "DC Core 20":
        baseline_source_dir = os.path.join(base_output_dir, "DC Core 20 Category")
    else:
        baseline_source_dir = os.path.join(base_output_dir, "ml-100k Genre")

    # A. Standard Baselines
    for base_name, filename in [("Random", "Random_results.pkl"),
                                ("MostPopular", "MostPopular_results.pkl"),
                                ("PureSVD", "PureSVD_results.pkl")]:
        path = os.path.join(baseline_source_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)

            if base_name == "PureSVD":
                # Create a unique key but we will clean it later
                key = f"PureSVD {data.get('d', 0):.2f}"
            else:
                key = base_name

            baseline_results[key] = {
                "Hit": data.get("Hit", 0) if "Hit" in data else data.get(f"Hit@{N}", 0),
                "NDCG": data.get("NDCG", 0) if "NDCG" in data else data.get(f"NDCG@{N}", 0),
                "Coverage": data.get("Coverage", 0)
            }

    # B. HybridSVD Results
    if Dataset == "DC Core 20":
        sim_types = ["Category", "Description", "Geolocation"]
        for sim in sim_types:
            folder = os.path.join(base_output_dir, f"DC Core 20 {sim}")
            params_p = os.path.join(folder, f"DC Core 20 {sim} best params.pkl")
            res_p = os.path.join(folder, f"DC Core 20 {sim} test results.pkl")

            if os.path.exists(params_p) and os.path.exists(res_p):
                with open(res_p, "rb") as f: res = pickle.load(f)
                label = f"HybridSVD {sim}"
                baseline_results[label] = {
                    "Hit": res.get(f"Hit@{N}", 0),
                    "NDCG": res.get(f"NDCG@{N}", 0),
                    "Coverage": res.get("Coverage", 0)
                }

    elif Dataset == "ml-100k":
        folder = os.path.join(base_output_dir, "ml-100k Genre")
        res_p = os.path.join(folder, "ml-100k Genre test results.pkl")
        if not os.path.exists(res_p): res_p = os.path.join(folder, "ml-100k test results.pkl")

        if os.path.exists(res_p):
            with open(res_p, "rb") as f: res = pickle.load(f)
            label = "HybridSVD Genres"
            baseline_results[label] = {
                "Hit": res.get(f"Hit@{N}", 0),
                "NDCG": res.get(f"NDCG@{N}", 0),
                "Coverage": res.get("Coverage", 0)
            }

    return baseline_results


# ---------------------------------------------------------
# 4. Plotter Function (Already Fixed)
# ---------------------------------------------------------
def plot_combined_dashboard(dc_results, ml_results, N, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))

    row_data = [(0, dc_results, "DC Core 20"), (1, ml_results, "ml-100k")]

    for row_idx, results, dataset_name in row_data:
        if not results: continue

        original_labels = list(results.keys())
        clean_labels = [clean_label(lbl) for lbl in original_labels]

        hits = [results[m]["Hit"] for m in original_labels]
        ndcgs = [results[m]["NDCG"] for m in original_labels]
        coverages = [results[m]["Coverage"] for m in original_labels]
        colors = [get_model_color(m) for m in original_labels]

        def plot_bar(ax, data, title, ylabel, ylim=None):
            x_pos = list(range(len(clean_labels)))
            ax.bar(x_pos, data, color=colors, edgecolor='black', alpha=0.8)
            ax.set_title(f"{dataset_name}\n{title}", fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12)
            if ylim:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim(0, max(data) * 1.2 if data else 1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=12)

        plot_bar(axes[row_idx, 0], hits, f"Hit@{N}", f"Hit@{N}")
        plot_bar(axes[row_idx, 1], ndcgs, f"NDCG@{N}", f"NDCG@{N}")
        plot_bar(axes[row_idx, 2], coverages, "Coverage", "Coverage", ylim=(0, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Combined_Baselines_Comparison.jpg"), dpi=300)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------
# 5. NEW FUNCTION: Generate LaTeX Table
# ---------------------------------------------------------
def generate_latex_table(dc_results, ml_results, N, output_dir):
    """
    Creates a combined LaTeX table comparing all models.
    """
    rows = []

    # Process both datasets
    for dataset_name, results in [("DC Core 20", dc_results), ("ml-100k", ml_results)]:
        if not results: continue

        # Find max values for bolding
        max_hit = max(v["Hit"] for v in results.values())
        max_ndcg = max(v["NDCG"] for v in results.values())
        max_cov = max(v["Coverage"] for v in results.values())

        for model_key, metrics in results.items():
            model_name = clean_label(model_key)

            # Format numbers (Bold the best ones)
            hit_val = metrics["Hit"]
            ndcg_val = metrics["NDCG"]
            cov_val = metrics["Coverage"]

            hit_str = f"\\textbf{{{hit_val:.4f}}}" if hit_val == max_hit else f"{hit_val:.4f}"
            ndcg_str = f"\\textbf{{{ndcg_val:.4f}}}" if ndcg_val == max_ndcg else f"{ndcg_val:.4f}"
            cov_str = f"\\textbf{{{cov_val:.4f}}}" if cov_val == max_cov else f"{cov_val:.4f}"

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                f"Hit@{N}": hit_str,
                f"NDCG@{N}": ndcg_str,
                "Coverage": cov_str
            })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Generate LaTeX
    latex_str = """
\\begin{table}[h]
    \\centering
    \\caption{Performance comparison of all models (Best scores in bold).}
    \\label{tab:model_results}
    \\begin{tabular}{l l c c c}
        \\toprule
        \\textbf{Dataset} & \\textbf{Model} & \\textbf{Hit@10} & \\textbf{NDCG@10} & \\textbf{Coverage} \\\\
        \\midrule
"""

    current_dataset = None
    for _, row in df.iterrows():
        # Add a midrule between datasets for clarity
        if row['Dataset'] != current_dataset:
            if current_dataset is not None:
                latex_str += "        \\addlinespace\n"
            current_dataset = row['Dataset']
            # Print dataset name only once per group (optional, or print every time)
            ds_display = f"\\textbf{{{row['Dataset']}}}"
        else:
            ds_display = ""  # Leave blank for cleaner look

        latex_str += f"        {ds_display} & {row['Model']} & {row[f'Hit@{N}']} & {row[f'NDCG@{N}']} & {row['Coverage']} \\\\\n"

    latex_str += """        \\bottomrule
    \\end{tabular}
\\end{table}
"""

    # Save to file
    table_path = os.path.join(output_dir, "results_table.tex")
    with open(table_path, "w") as f:
        f.write(latex_str)

    print(f"LaTeX Table saved to: {table_path}")
    print("-" * 30)
    print(latex_str)  # Print to console for quick checking


# ---------------------------------------------------------
# 6. Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    N_value = 10
    base_outputs = "outputs"
    output_directory = os.path.join(base_outputs, "Combined_Analysis")

    print("Gathering data...")
    dc_data = get_dataset_metrics("DC Core 20", N_value, base_output_dir=base_outputs)
    ml_data = get_dataset_metrics("ml-100k", N_value, base_output_dir=base_outputs)

    print("\nGenerating combined figure...")
    plot_combined_dashboard(dc_data, ml_data, N_value, output_directory)

    print("\nGenerating LaTeX table...")
    generate_latex_table(dc_data, ml_data, N_value, output_directory)