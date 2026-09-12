from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("E:/RoxanaProjects/DotPlotExcercise/output_Dotplot_breast_cancer")
OUT_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "Sample_code_number",
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
    "Class",  # Attribute 11: 2 = benign, 4 = malignant
]

FEATURE_COLUMNS = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]



def load_dataset():
    # fetch dataset 
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
    
    # # data (as pandas dataframes) 
    # X = breast_cancer_wisconsin_original.data.features 
    # y = breast_cancer_wisconsin_original.data.targets 
    
    # metadata 
    print(breast_cancer_wisconsin_original.metadata) 
    
    # variable information 
    print(breast_cancer_wisconsin_original.variables) 

    df = breast_cancer_wisconsin_original.data.original # one DataFrame
    return df

def prepare_records(df):
    before = len(df)

    # Remove Clump Thickness == 1
    df = df[df["Clump_thickness"] != 1].copy()
    
    # Drop rows with missing values in any scoring feature
    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    after_dropna = len(df)

    print(f"loaded Rows: {before} and after all dropping and filtering: {after_dropna}")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(int)
    df["Class"] = df["Class"].astype(int)

    return df.reset_index(drop=True)

def compute_score(df):

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X_std)
    centroids = kmeans.cluster_centers_
    centroid_means = centroids.mean(axis=1)

    # The higher average standardized feature values the higher-abnormality cluster
    low_centroid = np.argmin(centroid_means)
    high_centroid = np.argmax(centroid_means)

    distances = kmeans.transform(X_std)

    distance_to_low = distances[:, low_centroid]
    distance_to_high = distances[:, high_centroid] 

    # the closer to high-abnormality centroid the more score near 100
    score = (distance_to_low / (distance_to_low + distance_to_high) * 100)

    df = df.copy()
    df["Malignant_Score"] = score
    df["Cluster"] = cluster_labels  # kept for reference/inspection only

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"\nK-means cluster sizes: {dict(cluster_sizes)}")
    return df

def report_correlation_ROC(df):

    pearson_r, pearson_p = pearsonr(df["Malignant_Score"], df["Class"])
    spearman_r, spearman_p = spearmanr(df["Malignant_Score"], df["Class"])

        # Class is 2 (benign) / 4 (malignant); AUC needs 0/1 with 1 = positive class
    y_true = (df["Class"] == 4).astype(int)
    auc = roc_auc_score(y_true, df["Malignant_Score"])
    fpr, tpr, thresholds = roc_curve(y_true, df["Malignant_Score"])

    print("\nCorrelation between Malignant_Score and Attribute 11 (Class):")
    print(f"  Pearson  (point-biserial) r = {pearson_r:.3f}  (p = {pearson_p:.2e})")
    print(f"  Spearman rho              = {spearman_r:.3f}  (p = {spearman_p:.2e})")

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,       
    }
def export_csv(df, path: Path):
    export_cols = ["Sample_code_number"] + FEATURE_COLUMNS + ["Malignant_Score", "Class"]
    df[export_cols].to_csv(path, index=False)
    print(f"\nWrote scored data to {path}  ({len(df)} rows)")


def make_chart(df: pd.DataFrame, corr: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left: score distribution split by true class
    benign = df.loc[df["Class"] == 2, "Malignant_Score"]
    malignant = df.loc[df["Class"] == 4, "Malignant_Score"]
    axes[0].hist(benign, bins=20, alpha=0.6, label="Benign (2)", color="#4C72B0")
    axes[0].hist(malignant, bins=20, alpha=0.6, label="Malignant (4)", color="#C44E52")
    axes[0].set_xlabel("Malignant_Score (0-100)")
    axes[0].set_ylabel("Number of cases")
    axes[0].set_title("Score distribution by true Class")
    axes[0].legend()

    # Middle: boxplot, same comparison
    axes[1].boxplot(
        [benign, malignant],
        tick_labels=["Benign (2)", "Malignant (4)"],
        patch_artist=True,
        boxprops=dict(facecolor="#DDDDDD"),
    )
    axes[1].set_ylabel("Malignant_Score (0-100)")
    axes[1].set_title(
        f"Pearson r = {corr['pearson_r']:.2f}, Spearman rho = {corr['spearman_r']:.2f}"
    )

    # Right: ROC curve
    axes[2].plot(corr["fpr"], corr["tpr"], color="green", linewidth=2,
                 label=f"AUC = {corr['auc']:.3f}")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                 label="Chance (AUC = 0.5)")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC curve: score as ranking function")
    axes[2].legend(loc="lower right")
    axes[2].set_aspect("equal")

    fig.suptitle("Unsupervised malignant-likelihood score vs. Attribute 11 (held out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote chart to {path}")


def main():
    df = load_dataset()
    df = prepare_records(df)
    df = compute_score(df)
    corr = report_correlation_ROC(df)
    export_csv(df, OUT_DIR / "scored_data.csv")
    make_chart(df, corr, OUT_DIR / "score_summary.png")


if __name__ == "__main__":
    main()

