import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# 2. DATASET PATH
# ============================================================

DATA_PATH = r"C:\Users\psais\PycharmProjects\MLProject\Data\GlucoBench_benchmark_dataset.csv"


# ============================================================
# 3. OUTPUT PATH
# ============================================================

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("========================================")

print("Original dataset shape:", df.shape)


# ============================================================
# 5. SELECT NUMERICAL FEATURES
# ============================================================

features = [
    "glucose",
    "insulin_bolus",
    "insulin_basal",
    "carbs",
    "exercise_steps",
    "heart_rate",
    "skin_temp",
    "gsr",
    "stress_level",
    "alcohol",
    "hbA1c",
    "age",
    "weight",
    "carb_ratio",
    "insulin_sensitivity",
    "glucose_lag_1",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_roll_mean_1h"
]


X = df[features].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(
    X.median()
)


# ============================================================
# 7. STANDARDIZE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 8. APPLY PCA
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


# ============================================================
# 9. EXPLAINED VARIANCE
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\n========================================")
print("EXPLAINED VARIANCE")
print("========================================")

print(
    "PC1:",
    round(explained_variance[0] * 100, 2),
    "%"
)

print(
    "PC2:",
    round(explained_variance[1] * 100, 2),
    "%"
)

print(
    "Total:",
    round(
        explained_variance.sum() * 100,
        2
    ),
    "%"
)


# ============================================================
# 10. CREATE PCA DATAFRAME
# ============================================================

pca_data = pd.DataFrame(
    {
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1]
    }
)


# ============================================================
# 11. SAVE PCA DATA
# ============================================================

pca_file = os.path.join(
    OUTPUT_PATH,
    "pca_transformed_data.csv"
)

pca_data.to_csv(
    pca_file,
    index=False
)


# ============================================================
# 12. PCA COMPONENTS
# ============================================================

components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=["PC1", "PC2"]
)


print("\n========================================")
print("PCA COMPONENTS")
print("========================================")

print(
    components
)


# ============================================================
# 13. SAVE PCA COMPONENTS
# ============================================================

components_file = os.path.join(
    OUTPUT_PATH,
    "pca_components.csv"
)

components.to_csv(
    components_file
)


# ============================================================
# 14. PCA SCATTER PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    pca_data["PC1"],
    pca_data["PC2"],
    alpha=0.6
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "PCA: First Two Principal Components"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "pca_scatter_plot.png"
    )
)

plt.show()


# ============================================================
# 15. EXPLAINED VARIANCE PLOT
# ============================================================

pca_full = PCA()

pca_full.fit(
    X_scaled
)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance,
    marker="o"
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "pca_explained_variance.png"
    )
)

plt.show()


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n========================================")
print("PCA COMPLETED")
print("========================================")

print(
    "PCA data saved to:",
    pca_file
)

print(
    "PCA components saved to:",
    components_file
)