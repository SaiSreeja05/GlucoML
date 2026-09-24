import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

DATA_PATH = r"C:\Users\samyu\Desktop\archive (2)\GlucoBench_benchmark_dataset.csv"


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
print("K-MEANS CLUSTERING")
print("========================================")

print("Dataset shape:", df.shape)


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
# 7. SCALE FEATURES
# ============================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 8. FIND SILHOUETTE SCORES
# ============================================================

print("\n========================================")
print("TESTING DIFFERENT K VALUES")
print("========================================")

scores = []

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    scores.append(score)

    print(
        "K =", k,
        "Silhouette Score =",
        round(score, 4)
    )


# ============================================================
# 9. SELECT BEST K
# ============================================================

best_k = range(
    2,
    7
)[
    np.argmax(scores)
]


print("\nBest K:", best_k)


# ============================================================
# 10. TRAIN FINAL K-MEANS MODEL
# ============================================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


cluster_labels = final_model.fit_predict(
    X_scaled
)


# ============================================================
# 11. ADD CLUSTER LABELS
# ============================================================

df["cluster"] = cluster_labels


# ============================================================
# 12. CLUSTER COUNTS
# ============================================================

print("\n========================================")
print("CLUSTER COUNTS")
print("========================================")

print(
    df["cluster"].value_counts().sort_index()
)


# ============================================================
# 13. CLUSTER CENTERS
# ============================================================

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(
        final_model.cluster_centers_
    ),
    columns=features
)

cluster_centers.insert(
    0,
    "cluster",
    range(best_k)
)


print("\n========================================")
print("CLUSTER CENTERS")
print("========================================")

print(
    cluster_centers
)


# ============================================================
# 14. SAVE CLUSTERED DATA
# ============================================================

clustered_file = os.path.join(
    OUTPUT_PATH,
    "kmeans_clustered_data.csv"
)

df.to_csv(
    clustered_file,
    index=False
)


# ============================================================
# 15. SAVE CLUSTER CENTERS
# ============================================================

centers_file = os.path.join(
    OUTPUT_PATH,
    "kmeans_cluster_centers.csv"
)

cluster_centers.to_csv(
    centers_file,
    index=False
)


# ============================================================
# 16. PLOT SILHOUETTE SCORES
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(2, 7),
    scores,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "K-Means Silhouette Score"
)

plt.xticks(
    range(2, 7)
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "kmeans_silhouette_score.png"
    )
)

plt.show()


# ============================================================
# 17. GLUCOSE CLUSTER VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    df["glucose"],
    df["stress_level"],
    c=df["cluster"],
    alpha=0.6
)

plt.xlabel(
    "Glucose"
)

plt.ylabel(
    "Stress Level"
)

plt.title(
    "K-Means Clusters: Glucose vs Stress Level"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "kmeans_glucose_stress_clusters.png"
    )
)

plt.show()


# ============================================================
# 18. COMPLETION
# ============================================================

print("\n========================================")
print("K-MEANS COMPLETED")
print("========================================")

print(
    "Clustered data saved to:",
    clustered_file
)

print(
    "Cluster centers saved to:",
    centers_file
)