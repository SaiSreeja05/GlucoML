import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
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
print("DBSCAN CLUSTERING")
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
# 8. TEST DIFFERENT DBSCAN PARAMETERS
# ============================================================

print("\n========================================")
print("TESTING DBSCAN PARAMETERS")
print("========================================")

best_score = -1
best_eps = None
best_min_samples = None
best_labels = None

for eps in [0.2, 0.3, 0.4, 0.5]:

    for min_samples in [5, 10, 15]:

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan.fit_predict(
            X_scaled
        )

        unique_labels = set(labels)

        cluster_count = len(
            unique_labels - {-1}
        )

        if cluster_count >= 2:

            mask = labels != -1

            if np.sum(mask) > 1:

                score = silhouette_score(
                    X_scaled[mask],
                    labels[mask]
                )

                print(
                    "eps =", eps,
                    "min_samples =", min_samples,
                    "clusters =", cluster_count,
                    "silhouette =", round(score, 4)
                )

                if score > best_score:

                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels


# ============================================================
# 9. CHECK IF VALID PARAMETERS WERE FOUND
# ============================================================

if best_labels is None:

    print(
        "\nNo valid DBSCAN clustering was found."
    )

    print(
        "Try increasing eps values."
    )

    exit()


# ============================================================
# 10. DISPLAY BEST PARAMETERS
# ============================================================

print("\n========================================")
print("BEST DBSCAN PARAMETERS")
print("========================================")

print(
    "Best eps:",
    best_eps
)

print(
    "Best min_samples:",
    best_min_samples
)

print(
    "Best Silhouette Score:",
    round(best_score, 4)
)


# ============================================================
# 11. ADD CLUSTER LABELS
# ============================================================

df["cluster"] = best_labels


# ============================================================
# 12. COUNT CLUSTERS AND NOISE
# ============================================================

print("\n========================================")
print("CLUSTER COUNTS")
print("========================================")

cluster_counts = df[
    "cluster"
].value_counts().sort_index()

print(
    cluster_counts
)


# ============================================================
# 13. COUNT NOISE POINTS
# ============================================================

noise_count = np.sum(
    best_labels == -1
)

print("\nNoise points:", noise_count)


# ============================================================
# 14. SAVE CLUSTERED DATA
# ============================================================

clustered_file = os.path.join(
    OUTPUT_PATH,
    "dbscan_clustered_data.csv"
)

df.to_csv(
    clustered_file,
    index=False
)


# ============================================================
# 15. CREATE CLUSTER SUMMARY
# ============================================================

summary = []

for cluster in sorted(
    df["cluster"].unique()
):

    cluster_data = df[
        df["cluster"] == cluster
    ]

    summary.append(
        {
            "cluster": cluster,
            "count": len(cluster_data),
            "average_glucose": cluster_data["glucose"].mean(),
            "average_stress": cluster_data["stress_level"].mean(),
            "average_heart_rate": cluster_data["heart_rate"].mean(),
            "average_carbs": cluster_data["carbs"].mean(),
            "average_exercise_steps": cluster_data["exercise_steps"].mean()
        }
    )


cluster_summary = pd.DataFrame(
    summary
)


print("\n========================================")
print("CLUSTER SUMMARY")
print("========================================")

print(
    cluster_summary
)


# ============================================================
# 16. SAVE CLUSTER SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_PATH,
    "dbscan_cluster_summary.csv"
)

cluster_summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. VISUALIZATION
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
    "DBSCAN Clusters: Glucose vs Stress Level"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "dbscan_glucose_stress_clusters.png"
    )
)

plt.show()


# ============================================================
# 18. COMPLETION
# ============================================================

print("\n========================================")
print("DBSCAN COMPLETED")
print("========================================")

print(
    "Clustered data saved to:",
    clustered_file
)

print(
    "Cluster summary saved to:",
    summary_file
)