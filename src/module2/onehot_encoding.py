import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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
print("DATASET LOADED")
print("========================================")

print("Original shape:", df.shape)


# ============================================================
# 5. DROP UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [
    "user_id",
    "device_id",
    "notes"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ============================================================
# 6. HANDLE TIMESTAMP
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df = df.drop(
        columns=["timestamp"]
    )


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


if numeric_columns:

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )


if categorical_columns:

    for column in categorical_columns:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ============================================================
# 8. DISPLAY CATEGORICAL COLUMNS
# ============================================================

print("\n========================================")
print("CATEGORICAL COLUMNS")
print("========================================")

print(categorical_columns)


# ============================================================
# 9. ONE-HOT ENCODING
# ============================================================

if categorical_columns:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded_data = encoder.fit_transform(
        df[categorical_columns]
    )

    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=df.index
    )

else:

    encoded_df = pd.DataFrame(
        index=df.index
    )


# ============================================================
# 10. REMOVE ORIGINAL CATEGORICAL COLUMNS
# ============================================================

df_numeric = df.drop(
    columns=categorical_columns
)


# ============================================================
# 11. COMBINE NUMERICAL + ENCODED DATA
# ============================================================

final_df = pd.concat(
    [
        df_numeric,
        encoded_df
    ],
    axis=1
)


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("AFTER ONE-HOT ENCODING")
print("========================================")

print(
    "New shape:",
    final_df.shape
)

print("\nEncoded columns:")

print(
    encoded_columns
)


# ============================================================
# 13. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n========================================")
print("ENCODED DATA")
print("========================================")

print(
    final_df.head()
)


# ============================================================
# 14. SAVE ENCODED DATA
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_PATH,
    "one_hot_encoded_data.csv"
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n========================================")
print("ONE-HOT ENCODING COMPLETED")
print("========================================")

print(
    "Saved file:",
    OUTPUT_FILE
)