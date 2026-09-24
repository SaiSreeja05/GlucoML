import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "GlucoBench_benchmark_dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)

df = df.drop(
    columns=["user_id", "device_id", "notes"],
    errors="ignore"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

df = df.drop(columns=["timestamp"], errors="ignore")

df = df.dropna(subset=["glucose"])

X = df.drop(columns=["glucose"])
y = df["glucose"]


numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)


print("\nDecision Tree Regression")
print("------------------------")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))


results = pd.DataFrame({
    "Actual Glucose": y_test.values,
    "Predicted Glucose": predictions
})


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_predictions.csv"
    ),
    index=False
)


transformed_feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


trained_tree = pipeline.named_steps["model"]


plt.figure(figsize=(24, 14))

plot_tree(
    trained_tree,
    feature_names=transformed_feature_names,
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=7
)

plt.title("Decision Tree - Glucose Prediction")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_plot.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nDecision Tree files created successfully.")