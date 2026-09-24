import os
import math
import pandas as pd

from flask import Flask, render_template, send_from_directory

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "GlucoBench_benchmark_dataset.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output"
)


PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset():

    return pd.read_csv(DATA_PATH)


def load_csv(filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        return pd.read_csv(path)

    return None


# ---------------------------------------------------------
# BASIC DATASET INFORMATION
# ---------------------------------------------------------

def get_basic_info():

    df = load_dataset()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    missing_values = df.isnull().sum()

    missing_details = []

    for column, count in missing_values.items():

        if count > 0:

            missing_details.append({
                "column": column,
                "count": int(count)
            })

    glucose_mean = None
    glucose_median = None
    glucose_min = None
    glucose_max = None
    glucose_std = None

    if "glucose" in df.columns:

        glucose = df["glucose"].dropna()

        if len(glucose) > 0:

            glucose_mean = round(glucose.mean(), 4)
            glucose_median = round(glucose.median(), 4)
            glucose_min = round(glucose.min(), 4)
            glucose_max = round(glucose.max(), 4)
            glucose_std = round(glucose.std(), 4)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "columns_list": list(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_details": missing_details,
        "glucose_mean": glucose_mean,
        "glucose_median": glucose_median,
        "glucose_min": glucose_min,
        "glucose_max": glucose_max,
        "glucose_std": glucose_std
    }


# ---------------------------------------------------------
# REGRESSION METRICS
# ---------------------------------------------------------

def calculate_metrics(data):

    if data is None:

        return None

    if (
        "Actual Glucose" not in data.columns
        or
        "Predicted Glucose" not in data.columns
    ):

        return None


    actual = pd.to_numeric(
        data["Actual Glucose"],
        errors="coerce"
    )


    predicted = pd.to_numeric(
        data["Predicted Glucose"],
        errors="coerce"
    )


    valid = pd.DataFrame({
        "actual": actual,
        "predicted": predicted
    }).dropna()


    actual = valid["actual"]
    predicted = valid["predicted"]


    n = len(actual)


    if n == 0:

        return None


    error = actual - predicted


    absolute_error = error.abs()

    squared_error = error ** 2


    mae = absolute_error.mean()

    mse = squared_error.mean()

    rmse = math.sqrt(mse)


    ss_res = squared_error.sum()

    ss_tot = (
        (actual - actual.mean()) ** 2
    ).sum()


    if ss_tot != 0:

        r2 = 1 - (
            ss_res / ss_tot
        )

    else:

        r2 = 0


    first_actual = float(actual.iloc[0])

    first_predicted = float(predicted.iloc[0])

    first_error = first_actual - first_predicted

    first_absolute_error = abs(first_error)

    first_squared_error = first_error ** 2


    return {

        "mae": round(mae, 4),

        "mse": round(mse, 4),

        "rmse": round(rmse, 4),

        "r2": round(r2, 4),

        "n": n,

        "first_actual": round(first_actual, 4),

        "first_predicted": round(first_predicted, 4),

        "first_error": round(first_error, 4),

        "first_absolute_error": round(
            first_absolute_error,
            4
        ),

        "first_squared_error": round(
            first_squared_error,
            4
        ),

        "actual_mean": round(
            actual.mean(),
            4
        ),

        "ss_res": round(
            float(ss_res),
            4
        ),

        "ss_tot": round(
            float(ss_tot),
            4
        )

    }


# ---------------------------------------------------------
# LOGISTIC REGRESSION METRICS
# ---------------------------------------------------------

def calculate_logistic_metrics(data):

    if data is None:

        return None


    if (
        "Actual Class" not in data.columns
        or
        "Predicted Class" not in data.columns
    ):

        return None


    actual = data["Actual Class"]

    predicted = data["Predicted Class"]


    accuracy = accuracy_score(
        actual,
        predicted
    )


    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )


    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )


    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )


    cm = confusion_matrix(
        actual,
        predicted
    )


    if cm.shape == (2, 2):

        tn = int(cm[0][0])

        fp = int(cm[0][1])

        fn = int(cm[1][0])

        tp = int(cm[1][1])

    else:

        tn = 0
        fp = 0
        fn = 0
        tp = 0


    total = tn + fp + fn + tp


    return {

        "accuracy": round(accuracy, 4),

        "precision": round(precision, 4),

        "recall": round(recall, 4),

        "f1": round(f1, 4),

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp,

        "total": total

    }


# ---------------------------------------------------------
# CLUSTERING INFORMATION
# ---------------------------------------------------------

def calculate_cluster_info(data):

    if data is None:

        return None


    if "cluster" not in data.columns:

        return None


    cluster_values = data["cluster"]


    total_points = len(data)


    noise_points = int(
        (cluster_values == -1).sum()
    )


    normal_clusters = sorted(
        [
            int(x)
            for x in cluster_values.unique()
            if x != -1
        ]
    )


    cluster_sizes = []


    for cluster_number in normal_clusters:

        count = int(
            (cluster_values == cluster_number).sum()
        )

        cluster_sizes.append({

            "cluster": cluster_number,

            "count": count

        })


    return {

        "total_points": total_points,

        "clusters": len(normal_clusters),

        "noise_points": noise_points,

        "cluster_labels": normal_clusters,

        "cluster_sizes": cluster_sizes

    }


# ---------------------------------------------------------
# PCA INFORMATION
# ---------------------------------------------------------

def calculate_pca_info():

    components_path = os.path.join(
        OUTPUT_DIR,
        "pca_components.csv"
    )


    transformed_path = os.path.join(
        OUTPUT_DIR,
        "pca_transformed_data.csv"
    )


    pca_info = {

        "components": [],

        "component_rows": 0,

        "transformed_rows": 0,

        "transformed_columns": [],

        "pc1_variance": None,

        "pc2_variance": None,

        "total_variance": None

    }


    if os.path.exists(components_path):

        components = pd.read_csv(
            components_path
        )


        pca_info["components"] = list(
            components.columns
        )


        pca_info["component_rows"] = len(
            components
        )


        numeric_component_columns = components.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()


        if len(numeric_component_columns) >= 2:

            pc1_values = components[
                numeric_component_columns[0]
            ].abs()

            pc2_values = components[
                numeric_component_columns[1]
            ].abs()


            pca_info["pc1_variance"] = round(
                pc1_values.mean(),
                4
            )


            pca_info["pc2_variance"] = round(
                pc2_values.mean(),
                4
            )


    if os.path.exists(transformed_path):

        transformed = pd.read_csv(
            transformed_path
        )


        pca_info["transformed_rows"] = len(
            transformed
        )


        pca_info["transformed_columns"] = list(
            transformed.columns
        )


    return pca_info


# ---------------------------------------------------------
# EDA INFORMATION
# ---------------------------------------------------------

def get_eda_info():

    df = load_dataset()


    info = {

        "rows": len(df),

        "columns": len(df.columns),

        "duplicates": int(
            df.duplicated().sum()
        ),

        "total_missing": int(
            df.isnull().sum().sum()
        ),

        "missing": [],

        "numeric_summary": [],

        "categorical_summary": []

    }


    # Missing values

    missing = df.isnull().sum()


    for column, count in missing.items():

        if count > 0:

            info["missing"].append({

                "column": column,

                "count": int(count)

            })


    # Numeric summary

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns


    for column in numeric_columns:

        series = df[column].dropna()


        if len(series) == 0:

            continue


        info["numeric_summary"].append({

            "column": column,

            "mean": round(series.mean(), 4),

            "median": round(series.median(), 4),

            "min": round(series.min(), 4),

            "max": round(series.max(), 4),

            "std": round(series.std(), 4)

        })


    # Categorical summary

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns


    for column in categorical_columns:

        info["categorical_summary"].append({

            "column": column,

            "unique": int(
                df[column].nunique(dropna=True)
            )

        })


    return info


# ---------------------------------------------------------
# CHECK IMAGE
# ---------------------------------------------------------

def plot_exists(filename):

    path = os.path.join(
        PLOTS_DIR,
        filename
    )

    return os.path.exists(path)


def output_exists(filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    return os.path.exists(path)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
def home():

    info = get_basic_info()


    return render_template(
        "home.html",
        info=info
    )


# ---------------------------------------------------------
# SESSION
# ---------------------------------------------------------

@app.route("/session/<section>")
def session(section):

    info = get_basic_info()


    sections = {

        # -------------------------------------------------
        # EDA
        # -------------------------------------------------

        "eda": {

            "title": "Exploratory Data Analysis",

            "description":
                "Explore dataset structure, distributions, missing values and relationships.",

            "type": "eda"

        },


        # -------------------------------------------------
        # PREPROCESSING
        # -------------------------------------------------

        "preprocessing": {

            "title": "Data Preprocessing",

            "description":
                "Missing-value handling and feature scaling.",

            "type": "preprocessing"

        },


        # -------------------------------------------------
        # ONE HOT
        # -------------------------------------------------

        "one-hot": {

            "title": "One-Hot Encoding",

            "description":
                "Categorical features converted into numerical binary columns.",

            "type": "one-hot",

            "file":
                "one_hot_encoded_data.csv"

        },


        # -------------------------------------------------
        # LINEAR
        # -------------------------------------------------

        "linear": {

            "title": "Linear Regression",

            "description":
                "Linear Regression glucose prediction and mathematical evaluation.",

            "type": "model",

            "file":
                "linear_regression_predictions.csv",

            "formula":
                "MAE = Σ|Actual − Predicted| / N"

        },


        # -------------------------------------------------
        # RIDGE
        # -------------------------------------------------

        "ridge": {

            "title": "Ridge Regression",

            "description":
                "Ridge Regression glucose prediction with L2 regularization.",

            "type": "model",

            "file":
                "ridge_regression_predictions.csv",

            "formula":
                "Ridge minimizes SSE + α × Σ(coefficient²)"

        },


        # -------------------------------------------------
        # LASSO
        # -------------------------------------------------

        "lasso": {

            "title": "Lasso Regression",

            "description":
                "Lasso Regression glucose prediction with L1 regularization.",

            "type": "model",

            "file":
                "lasso_regression_predictions.csv",

            "formula":
                "Lasso minimizes SSE + α × Σ|coefficient|"

        },


        # -------------------------------------------------
        # LOGISTIC
        # -------------------------------------------------

        "logistic": {

            "title": "Logistic Regression",

            "description":
                "Binary classification and evaluation calculations.",

            "type": "classification",

            "file":
                "logistic_regression_predictions.csv"

        },


        # -------------------------------------------------
        # DECISION TREE
        # -------------------------------------------------

        "decision-tree": {

            "title": "Decision Tree Regression",

            "description":
                "Decision Tree glucose prediction, evaluation and tree visualization.",

            "type": "tree",

            "file":
                "decision_tree_predictions.csv",

            "tree_image":
                "decision_tree_plot.png"

        },


        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------

        "random-forest": {

            "title": "Random Forest Regression",

            "description":
                "Random Forest glucose prediction and Tree 1 visualization.",

            "type": "tree",

            "file":
                "random_forest_predictions.csv",

            "tree_image":
                "random_forest_tree.png"

        },


        # -------------------------------------------------
        # ADABOOST
        # -------------------------------------------------

        "adaboost": {

            "title": "AdaBoost",

            "description":
                "AdaBoost regression prediction and evaluation calculations.",

            "type": "model",

            "file":
                "adaboost_predictions.csv"

        },


        # -------------------------------------------------
        # GRADIENT BOOSTING
        # -------------------------------------------------

        "gradient-boosting": {

            "title": "Gradient Boosting",

            "description":
                "Gradient Boosting regression prediction and evaluation calculations.",

            "type": "model",

            "file":
                "gradient_boosting_predictions.csv"

        },


        # -------------------------------------------------
        # XGBOOST
        # -------------------------------------------------

        "xgboost": {

            "title": "XGBoost",

            "description":
                "XGBoost regression prediction and evaluation calculations.",

            "type": "model",

            "file":
                "xgboost_predictions.csv"

        },


        # -------------------------------------------------
        # LIGHTGBM
        # -------------------------------------------------

        "lightgbm": {

            "title": "LightGBM",

            "description":
                "LightGBM regression prediction and evaluation calculations.",

            "type": "model",

            "file":
                "lightgbm_predictions.csv"

        },


        # -------------------------------------------------
        # KMEANS
        # -------------------------------------------------

        "kmeans": {

            "title": "K-Means Clustering",

            "description":
                "K-Means clustering calculation, cluster assignments and visualization.",

            "type": "clustering",

            "file":
                "kmeans_clustered_data.csv",

            "image":
                "kmeans_glucose_stress_clusters.png"

        },


        # -------------------------------------------------
        # DBSCAN
        # -------------------------------------------------

        "dbscan": {

            "title": "DBSCAN Clustering",

            "description":
                "DBSCAN clustering calculation, clusters and noise points.",

            "type": "clustering",

            "file":
                "dbscan_clustered_data.csv",

            "image":
                "dbscan_glucose_stress_clusters.png"

        },


        # -------------------------------------------------
        # PCA
        # -------------------------------------------------

        "pca": {

            "title": "Principal Component Analysis",

            "description":
                "PCA transformation, explained variance and dimensionality reduction.",

            "type": "pca",

            "file":
                "pca_transformed_data.csv"

        },


        # -------------------------------------------------
        # COMPARISON
        # -------------------------------------------------

        "comparison": {

            "title": "Model Comparison",

            "description":
                "Comparison of regression model evaluation metrics.",

            "type": "comparison"

        }

    }


    if section not in sections:

        return "Section not found", 404


    current = sections[section]


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    data = None


    if "file" in current:

        data = load_csv(
            current["file"]
        )


    # -----------------------------------------------------
    # REGRESSION METRICS
    # -----------------------------------------------------

    metrics = None


    if current["type"] in [
        "model",
        "tree"
    ]:

        metrics = calculate_metrics(
            data
        )


    # -----------------------------------------------------
    # LOGISTIC METRICS
    # -----------------------------------------------------

    logistic_metrics = None


    if current["type"] == "classification":

        logistic_metrics = calculate_logistic_metrics(
            data
        )


    # -----------------------------------------------------
    # CLUSTER INFORMATION
    # -----------------------------------------------------

    cluster_info = None


    if current["type"] == "clustering":

        cluster_info = calculate_cluster_info(
            data
        )


    # -----------------------------------------------------
    # PCA INFORMATION
    # -----------------------------------------------------

    pca_info = None


    if current["type"] == "pca":

        pca_info = calculate_pca_info()


    # -----------------------------------------------------
    # EDA INFORMATION
    # -----------------------------------------------------

    eda_info = None


    if current["type"] == "eda":

        eda_info = get_eda_info()


    # -----------------------------------------------------
    # EDA PLOT STATUS
    # -----------------------------------------------------

    eda_plots = {

        "glucose_distribution":
            plot_exists(
                "glucose_distribution.png"
            ),

        "glucose_boxplot":
            plot_exists(
                "glucose_boxplot.png"
            ),

        "correlation_heatmap":
            plot_exists(
                "correlation_heatmap.png"
            ),

        "missing_values":
            plot_exists(
                "missing_values.png"
            ),

        "numerical_distributions":
            plot_exists(
                "numerical_distributions.png"
            ),

        "categorical_distributions":
            plot_exists(
                "categorical_distributions.png"
            )

    }


    # -----------------------------------------------------
    # TREE IMAGE STATUS
    # -----------------------------------------------------

    tree_image_exists = False


    if current["type"] == "tree":

        tree_image_exists = output_exists(
            current["tree_image"]
        )


    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    comparison = []


    if current["type"] == "comparison":


        model_files = {

            "Linear Regression":
                "linear_regression_predictions.csv",

            "Ridge Regression":
                "ridge_regression_predictions.csv",

            "Lasso Regression":
                "lasso_regression_predictions.csv",

            "Decision Tree":
                "decision_tree_predictions.csv",

            "Random Forest":
                "random_forest_predictions.csv",

            "AdaBoost":
                "adaboost_predictions.csv",

            "Gradient Boosting":
                "gradient_boosting_predictions.csv",

            "XGBoost":
                "xgboost_predictions.csv",

            "LightGBM":
                "lightgbm_predictions.csv"

        }


        for model_name, filename in model_files.items():


            model_data = load_csv(
                filename
            )


            model_metrics = calculate_metrics(
                model_data
            )


            if model_metrics is None:

                continue


            comparison.append({

                "model":
                    model_name,

                "mae":
                    model_metrics["mae"],

                "mse":
                    model_metrics["mse"],

                "rmse":
                    model_metrics["rmse"],

                "r2":
                    model_metrics["r2"]

            })


    # -----------------------------------------------------
    # ONE HOT COLUMNS
    # -----------------------------------------------------

    encoded_columns = []


    if current["type"] == "one-hot":

        if data is not None:

            encoded_columns = list(
                data.columns
            )


    # -----------------------------------------------------
    # RENDER
    # -----------------------------------------------------

    return render_template(

        "session.html",

        info=info,

        section=current,

        data=data,

        metrics=metrics,

        logistic_metrics=logistic_metrics,

        cluster_info=cluster_info,

        pca_info=pca_info,

        eda_info=eda_info,

        eda_plots=eda_plots,

        tree_image_exists=tree_image_exists,

        comparison=comparison,

        encoded_columns=encoded_columns,

        section_name=section

    )


# ---------------------------------------------------------
# EDA PLOTS
# ---------------------------------------------------------

@app.route("/plots/<filename>")
def plots(filename):

    file_path = os.path.join(
        PLOTS_DIR,
        filename
    )


    if not os.path.exists(file_path):

        return (
            f"Plot file not found: {filename}",
            404
        )


    return send_from_directory(
        PLOTS_DIR,
        filename
    )


# ---------------------------------------------------------
# OUTPUT FILES
# ---------------------------------------------------------

@app.route("/output/<filename>")
def output_file(filename):

    file_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    if not os.path.exists(file_path):

        return (
            f"Output file not found: {filename}",
            404
        )


    return send_from_directory(
        OUTPUT_DIR,
        filename
    )


# ---------------------------------------------------------
# RUN FLASK
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )