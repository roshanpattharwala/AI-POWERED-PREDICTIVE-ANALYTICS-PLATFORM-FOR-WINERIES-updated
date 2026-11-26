
# =============================================================================
# WINERY ANALYTICS PLATFORM
# Flask Deployment Application
# =============================================================================

import os
import __main__
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, flash


# =============================================================================
# CUSTOM IQR WINSORIZER CLASS
# Required to load the saved machine learning pipeline
# =============================================================================

class IQRWinsorizer:

    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(
                X,
                columns=self.feature_names_in_
            )

        self.feature_names_in_ = np.asarray(
            X_df.columns,
            dtype=object
        )

        self.columns_ = (
            list(self.columns)
            if self.columns is not None
            else list(X_df.columns)
        )

        self.bounds_ = {}

        for column in self.columns_:

            q1 = X_df[column].quantile(0.25)
            q3 = X_df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            self.bounds_[column] = (lower, upper)

        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
        else:
            X_out = pd.DataFrame(
                X,
                columns=self.feature_names_in_
            )

        for column in self.columns_:

            lower, upper = self.bounds_[column]

            X_out.loc[:, column] = X_out[column].clip(
                lower=lower,
                upper=upper
            )

        return X_out

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = self.feature_names_in_

        return np.asarray(
            input_features,
            dtype=object
        )


# Register class for compatibility with the saved pickle.
# The original model references __main__.IQRWinsorizer.

__main__.IQRWinsorizer = IQRWinsorizer


# =============================================================================
# CREATE FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

app.secret_key = "winery_secret_key_2026"


# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =============================================================================
# MODEL FILE PATHS
# =============================================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_winery_model.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

DEPLOYMENT_FILE = os.path.join(
    MODEL_DIR,
    "deployment_info.pkl"
)


# =============================================================================
# LOAD MACHINE LEARNING MODEL
# =============================================================================

try:

    best_model = joblib.load(
        MODEL_FILE
    )

    feature_names = joblib.load(
        FEATURE_FILE
    )

    deployment_info = joblib.load(
        DEPLOYMENT_FILE
    )

except Exception as error:

    raise RuntimeError(
        f"\nDeployment loading failed.\n{error}"
    )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

MODEL_NAME = deployment_info["Model Name"]

MODEL_TYPE = deployment_info["Model Type"]

CV_SCORE = deployment_info.get(
    "CV Weighted F1",
    deployment_info.get("Cross Validation F1", "N/A")
)

FEATURE_COUNT = deployment_info["Number of Features"]

TRAINING_SAMPLES = deployment_info["Training Samples"]

TESTING_SAMPLES = deployment_info["Testing Samples"]

FEATURE_ORDER = feature_names.copy()


# =============================================================================
# APPLICATION STARTUP MESSAGE
# =============================================================================

print("\n")

print("=" * 70)

print("WINERY ANALYTICS PLATFORM")

print("=" * 70)

print(f"Model Loaded        : {MODEL_NAME}")

print(f"Model Type          : {MODEL_TYPE}")

print(f"Training Samples    : {TRAINING_SAMPLES}")

print(f"Testing Samples     : {TESTING_SAMPLES}")

print(f"Number of Features  : {FEATURE_COUNT}")

print(f"Cross Validation F1 : {CV_SCORE}")

print("=" * 70)

print("Deployment Ready")

print("=" * 70)

print("\n")


# =============================================================================
# HELPER FUNCTION
# VALIDATE NUMERIC INPUT
# =============================================================================

def get_float(form, field_name):

    value = form.get(field_name, "").strip()

    if value == "":
        raise ValueError(
            f"{field_name} is required."
        )

    try:

        value = float(value)

    except ValueError:

        raise ValueError(
            f"{field_name} must be numeric."
        )

    if not np.isfinite(value):
        raise ValueError(
            f"{field_name} must be a valid number."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return value


# =============================================================================
# HELPER FUNCTION
# PREPARE FEATURE VECTOR
# =============================================================================

def prepare_features(form):

    # -------------------------------------------------------------------------
    # Read User Inputs
    # -------------------------------------------------------------------------

    fixed_acidity = get_float(
        form, "fixed_acidity"
    )

    volatile_acidity = get_float(
        form, "volatile_acidity"
    )

    citric_acid = get_float(
        form, "citric_acid"
    )

    residual_sugar = get_float(
        form, "residual_sugar"
    )

    chlorides = get_float(
        form, "chlorides"
    )

    free_sulfur_dioxide = get_float(
        form, "free_sulfur_dioxide"
    )

    total_sulfur_dioxide = get_float(
        form, "total_sulfur_dioxide"
    )

    density = get_float(
        form, "density"
    )

    ph = get_float(
        form, "ph"
    )

    sulphates = get_float(
        form, "sulphates"
    )

    alcohol = get_float(
        form, "alcohol"
    )

    # -------------------------------------------------------------------------
    # Wine Type
    # -------------------------------------------------------------------------

    wine_type = form.get("wine_type")

    if wine_type == "red":

        is_red = 1

    elif wine_type == "white":

        is_red = 0

    else:

        raise ValueError(
            "Please select a valid Wine Type."
        )

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 1
    # Free_to_Total_SO2_Ratio
    # -------------------------------------------------------------------------

    if total_sulfur_dioxide == 0:

        free_to_total_ratio = 0

    else:

        free_to_total_ratio = (
            free_sulfur_dioxide /
            total_sulfur_dioxide
        )

    # -------------------------------------------------------------------------
    # ENGINEERED FEATURE 2
    # Total_Acidity_Index
    # -------------------------------------------------------------------------

    total_acidity_index = (
        fixed_acidity +
        volatile_acidity -
        ph
    )

    # -------------------------------------------------------------------------
    # FEATURE DICTIONARY
    # -------------------------------------------------------------------------

    feature_dictionary = {

        "fixed acidity": fixed_acidity,

        "volatile acidity": volatile_acidity,

        "citric acid": citric_acid,

        "residual sugar": residual_sugar,

        "chlorides": chlorides,

        "free sulfur dioxide": free_sulfur_dioxide,

        "total sulfur dioxide": total_sulfur_dioxide,

        "density": density,

        "pH": ph,

        "sulphates": sulphates,

        "alcohol": alcohol,

        "is_red": is_red,

        "Free_to_Total_SO2_Ratio": free_to_total_ratio,

        "Total_Acidity_Index": total_acidity_index

    }

    # -------------------------------------------------------------------------
    # ARRANGE FEATURES IN TRAINING ORDER
    # -------------------------------------------------------------------------

    try:

        ordered_features = [

            feature_dictionary[name]

            for name in FEATURE_ORDER

        ]

    except KeyError as error:

        raise ValueError(
            f"Missing model feature: {error}"
        )

    # -------------------------------------------------------------------------
    # CREATE DATAFRAME
    # -------------------------------------------------------------------------

    input_df = pd.DataFrame(

        [ordered_features],

        columns=FEATURE_ORDER

    )

    # -------------------------------------------------------------------------
    # RETURN ORIGINAL FEATURES
    #
    # The saved model pipeline already includes:
    # 1. IQRWinsorizer
    # 2. StandardScaler
    # 3. RandomForestClassifier
    #
    # Therefore, do not scale the data separately here.
    # -------------------------------------------------------------------------

    return input_df


# =============================================================================
# HOME PAGE
# =============================================================================

@app.route("/")

def home():

    return render_template(

        "dashboard.html",

        model_name=MODEL_NAME,

        model_type=MODEL_TYPE,

        feature_count=FEATURE_COUNT,

        cv_score=CV_SCORE,

        prediction=None,

        probability=None,

        prediction_status=None

    )


# =============================================================================
# WINE QUALITY PREDICTION ROUTE
# =============================================================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # ---------------------------------------------------------------------
        # PREPARE FEATURES
        # ---------------------------------------------------------------------

        input_features = prepare_features(
            request.form
        )

        # ---------------------------------------------------------------------
        # PREDICT WINE QUALITY
        # ---------------------------------------------------------------------

        prediction = best_model.predict(
            input_features
        )[0]

        # ---------------------------------------------------------------------
        # PREDICTION PROBABILITY
        # ---------------------------------------------------------------------

        probability = "Not Available"

        if hasattr(best_model, "predict_proba"):

            probabilities = best_model.predict_proba(
                input_features
            )

            confidence = np.max(
                probabilities
            )

            probability = f"{confidence * 100:.2f}%"

        # ---------------------------------------------------------------------
        # PREDICTION STATUS
        # ---------------------------------------------------------------------

        prediction_status = (
            "Prediction Completed Successfully"
        )

        # ---------------------------------------------------------------------
        # RENDER DASHBOARD
        # ---------------------------------------------------------------------

        return render_template(

            "dashboard.html",

            model_name=MODEL_NAME,

            model_type=MODEL_TYPE,

            feature_count=FEATURE_COUNT,

            cv_score=CV_SCORE,

            prediction=prediction,

            probability=probability,

            prediction_status=prediction_status

        )

    except Exception as error:

        app.logger.exception(
            "Wine prediction failed"
        )

        flash(
            str(error),
            "error"
        )

        return render_template(

            "dashboard.html",

            model_name=MODEL_NAME,

            model_type=MODEL_TYPE,

            feature_count=FEATURE_COUNT,

            cv_score=CV_SCORE,

            prediction=None,

            probability=None,

            prediction_status=None

        )


# =============================================================================
# RUN FLASK APPLICATION
# =============================================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
