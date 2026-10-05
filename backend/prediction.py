import os
import json
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "kickstarter_catboost_final.cbm"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "threshold.json"
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("Loading Kickstarter model...")

model = CatBoostClassifier()
model.load_model(MODEL_PATH)


# ============================================================
# LOAD EXACT FEATURE LIST
# ============================================================

with open(FEATURE_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)


# ============================================================
# LOAD SAVED THRESHOLD
# ============================================================

with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)

FINAL_THRESHOLD = float(
    threshold_data["threshold"]
)


print("=" * 60)
print("KICKSTARTER PREDICTION PIPELINE READY")
print("=" * 60)
print("Model          :", type(model).__name__)
print("Features       :", len(FEATURE_NAMES))
print("Threshold      :", FINAL_THRESHOLD)
print("=" * 60)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_new_project(
    name,
    goal,
    usd_goal_real,
    launch_date,
    deadline_date,
    category,
    main_category,
    currency,
    country
):

    # --------------------------------------------------------
    # 1. CLEAN INPUTS
    # --------------------------------------------------------

    name = str(name).strip()
    category = str(category).strip()
    main_category = str(main_category).strip()
    currency = str(currency).strip().upper()
    country = str(country).strip().upper()

    goal = float(goal)
    usd_goal_real = float(usd_goal_real)

    launch_date = pd.to_datetime(launch_date)
    deadline_date = pd.to_datetime(deadline_date)


    # --------------------------------------------------------
    # 2. BASIC VALIDATION
    # --------------------------------------------------------

    if not name:
        raise ValueError(
            "Project name cannot be empty."
        )

    if goal <= 0:
        raise ValueError(
            "Goal must be greater than 0."
        )

    if usd_goal_real <= 0:
        raise ValueError(
            "USD goal must be greater than 0."
        )

    if deadline_date <= launch_date:
        raise ValueError(
            "Deadline date must be after launch date."
        )


    # --------------------------------------------------------
    # 3. VALID MAIN CATEGORIES
    # --------------------------------------------------------

    valid_main_categories = [
        "Comics",
        "Crafts",
        "Dance",
        "Design",
        "Fashion",
        "Film & Video",
        "Games",
        "Journalism",
        "Music",
        "Photography",
        "Publishing",
        "Technology",
        "Theater"
    ]

    if main_category not in valid_main_categories:
        raise ValueError(
            f"Unknown main category: '{main_category}'. "
            f"Valid values are: {valid_main_categories}"
        )


    # --------------------------------------------------------
    # 4. VALIDATE CATEGORY
    # --------------------------------------------------------

    category_column = f"category_{category}"

    if category_column not in FEATURE_NAMES:

        available_categories = [
            col.replace("category_", "")
            for col in FEATURE_NAMES
            if col.startswith("category_")
        ]

        raise ValueError(
            f"Unknown category: '{category}'. "
            f"Available categories: {available_categories}"
        )


    # --------------------------------------------------------
    # 5. DATE PROCESSING
    # --------------------------------------------------------

    duration_days = (
        deadline_date - launch_date
    ).days


    # --------------------------------------------------------
    # 6. CREATE EXACT FEATURE ROW
    # --------------------------------------------------------

    model_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=FEATURE_NAMES
    )


    # --------------------------------------------------------
    # 7. NUMERICAL FEATURES
    # --------------------------------------------------------

    if "goal" in model_input.columns:
        model_input.loc[0, "goal"] = goal

    if "usd_goal_real" in model_input.columns:
        model_input.loc[0, "usd_goal_real"] = usd_goal_real

    if "duration_days" in model_input.columns:
        model_input.loc[0, "duration_days"] = duration_days


    # --------------------------------------------------------
    # 8. LAUNCH DATE FEATURES
    # --------------------------------------------------------

    if "launch_year" in model_input.columns:
        model_input.loc[0, "launch_year"] = launch_date.year

    if "launch_month" in model_input.columns:
        model_input.loc[0, "launch_month"] = launch_date.month

    if "launch_day" in model_input.columns:
        model_input.loc[0, "launch_day"] = launch_date.day

    if "launch_weekday" in model_input.columns:
        model_input.loc[0, "launch_weekday"] = launch_date.weekday()


    # --------------------------------------------------------
    # 9. DEADLINE DATE FEATURES
    # --------------------------------------------------------

    if "deadline_year" in model_input.columns:
        model_input.loc[0, "deadline_year"] = deadline_date.year

    if "deadline_month" in model_input.columns:
        model_input.loc[0, "deadline_month"] = deadline_date.month

    if "deadline_weekday" in model_input.columns:
        model_input.loc[0, "deadline_weekday"] = deadline_date.weekday()


    # --------------------------------------------------------
    # 10. PROJECT NAME FEATURES
    # --------------------------------------------------------

    name_clean = name.strip()

    name_length = len(name_clean)

    name_word_count = len(
        name_clean.split()
    )

    if "name_length" in model_input.columns:
        model_input.loc[0, "name_length"] = name_length

    if "name_word_count" in model_input.columns:
        model_input.loc[0, "name_word_count"] = name_word_count


    # --------------------------------------------------------
    # 11. CATEGORY ONE-HOT
    # --------------------------------------------------------

    model_input.loc[0, category_column] = 1


    # --------------------------------------------------------
    # 12. MAIN CATEGORY ONE-HOT
    # --------------------------------------------------------

    main_category_column = (
        f"main_category_{main_category}"
    )

    if main_category_column not in FEATURE_NAMES:
        raise ValueError(
            f"Main category '{main_category}' "
            f"is not present in the trained model."
        )

    model_input.loc[0, main_category_column] = 1


    # --------------------------------------------------------
    # 13. CURRENCY ONE-HOT
    # --------------------------------------------------------

    currency_column = f"currency_{currency}"

    if currency_column not in FEATURE_NAMES:
        raise ValueError(
            f"Unknown currency '{currency}'."
        )

    model_input.loc[0, currency_column] = 1


    # --------------------------------------------------------
    # 14. COUNTRY ONE-HOT
    # --------------------------------------------------------

    country_column = f"country_{country}"

    if country_column not in FEATURE_NAMES:
        raise ValueError(
            f"Unknown country '{country}'."
        )

    model_input.loc[0, country_column] = 1


    # --------------------------------------------------------
    # 15. FORCE EXACT FEATURE ORDER
    # --------------------------------------------------------

    model_input = model_input[
        FEATURE_NAMES
    ]


    # --------------------------------------------------------
    # 16. FINAL FEATURE COUNT CHECK
    # --------------------------------------------------------

    if model_input.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature mismatch. "
            f"Expected {len(FEATURE_NAMES)}, "
            f"received {model_input.shape[1]}"
        )


    # --------------------------------------------------------
    # 17. PREDICT PROBABILITY
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(
            model_input
        )[0][1]
    )


    # --------------------------------------------------------
    # 18. APPLY SAVED THRESHOLD
    # --------------------------------------------------------

    prediction = int(
        probability >= FINAL_THRESHOLD
    )


    # --------------------------------------------------------
    # 19. RETURN CLEAN RESULT
    # --------------------------------------------------------

    return {
        "prediction": prediction,

        "result": (
            "SUCCESSFUL"
            if prediction == 1
            else "NOT SUCCESSFUL"
        ),

        "probability": probability,

        "probability_percent": round(
            probability * 100,
            2
        ),

        "threshold": FINAL_THRESHOLD,

        "threshold_percent": round(
            FINAL_THRESHOLD * 100,
            2
        ),

        "duration_days": duration_days,

        "project_name": name,

        "category": category,

        "main_category": main_category,

        "currency": currency,

        "country": country
    }