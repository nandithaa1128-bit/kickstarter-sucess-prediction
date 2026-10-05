from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prediction import predict_new_project, FEATURE_NAMES


# ============================================================
# CREATE API
# ============================================================

app = FastAPI(
    title="Kickstarter Success Prediction API",
    description="API for predicting Kickstarter project success.",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT FORMAT
# ============================================================

class ProjectInput(BaseModel):

    name: str
    goal: float
    usd_goal_real: float

    launch_date: str
    deadline_date: str

    category: str
    main_category: str

    currency: str
    country: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Kickstarter Success Prediction API is running"
    }


# ============================================================
# GET VALID MODEL OPTIONS
# ============================================================

@app.get("/options")
def get_options():

    categories = sorted([
        col.replace("category_", "", 1)
        for col in FEATURE_NAMES
        if col.startswith("category_")
    ])

    main_categories = sorted([
        col.replace("main_category_", "", 1)
        for col in FEATURE_NAMES
        if col.startswith("main_category_")
    ])

    currencies = sorted([
        col.replace("currency_", "", 1)
        for col in FEATURE_NAMES
        if col.startswith("currency_")
    ])

    countries = sorted([
        col.replace("country_", "", 1)
        for col in FEATURE_NAMES
        if col.startswith("country_")
    ])

    return {
        "categories": categories,
        "main_categories": main_categories,
        "currencies": currencies,
        "countries": countries
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(project: ProjectInput):

    try:

        result = predict_new_project(
            name=project.name,
            goal=project.goal,
            usd_goal_real=project.usd_goal_real,

            launch_date=project.launch_date,
            deadline_date=project.deadline_date,

            category=project.category,
            main_category=project.main_category,

            currency=project.currency,
            country=project.country
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )