# Kickstarter Project Success Prediction using Machine Learning

An end-to-end machine learning web application that predicts whether a Kickstarter project is likely to be **SUCCESSFUL** or **NOT SUCCESSFUL** based on project details such as funding goal, launch/deadline dates, category, currency, country, and project name.

The final system uses a **CatBoost Classifier** with **219 features**, a tuned decision threshold of **0.38**, a **FastAPI** backend, and an HTML/CSS/JavaScript frontend.

---

## Problem Statement

Kickstarter projects can have very different outcomes even when their basic characteristics are similar.

This project builds a machine learning system that learns patterns from historical Kickstarter campaigns and predicts the likely success of a new project.

The application provides:

- Predicted outcome: **SUCCESSFUL** or **NOT SUCCESSFUL**
- Probability of success
- Campaign duration
- Decision threshold used for classification

---

## Objectives

- Clean and analyze historical Kickstarter project data.
- Perform Exploratory Data Analysis (EDA).
- Engineer useful numerical, date-based, text-length, and categorical features.
- Compare multiple classification models.
- Select and tune a final CatBoost model.
- Optimize the probability threshold using F1-score.
- Deploy the trained model through a FastAPI backend.
- Provide a web interface for making predictions.

---

## Dataset

The original dataset contains **378,661 Kickstarter projects** and 15 columns.

### Original Columns

```text
ID
name
category
main_category
currency
deadline
goal
launched
pledged
state
backers
country
usd pledged
usd_pledged_real
usd_goal_real

After preprocessing, the cleaned dataset contains:
- 331,672 records
- 11 columns
The final machine learning input contains 219 features.
Data Preparation
The overall workflow is:
Raw Kickstarter Dataset
          ↓
Data Cleaning
          ↓
Date Processing
          ↓
Feature Engineering
          ↓
Categorical Encoding
          ↓
Train / Validation / Test Split
          ↓
Model Training
          ↓
Threshold Optimization
          ↓
Final Model

Key preprocessing steps
- Cleaning the dataset
- Processing launch and deadline dates
- Calculating campaign duration
- Extracting year, month, day, and weekday information
- Creating project-name length features
- Encoding categorical variables
- Maintaining the exact 219-feature order during deployment
Machine Learning Models
The following classification approaches were explored:
- Logistic Regression
- Random Forest
- XGBoost
- HistGradientBoosting
- CatBoost
The final deployment model is a CatBoost Classifier.
Final CatBoost Configuration
Iterations        : 700
Depth             : 6
Learning Rate     : 0.08
L2 Regularization : 10
Class Weights     : [1, 1.2]
Random Seed       : 42

Decision Threshold Optimization
Instead of directly using the default probability threshold of 0.50, multiple thresholds were evaluated on the validation set.
The threshold that maximized the F1-score was selected.
Selected Threshold
Threshold     : 0.38
Validation F1 : 0.6608

Deployment Rule
Probability >= 0.38  →  SUCCESSFUL

Probability <  0.38  →  NOT SUCCESSFUL

The selected threshold is stored in:
models/threshold.json

Final Model Performance
The final model was evaluated on a held-out test set using the selected threshold of 0.38.
Metric	Result
Accuracy	66.29%
Precision	55.63%
Recall	81.69%
F1-Score	66.18%
ROC-AUC	76.53%


Confusion Matrix
                  Predicted
                0          1
Actual  0     22286      17458
        1      4906      21885

System Architecture
┌──────────────────────────────┐
│       User / Web Browser     │
│       HTML + CSS + JS        │
└──────────────┬───────────────┘
               │
               │ POST /predict
               ▼
┌──────────────────────────────┐
│       FastAPI Backend        │
│          app.py              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Prediction Pipeline     │
│       prediction.py          │
│                              │
│  Input validation            │
│  Feature construction        │
│  219-feature alignment       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       CatBoost Model         │
│  kickstarter_catboost_final  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Probability + Threshold 0.38 │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ SUCCESSFUL / NOT SUCCESSFUL  │
└──────────────────────────────┘

Web Application
The frontend accepts the following project details:
- Project name
- Funding goal
- USD funding goal
- Category
- Main category
- Currency
- Country
- Launch date
- Deadline
The frontend sends the project information to the FastAPI /predict endpoint.
The backend:
1. Validates the input.
2. Processes the project information.
3. Creates the required 219 features.
4. Aligns the features with the trained model.
5. Loads the CatBoost model.
6. Calculates the success probability.
7. Applies the 0.38 threshold.
8. Returns the prediction to the frontend.
API Endpoints
GET /
Checks whether the API is running.
GET /options
Returns the available categories, main categories, currencies, and countries used by the model.
POST /predict
Accepts project details and returns the prediction result.
Example Prediction
A held-out Kickstarter project was used to verify the complete prediction pipeline.
Project
Innocents, a truly terrifying roleplaying game

Project Details
Category       : Tabletop Games
Main Category  : Games
Currency       : USD
Country        : US

Prediction Result
Probability : 98.86%
Result      : SUCCESSFUL
Threshold   : 38%
Duration    : 7 days

This demonstrates the complete flow from project information through the backend and trained model to the final prediction.
Project Structure
kickstarter-success-prediction/
│
├── data/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── feature_names.json
│   ├── kickstarter_catboost_final.cbm
│   ├── model_metadata.json
│   └── threshold.json
│
├── notebook/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── results/
│   ├── plots/
│   ├── final_model_results.csv
│   └── random_forest_model.pkl
│
├── backend/
│   ├── app.py
│   ├── prediction.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── .gitattributes
└── README.md

Technologies Used
Programming Languages
- Python
- HTML
- CSS
- JavaScript
Machine Learning
- CatBoost
- Scikit-learn
- XGBoost
Data Processing
- Pandas
- NumPy
Backend
- FastAPI
- Uvicorn
Development and Analysis
- Jupyter Notebook
- VS Code
How to Run
1. Clone the Repository
git clone https://github.com/sanjayv-044/kickstarter-sucess-prediction.git
cd kickstarter-sucess-prediction

2. Create a Virtual Environment
For Windows:
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r backend/requirements.txt

4. Start the FastAPI Backend
From the project root:
uvicorn backend.app:app --reload

The API will run at:
http://127.0.0.1:8000

5. Open FastAPI Documentation
Open:
http://127.0.0.1:8000/docs

This provides the interactive Swagger API documentation.
6. Start the Frontend
Open:
frontend/index.html

using a local development server such as VS Code Live Server.
The frontend communicates with:
http://127.0.0.1:8000

Model Files
The models/ directory contains the files required for deployment.
kickstarter_catboost_final.cbm
The trained CatBoost classification model.
feature_names.json
Contains the exact 219 feature names and their expected order.
threshold.json
Stores the selected probability threshold:
0.38

model_metadata.json
Contains model-related metadata.
Limitations
- The prediction is probabilistic and cannot guarantee the actual outcome of a future campaign.
- Model performance depends on the historical training data.
- Real-world factors that are not represented in the dataset may influence campaign success.
- The model should be treated as a decision-support tool rather than a guarantee of campaign performance.
Future Scope
Possible future improvements include:
- Adding richer project-description text features.
- Incorporating campaign updates and engagement information.
- Adding more recent Kickstarter data.
- Adding explainable AI to show important factors behind predictions.
- Improving the user interface and deployment environment.
- Providing additional campaign planning insights.
Team
Project: Kickstarter Project Success Prediction using Machine Learning
Team Members
- Nanditha A — PES University
- Sanjay V — GitHub: sanjayv-044
Conclusion
This project demonstrates an end-to-end machine learning workflow for Kickstarter project success prediction, starting from data preprocessing and exploratory analysis through model development, threshold optimization, evaluation, and web deployment.
The final system uses a CatBoost classifier with 219 features and a tuned decision threshold of 0.38, and provides predictions through a FastAPI-powered web application.
