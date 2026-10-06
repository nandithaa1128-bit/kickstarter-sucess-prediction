# Kickstarter Project Success Prediction

A machine learning project that predicts whether a Kickstarter campaign will be successful, based on details you'd know before launching. It comes with a small web app where you can type in a project and get a prediction.

Made by Nanditha A and Sanjay V, PES University.

## What it does

You enter:

- project name
- funding goal and USD goal
- category and main category
- currency and country
- launch date and deadline

You get back:

- SUCCESSFUL or NOT SUCCESSFUL
- the probability of success
- the campaign duration

## Dataset

- Started with 378,661 Kickstarter projects (15 columns)
- 331,672 records left after cleaning
- 219 features go into the final model

Features come from the dates (year, month, day, weekday, campaign duration), the length of the project name, and the encoded categorical columns (category, main category, currency, country).

## Models

I tried Logistic Regression, Random Forest, XGBoost, HistGradientBoosting and CatBoost. CatBoost is the one used in the final app.

Final CatBoost settings:

| Parameter | Value |
|---|---|
| Iterations | 700 |
| Depth | 6 |
| Learning rate | 0.08 |
| L2 regularization | 10 |
| Class weights | [1, 1.2] |
| Random seed | 42 |

### Threshold

The default cutoff of 0.50 isn't used. I tried a range of thresholds on the validation set and picked the one with the best F1-score, which was **0.38** (validation F1 = 0.6608).

- probability >= 0.38 -> SUCCESSFUL
- probability < 0.38 -> NOT SUCCESSFUL

## Results (test set)

| Metric | Score |
|---|---|
| Accuracy | 66.29% |
| Precision | 55.63% |
| Recall | 81.69% |
| F1-score | 66.18% |
| ROC-AUC | 76.53% |

Recall is much higher than precision, so the model catches most successful projects but also marks some failing ones as successful. This is partly because of the lower threshold. The model only sees basic project details, not things like the description, images or updates, so don't treat the output as a guarantee.

## How the app works

```
Browser (HTML, CSS, JS)
        |
   FastAPI backend
        |
 prediction pipeline  ->  219 features
        |
 CatBoost model  ->  probability
        |
 threshold 0.38  ->  SUCCESSFUL / NOT SUCCESSFUL
```

## Screenshots

Add your screenshots here:

- input page
- successful prediction
- not successful prediction

## Running it

Install the requirements:

```bash
pip install -r requirements.txt
```

Start the backend (CHECK: change `main:app` to your actual file name and app variable):

```bash
uvicorn main:app --reload
```

Then open the frontend page in your browser (CHECK: write which file to open, or the local URL such as `http://127.0.0.1:8000`).

## Tech used

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, FastAPI, Uvicorn, HTML, CSS, JavaScript, Jupyter Notebook, VS Code

## Possible improvements

- use the project description text as a feature
- train on more recent Kickstarter data
- add explainability so users can see why a project got its score
- include campaign engagement data
- improve the deployment and the UI
