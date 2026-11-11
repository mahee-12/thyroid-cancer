# 🩺 Thyroid Cancer Recurrence Prediction

Predicts whether differentiated thyroid cancer will recur (`Recurred`: Yes / No) from clinical,
demographic and histopathological variables, with a Streamlit app for live predictions.

## Dataset
Thyroid Differentiated Cancer dataset: 383 patients, 17 columns (age, gender, smoking,
radiotherapy history, thyroid function, adenopathy, pathology, focality, ATA risk, TNM,
stage, treatment response, recurrence).

## Approach
1. Data checks and EDA (recurrence by gender, risk category and treatment response)
2. Label encoding and correlation heatmap
3. Decision Tree and Random Forest classifiers
4. Evaluation with confusion matrix, classification report, train / test / 5-fold CV accuracy

## Key findings
- Treatment **Response**, ATA **Risk** and lymph node **Adenopathy** are the strongest drivers of recurrence.
- Random Forest: about 95% test accuracy, with train, test and CV close together.
- Response and Risk are assessed after treatment, so this works as a follow-up prognosis tool.

## Files
- `Thyroid_Cancer_Recurrence_Prediction_Model.ipynb` – full analysis and models
- `app.py` – Streamlit app (trains the model on startup)
- `Thyroid_Diff.csv` – dataset
- `requirements.txt` – dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit

## Author
Mohammed Maher Pasha
