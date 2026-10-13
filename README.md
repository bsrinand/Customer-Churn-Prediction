# Telco Customer Churn Prediction

A classification project that predicts which telecom customers are likely to cancel their service, so a company could reach out to them before they leave.

This repo covers stage 1: cleaning the data, preprocessing it and training the first models.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. It has 7,043 customers and 21 columns: account details, the services each customer uses, billing info, and whether they churned. About 26.5% of customers churned.

## What I did

**Cleaning**
- Dropped `customerID`, since it doesn't help with prediction.
- `TotalCharges` was stored as text and had 11 blank values. All 11 were brand-new customers (tenure 0) who hadn't been billed yet, so I set them to 0 and converted the column to numbers.

**Encoding**
- Yes/No columns and gender were label encoded to 0/1.
- For the add-on services (OnlineSecurity, StreamingTV and so on), "No internet service" was merged into "No". The `InternetService` column already holds that information, so keeping it would only add duplicate columns.
- InternetService, Contract and PaymentMethod were one-hot encoded.

**Split and scaling**
- 80/20 train/test split.
- `StandardScaler` on tenure, MonthlyCharges and TotalCharges, fitted on the training set only.

## Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.811 | 0.626 | 0.586 | 0.858 |
| SVM (RBF kernel) | 0.812 | 0.640 | 0.546 | — |
| Logistic Regression, threshold 0.3 | ~0.762 | ~0.512 | ~0.805 | 0.858 |

Precision and recall are for the churn class.

## Takeaways

- Accuracy tops out around 80% on this dataset whatever model you use. Predicting "no churn" for everyone already gets about 73%, so accuracy on its own doesn't say much here.
- Recall matters more for this problem. Missing a customer who's about to leave costs far more than sending a retention offer to someone who would have stayed.
- Lowering the decision threshold from 0.5 to 0.3 raised recall from about 59% to 80%. ROC-AUC stays the same because the model's ranking of customers doesn't change, only the cut-off.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, Google Colab

## How to run

1. Open `Telco Churn Prediction.ipynb` in Google Colab or Jupyter.
2. Download the CSV from the Kaggle link above and upload it next to the notebook.
3. Run all cells.

