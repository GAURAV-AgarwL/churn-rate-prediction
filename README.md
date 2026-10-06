# Customer Churn Analysis and Prediction

In this project I analyzed why customers are leaving a bank and built a model to predict which customers are likely to churn.

## Dataset

- Bank Customer Churn dataset (`Churn_Modelling.csv`)
- 10,000 customers, 14 columns
- Target column: `Exited` (1 = left the bank, 0 = stayed)

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow (Keras), Tableau

## What I Did

1. Checked data quality (no missing values or duplicates)
2. Removed ID columns that don't affect churn
3. Did EDA to find churn patterns by country, gender, age, activity, products and balance
4. Wrote down key insights and business recommendations
5. Built a Logistic Regression model as a baseline
6. Built an ANN with class weights and early stopping to handle imbalanced data
7. Compared both models using recall, F1 score, confusion matrix and ROC-AUC
8. Gave every customer a churn risk score (Low / Medium / High) and exported it for a Tableau dashboard

## Key Insights

- **20.4%** of customers churned, and they held **24%** of the bank's total balance
- **Germany** has double the churn rate (**32%**) of France and Spain (**16 to 17%**)
- **Age** is the biggest factor. Churn goes from **7.5%** (ages 18 to 30) to **56%** (ages 51 to 60)
- **Inactive members** churn almost twice as much as active ones (**27% vs 14%**)
- Customers with **2 products** churn the least (**8%**), while most customers with **3 or 4 products** left
- **Female** customers churn more than male customers (**25% vs 17%**)

## Recommendations

- Look into the German market first since it has the highest churn and highest balances
- Give retention offers to customers aged 40 to 60
- Run re-engagement campaigns for inactive members
- Cross-sell a second product to single-product customers

## Model Results

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 71% | 70% | 0.50 | 0.78 |
| ANN | 80% | 74% | 0.60 | 0.86 |

I focused on **recall** instead of accuracy because the data is imbalanced, and missing a customer who is about to leave costs the bank more than a false alarm. Customers marked as **High risk** by the model actually churned at **70%**.

## Dashboard

Tableau dashboard: [Link](#)

## Files

- `Churn Rate Prediction.ipynb` : full analysis and models
- `Churn_Modelling.csv` : dataset
- `churn_dashboard_data.csv` : customer data with churn risk scores, used for the dashboard

## How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Then open `Churn Rate Prediction.ipynb` in Jupyter and run all cells.
