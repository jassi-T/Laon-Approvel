# Loan Approval Prediction using Support Vector Machine (SVM)

## Project Description

This project builds a machine learning model to predict whether a loan
application will be **Approved** or **Rejected**. It uses a Support
Vector Machine (SVM) classifier with an RBF kernel and a loan applicant
dataset.

The notebook demonstrates the complete workflow, including data
exploration, cleaning, preprocessing, visualization, model training,
evaluation, and prediction for a new applicant.

## Project Workflow

1.  Load the loan dataset using Pandas.
2.  Explore the dataset's shape, data types, missing values, duplicates,
    and value distributions.
3.  Clean the data:
    -   Remove duplicate rows.
    -   Convert income values to numeric format.
    -   Treat credit scores above 900 as invalid values.
    -   Fill missing credit scores and years of work experience using
        their median values.
    -   Standardize employment and loan-status labels.
4.  Encode employment status:
    -   Salaried = `1`
    -   Self-employed = `0`
5.  Compare numerical feature averages by loan status.
6.  Visualize credit score against loan amount, using colors to
    distinguish approved and rejected applications.
7.  Split the data into training and testing sets (80:20), using
    stratification.
8.  Standardize the features with `StandardScaler`.
9.  Train an SVM classifier using an RBF kernel (`C=1`).
10. Evaluate the model and predict the outcome for a new applicant.

## Features Used

The model uses these five input features:

  Feature           Description
  ----------------- -------------------------------------------------------
  `income`          Applicant's income
  `credit_score`    Applicant's credit score
  `loan_amount`     Requested loan amount
  `years_working`   Applicant's years of work experience
  `employment`      Employment category (Salaried = 1, Self-employed = 0)

**Target:** `loan_status` (`Approved` or `Rejected`)

## Model

-   Algorithm: Support Vector Machine (SVM)
-   Classifier: `sklearn.svm.SVC`
-   Kernel: RBF
-   C parameter: `1`
-   Feature scaling: `StandardScaler`
-   Train/test split: 80% / 20%
-   Random state: `42`
-   Stratification: Enabled

## Results

The notebook's recorded evaluation results are:

  Metric                Result
  ------------------- --------
  Training accuracy      90.8%
  Testing accuracy       93.3%

### Confusion Matrix

  Actual  Predicted     Approved   Rejected
  ------------------- ---------- ----------
  Approved                    13          1
  Rejected                     1         15

The test set contains 30 records. These results are from the dataset and
notebook run included in this project; performance on other data may
differ.

## Example Prediction

The notebook tests a sample applicant with the following details:

-   Income: `800000`
-   Credit score: `760`
-   Loan amount: `300000`
-   Years working: `6`
-   Employment: `1` (Salaried)

**Recorded prediction:** `Approved`

## Requirements

Install the required Python libraries:

``` bash
pip install pandas matplotlib scikit-learn jupyter
```

## How to Run

1.  Download or clone this repository.

2.  Keep `loan(1).ipynb` and `loan_approval.csv` in the same folder.

3.  Install the required libraries.

4.  Open Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

5.  Open `loan(1).ipynb` and run the cells from top to bottom.

## Dataset

The notebook loads a CSV file named `loan_approval.csv`:

``` python
df = pd.read_csv("loan_approval.csv")
```

Make sure this file is included in the repository or update the path in
the notebook. The notebook expects the columns `income`, `credit_score`,
`loan_amount`, `years_working`, `employment`, and `loan_status`.

## Project Structure

``` text
loan-approval-prediction/
├── loan(1).ipynb
├── loan_approval.csv
└── README.md
```

## Limitations

-   The model's performance depends on the quality and
    representativeness of the dataset.
-   The recorded test accuracy is not a guarantee of real-world loan
    decisions.
-   This is an educational machine learning project, not a production
    lending or credit-risk system.
-   The dataset should be checked for fairness, data quality, and
    suitability before any real-world use.

## Author

Harpreet Singh and GitHub profile jass-T.

## Live Demo 

You can access my demo from here
https://laon-approvel-ldheuvgdlavpyprnetbrfs.streamlit.app/
