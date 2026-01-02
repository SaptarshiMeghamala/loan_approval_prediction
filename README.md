# Loan Approval Prediction Project

This is my beginner machine learning project where I tried to predict
whether a loan will be approved or not based on applicant details.

I built this project to understand how classification models work
and how machine learning is applied to real-life problems like banking.

## What this project does
The model takes details like income, loan amount, credit score,
employment status, education level, and loan term
and predicts whether the loan is approved or rejected.

## Dataset Information
The dataset contains the following columns:
- applicant_income
- loan_amount
- credit_score
- employment_status (Yes / No)
- education (Yes / No)
- loan_term_months

Target column:
- loan_approved (Yes / No)

## Model Used
- Logistic Regression

I chose Logistic Regression because the output is binary
(loan approved or not approved).

## Steps I followed
- Loaded the dataset using pandas
- Converted categorical values into numerical form
- Applied MinMax scaling on numerical features
- Split the data into training and testing sets
- Trained the model using Logistic Regression
- Evaluated the model using accuracy and confusion matrix
- Tested the model with user input

## What I learned
- Difference between regression and classification
- Why scaling is important
- How train-test split works
- How Logistic Regression makes predictions

## Result
The model predicts loan approval based on the input values.
This project helped me understand the complete machine learning workflow.

## Note
This project is part of my learning journey in machine learning.
