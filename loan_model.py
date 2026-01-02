import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report


df = pd.read_csv('loan_approval_prediction/loan_approval_prediction.csv')
print(df.head())


le = LabelEncoder()
df['employment_status'] = le.fit_transform(df['employment_status'])
df['education'] = le.fit_transform(df['education'])
df['loan_approved'] = le.fit_transform(df['loan_approved'])

scaler = MinMaxScaler()
min_coloumns = ['applicant_income' ,'loan_amount' ,'credit_score']
df[min_coloumns] = scaler.fit_transform(df[min_coloumns])

x = df[['applicant_income' ,'loan_amount' ,'credit_score' ,'employment_status' ,'education' , 'loan_term_months']]
y = df['loan_approved']

x_train  , x_test , y_train  , y_test = train_test_split(
    x,y , test_size = 0.2 , random_state= 42
) 
model = LogisticRegression()
model.fit(x_train , y_train)
prediction = model.predict(x_test)
print("Classification Report:\n", classification_report(y_test, prediction))

cm = confusion_matrix(y_test , prediction)
print("Confusion Matrix:\n", cm)
print("Accuracy: ", accuracy_score(y_test , prediction))

# ------------------ USER INPUT ------------------
print("--------- Predict Loan Approval ---------")

try:
    applicant_income = float(input("Enter applicant income: "))
    loan_amount = float(input('enter  loan amount :'))
    credit_score = float(input('enter credit score :'))
    employment_status = int(input('enter employment status (0 or 1) :'))
    education = int(input('enter education level (0 or 1) :'))
    loan_term_months = int(input('enter loan term in months :'))
    user_input = pd.DataFrame([[applicant_income, loan_amount, credit_score, employment_status, education, loan_term_months]])
    user_input.columns = ['applicant_income' ,'loan_amount' ,'credit_score' ,'employment_status' ,'education' , 'loan_term_months']

    user_input[min_coloumns] = scaler.transform(user_input[min_coloumns])
    print("Scaled User Input:\n", user_input)

    result = model.predict(user_input)[0]
    if  result == 1:
        print("Loan Approved")
    else:
        print("Loan Not Approved")

except Exception as e:
    print("Error:", e)
