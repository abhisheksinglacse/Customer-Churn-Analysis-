# Import required libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
# OneHotEncoder is not needed if using pd.get_dummies()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
telco_demog = pd.read_csv('telecom_demographics.csv')
telco_usage = pd.read_csv('telecom_usage.csv')

# Join data
churn_df = telco_demog.merge(telco_usage, on='customer_id')

# Identify churn rate
churn_rate = churn_df['churn'].value_counts() / len(churn_df)
print(churn_rate)

# Identify categorical variables
print(churn_df.info())

# One Hot Encoding for categorical variables
churn_df = pd.get_dummies(churn_df, columns=['telecom_partner', 'gender', 'state', 'city', 'registration_event'])

# Feature Scaling
scaler = StandardScaler()

# 'customer_id' is not a feature
features = churn_df.drop(['customer_id', 'churn'], axis=1)
features_scaled = scaler.fit_transform(features)

# Target variable
target = churn_df['churn']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Logistic Regression predictions
logreg_pred = logreg.predict(X_test)

# Logistic Regression evaluation
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

# Instantiate the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Random Forest predictions
rf_pred = rf.predict(X_test)

# Random Forest evaluation
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Which accuracy score is higher? Ridge or RandomForest
higher_accuracy = "RandomForest"