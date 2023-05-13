# load the data
import pandas as pd
import numpy as np
train_df = pd.read_csv(r"C:\Users\nagak\Downloads\train.csv")
test_df = pd.read_csv(r"C:\Users\nagak\Downloads\test.csv")
# Drop columns that are not useful for our model
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# Handle missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
# One-hot encode categorical features
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'])
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'])
# model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X_train, X_val, y_train, y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], test_size=0.2)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on validation set: {accuracy}")
# model testing
test_pred = rfc.predict(test_df)
output = pd.DataFrame({'PassengerId': pd.read_csv(r"C:\Users\nagak\Downloads\test.csv")['PassengerId'], 'Survived': test_pred})
output.to_csv('submission.csv', index=False)
predictions = pd.read_csv('submission.csv')
print(predictions.to_string())
importances = rfc.feature_importances_
feature_names = train_df.drop('Survived', axis=1).columns
for feature_name, importance in zip(feature_names, importances):
    print(f"{feature_name}: {importance}")

