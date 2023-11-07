import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the CSV file
data = pd.read_csv('lead_conversion_data.csv')

# Features (X) and target variable (y)
X = data[["Lead_Source", "Lead_Score", "Time_Spent (minutes)"]]
y = data["Conversion"]

# Convert lead source to numerical values using one-hot encoding
X = pd.get_dummies(X, columns=["Lead_Source"], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Conversion", "Conversion"])

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(report)