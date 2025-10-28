import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/content/Telcom_Customer_Churn.csv", engine='python', on_bad_lines='skip')
data

data.columns

data.shape

data.nunique()

data.isna().sum()

# Check the number of rows before removing duplicates
print("Number of rows before removing duplicates:", len(data))

# Remove duplicate records
data_cleaned = data.drop_duplicates()

# Measure of frequency destribution
unique, counts = np.unique(data['tenure'], return_counts=True)
print(unique, counts)

# Measure of frequency destribution
unique, counts = np.unique(data['MonthlyCharges'], return_counts=True)
print(unique, counts)

# Measure of frequency destribution
unique, counts = np.unique(data['TotalCharges'], return_counts=True)
print(unique, counts)

plt.boxplot(data['tenure'])
plt.show()

X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape
X_test.shape
y_test.shape

# Export the cleaned dataset to a CSV file
data.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)
