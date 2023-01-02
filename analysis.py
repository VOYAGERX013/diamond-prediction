# Importing the necessary modules

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Importing the ensemble module from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Getting all the csv files
csv_files = os.listdir("C:/Users/Aryaan/OneDrive/Desktop/Internship/dataset")

# Creating an empty dataframe
df = pd.DataFrame()

# Filling data from all csv files to the dataframe
for file in csv_files:
    df_temporary = pd.read_csv("./dataset/" + file)
    df = df.append(df_temporary, ignore_index=True)

# Renaming all the unnamed columns
for col in df.columns:
    if str(col).replace(" ", "").replace(".", "").isnumeric():
        col_index = df.columns.get_loc(col)
        df.rename(columns={col: f"Column {col_index + 1}"}, inplace=True)

# Dropping the last column which contains mostly NaN values
df = df.drop(columns=["Column 1060"])

# Display the first five records from the database
print(df.head())

# Performing EDA on the given data (retrieving standard deviation, mean, count and other statistics)
print(df.describe())

# Dropping the calibration records
df = df[df["Diamond"].str.contains("Calibration") == False]

# Quantifying the diamond quality information
df.loc[df["Diamond"].str.contains("None", na=False), "Diamond"] = 0
df.loc[df["Diamond"].str.contains("Faint", na=False), "Diamond"] = 1
df.loc[df["Diamond"].str.contains("Medium", na=False), "Diamond"] = 2

# Analyze the data through the correlations of each field with each other (more EDA)
correlations = df.corr()
print(correlations)

# Getting the numerical columns from the database
renamed_columns = [col for col in df.columns if col.startswith("Column")]

# Assigning the feature variables to a value X
X = df[["L*", "a*", "b*", "X", "Y", "Z", "Dominant Wavelenght", "Whiteness", "Purity", "Tint", "Chroma", "Hue", "Color Temperature"] + renamed_columns]

# Extracting the target value
y = df["Diamond"]

y=y.astype('int')

# Split the data into training and testing sets with 20% of it being testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Call the Random Forest classifier
cls = RandomForestClassifier()

# Train the classifier on the training data
cls.fit(X_train, y_train)

training_accuracy = accuracy_score(y_train, cls.predict(X_train))
testing_accuracy = accuracy_score(y_test, cls.predict(X_test))

# Compare these two accuracies to determine if model is overfit or underfit
print(f"Training data has {training_accuracy * 100}%")
print(f"Testing data has {testing_accuracy * 100}%")

# Saving the predictions for all records
df["prediction"] = cls.predict(X)

# See first five records
print(df.head())
