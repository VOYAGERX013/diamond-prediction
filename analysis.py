# Importing the necessary modules

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Getting all the csv files
csv_files = os.listdir("path/to/dataset")
print(csv_files)
# Creating an empty dataframe
df = pd.DataFrame()

# Filling data from all csv files to the dataframe
for file in csv_files:
    df_temporary = pd.read_csv("./dataset/" + file)
    df = df.append(df_temporary, ignore_index=True)

print(df.head())

# Dropping the calibration records
df = df[df["Diamond"].str.contains("Calibration") == False]

# Quantifying the diamond quality information
df.loc[df["Diamond"].str.contains("None", na=False), "Diamond"] = 0
df.loc[df["Diamond"].str.contains("Faint", na=False), "Diamond"] = 1
df.loc[df["Diamond"].str.contains("Medium", na=False), "Diamond"] = 2

# Extracting the features
X = df[["L*", "a*", "b*", "X", "Y", "Z", "Dominant Wavelenght", "Whiteness", "Purity", "Tint", "Chroma", "Hue", "Color Temperature"]]

# Extracting the target value
y = df["Diamond"]

y=y.astype('int')

# Performing train test split
split = int(0.8 * len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Using support vector classifier to train the model for classification
cls = SVC().fit(X_train, y_train)

# Get the accuracy of the model by trying it on the testing set
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print(f"Test Accuracy: {accuracy_test * 100}%")

# Saving the predictions for all records
df["prediction"] = cls.predict(X)

# Display the first five records
print(df.head())
