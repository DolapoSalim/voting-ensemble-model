import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target  # Classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

# Convert to DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
print(df.head())  # Show the first few rows

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. Make predictions
y_pred = rf.predict(X_test)

# 5. Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# 6. Feature Importance Visualization
importances = rf.feature_importances_
plt.bar(iris.feature_names, importances, color='blue', edgecolor='k')
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest (Iris Dataset)")
plt.show()