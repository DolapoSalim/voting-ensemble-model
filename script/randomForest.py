# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # 1. Generate a synthetic dataset
# X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)

# # 2. Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. Create and train a RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # 4. Make predictions
# y_pred = rf.predict(X_test)

# # 5. Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Accuracy: {accuracy:.2f}")

# # 6. Feature Importance Visualization
# importances = rf.feature_importances_
# plt.bar(range(len(importances)), importances, color='blue', edgecolor='k')
# plt.xlabel("Feature Index")
# plt.ylabel("Importance Score")
# plt.title("Feature Importance in Random Forest")
# plt.show()