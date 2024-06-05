from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

# fetch dataset
parkinsons = fetch_ucirepo(id=174)

# data (as pandas dataframes)
X = parkinsons.data.features
y = parkinsons.data.targets

# metadata
print(parkinsons.metadata)


# Print metadata
#print("UCI ID:", parkinsons.metadata.uci_id)
#print("Number of Instances:", parkinsons.metadata.num_instances)
#print("Summary:", parkinsons.metadata.additional_info.summary)

# variable information
#print(parkinsons.variables)

# Example: Display first few rows of features and targets
print(X.head())
print(y.head())



#model = LinearRegression()
#model.fit(X, y)
#print("Model trained successfully.")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train a classifier


# Initialize the classifier
classifier = LogisticRegression(random_state=42)

# Train the classifier
classifier.fit(X_train_scaled, y_train)

# Step 3: Evaluate the classifier
y_pred = classifier.predict(X_test_scaled)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Plotting the confusion matrix


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()





