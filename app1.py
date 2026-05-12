import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Title
st.title("SVM Classification on Iris Dataset")

# Load dataset
data = load_iris()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create SVM model
svm_classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)

# Train model
svm_classifier.fit(X_train, y_train)

# Predictions
y_pred = svm_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.write("## Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Show predictions table
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.write("## Prediction Results")
st.dataframe(results)