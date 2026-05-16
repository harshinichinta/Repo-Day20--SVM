import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("Support Vector Regression (SVR)")

# Create Dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# SVR Model
svm_regressor = SVR(
    kernel='rbf',
    C=100,
    gamma='scale',
    epsilon=0.1
)

# Train Model
svm_regressor.fit(X_train, y_train)

# Predictions
y_pred = svm_regressor.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Metrics
st.subheader("Model Evaluation")

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot
fig, ax = plt.subplots()

ax.scatter(X_test, y_test, color='blue', label='Actual Data')
ax.scatter(X_test, y_pred, color='red', label='Predicted Data')

ax.set_title("SVR Regression")
ax.set_xlabel("X Values")
ax.set_ylabel("Target Values")

ax.legend()

st.pyplot(fig)