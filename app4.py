import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Title
st.title("KNN Regression using California Housing Dataset")

# Load Dataset
housing = fetch_california_housing()

x = housing.data
y = housing.target

# Create DataFrame
df = pd.DataFrame(x, columns=housing.feature_names)
df['Target'] = y

# Display Dataset
st.subheader("Dataset")
st.dataframe(df)

# Sidebar Inputs
st.sidebar.header("Model Parameters")

k = st.sidebar.slider("Select K Value", 1, 20, 5)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=42
)

# Create KNN Regressor
knn_regressor = KNeighborsRegressor(
    n_neighbors=k,
    metric='minkowski',
    p=2
)

# Train Model
knn_regressor.fit(x_train, y_train)

# Prediction
y_pred = knn_regressor.predict(x_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Results
st.subheader("Model Performance")

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R2 Score: {r2:.2f}")

# Actual vs Predicted Plot
st.subheader("Actual vs Predicted Values")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred)

ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)