import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load dataset
Cancer_data = load_breast_cancer()

x = Cancer_data.data
y = Cancer_data.target

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = KNeighborsRegressor(n_neighbors=3)

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("KNN Regression Cancer Prediction")

st.write("### Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R² Score:", r2)

# Display predictions
results = pd.DataFrame({
    "Actual": y_test[:10],
    "Predicted": y_pred[:10]
})

st.write("### Sample Predictions")
st.dataframe(results)