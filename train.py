import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from utils.preprocessing import load_data, clean_data, preprocess_data
from math import sqrt  # Import the square root function

# Load the dataset
data_path = 'datasets/CrimesOnWomenData.csv'
df = load_data(data_path)

# Clean and preprocess the data
df = clean_data(df)
df, crime_columns = preprocess_data(df)  # Unpack both values here

# Prepare features and target
X = df[['Year'] + crime_columns[:-1]]  # Exclude 'Total_Crimes' from features
y = df['Total_Crimes']  # Target: Total Crimes

# Debugging: Print features and target
print("Features used:", X.columns)
print("Target variable:", y.name)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models and evaluate their performance
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

best_model = None
best_score = -float('inf')

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)  # Compute MSE
    rmse = sqrt(mse)  # Compute RMSE manually
    
    print(f"Model: {name}")
    print(f"  R^2 Score: {r2:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Track the best model
    if r2 > best_score:
        best_score = r2
        best_model = model

# Save the best model
model_path = 'models/crime_prediction_model.pkl'
joblib.dump(best_model, model_path)
print(f"\nBest Model saved to {model_path} with R^2 Score: {best_score:.2f}")