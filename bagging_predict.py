import pickle
import pandas as pd
from BaggingRegressor import DecisionTreeRegressor, BaggingRegressor


with open("sri_bagging_model_v2.pkl", "rb") as file:
    bagging_model = pickle.load(file)

def predict_yield():
    # Load the saved model


    # Define input fields
    print("Enter the following details for prediction:")
    crop = input("Crop: ")
    district = input("District: ")
    year = int(input("Year: "))
    season = input("Season: ")
    area = float(input("Area (Hectare): "))
    annual_temp = float(input("Annual Temp (°C): "))
    fertilizer = float(input("Fertilizer (KG per Hectare): "))
    annual_rainfall = float(input("Annual Rainfall (Millimeters): "))

    # Create a dataframe for the input
    input_data = pd.DataFrame({
        'Crop': [crop],
        'District': [district],
        'Year': [year],
        'Season': [season],
        'Area(Hectare)': [area],
        'Annual_Temp': [annual_temp],
        'Fertilizer(KG_per_hectare)': [fertilizer],
        'ANNUAL_RAINFALL(Millimeters)': [annual_rainfall]
    })

    # One-hot encode the input (aligning with the training dataset)
    original_data = pd.read_csv('data.csv')  # Load the original data to get the full feature set
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    original_features = pd.get_dummies(original_data.drop(['Yield(Tonne/Hectare)'], axis=1), drop_first=True)

    # Align input features with the model's training features
    for col in original_features.columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0  # Add missing columns with default value 0

    # Ensure column order matches the training data
    input_data_encoded = input_data_encoded[original_features.columns]

    # Predict using the model
    prediction = bagging_model.predict(input_data_encoded.values)

    # Output the predicted yield
    print(f"Predicted Yield (Tonne/Hectare): {prediction[0]:.2f}")


if __name__ == "__main__":
    predict_yield()
