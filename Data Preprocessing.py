import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
from scipy.stats import gmean

# Set the file path
file_path = r"C:/Users/Harsh/OneDrive/Desktop/python practice/Iris.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

# Load data
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()


# Preprocessing function
def preprocess_data(data, fill_null=1, scaler_val=1, apply_pca=0, n_components=None):
    try:
        # Handle missing numeric values
        if fill_null == 0:
            data.dropna(inplace=True)  # Drop rows with null values
        elif fill_null == 1:
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing_values = data[col].dropna()

                # Ensure only positive values are used for gmean()
                positive_values = non_missing_values[non_missing_values > 0]

                if len(positive_values) > 0:
                    geometric_mean = gmean(positive_values)

                    # Fix incompatible dtype issue
                    if np.issubdtype(data[col].dtype, np.integer):
                        data.loc[data[col].isna(), col] = int(geometric_mean)  # Convert to int
                    else:
                        data.loc[data[col].isna(), col] = geometric_mean  # Keep float

        # Scale numeric columns
        if scaler_val == 1:
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            joblib.dump(scaler, "scaler.pkl")

        # Encode categorical variables
        label_encoders = {}
        for col in data.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        joblib.dump(label_encoders, "encoder.pkl")

        # Apply PCA
        if apply_pca == 1:
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(data[numeric_cols])

            # Create DataFrame with PCA components
            pca_columns = [f"PCA_{i + 1}" for i in range(pca_data.shape[1])]
            pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)

            # Drop original numeric columns and add PCA columns
            data.drop(columns=numeric_cols, inplace=True)
            data = pd.concat([data, pca_df], axis=1)
            joblib.dump(pca, "PCA.pkl")

        return data
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


if __name__ == "__main__":
    # Process the data
    processed_data = preprocess_data(data)

    if processed_data is not None:
        # Save the preprocessed data
        processed_data_file = "processed_data7.csv"
        processed_data.to_csv(processed_data_file, index=False)
        print(f"Preprocessed data saved as {processed_data_file}")

