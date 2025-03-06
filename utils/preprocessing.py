import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert 'Year' to integer if it's not already
    df['Year'] = df['Year'].astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the dataset for analysis."""
    # Define crime columns based on the description.csv
    crime_columns = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
    
    # Create a new column for total crimes
    df['Total_Crimes'] = df[crime_columns].sum(axis=1)
    
    # Normalize the crime columns
    scaler = StandardScaler()
    df[crime_columns] = scaler.fit_transform(df[crime_columns])
    
    return df, crime_columns  # Return both the DataFrame and crime columns