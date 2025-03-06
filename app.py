import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import load_data, clean_data, preprocess_data
from visuals.plots import plot_crime_trends, plot_crime_distribution
import joblib

# Load and preprocess data
df = load_data('datasets/CrimesOnWomenData.csv')
df = clean_data(df)
df, crime_columns = preprocess_data(df)

# Load the pre-trained model
model = joblib.load('models/crime_prediction_model.pkl')

# Title
st.title("Crime Against Women - Analysis and Prediction")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Overview", "Analysis", "Prediction"])

if menu == "Overview":
    st.header("Dataset Overview")
    
    # Display dataset statistics
    st.write(f"Number of Rows: {len(df)}")
    st.write(f"Number of Unique States: {df['State'].nunique()}")
    st.write(f"Years Covered: {df['Year'].min()} to {df['Year'].max()}")
    
    # Filter by State or Year
    filter_option = st.radio("Filter Dataset By:", ["None", "State", "Year"])
    if filter_option == "State":
        selected_state = st.selectbox("Select State", df['State'].unique())
        filtered_df = df[df['State'] == selected_state]
    elif filter_option == "Year":
        selected_year = st.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()))
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Display filtered dataset
    st.dataframe(filtered_df)

elif menu == "Analysis":
    st.header("Exploratory Data Analysis")
    
    # Plot crime trends
    st.subheader("Crime Trends Over the Years")
    fig1 = plot_crime_trends(df)
    st.pyplot(fig1)
    
    # Plot crime distribution
    st.subheader("Distribution of Different Types of Crimes")
    fig2 = plot_crime_distribution(df)
    st.pyplot(fig2)
    
    # Compare crime trends across states
    st.subheader("Compare Crime Trends Across States")
    selected_states = st.multiselect("Select States for Comparison", df['State'].unique(), default=df['State'].unique()[:3])
    filtered_df = df[df['State'].isin(selected_states)]
    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        for state in selected_states:
            state_data = filtered_df[filtered_df['State'] == state]
            sns.lineplot(x='Year', y='Total_Crimes', data=state_data, label=state)
        plt.title('Crime Trends Across Selected States')
        plt.xlabel('Year')
        plt.ylabel('Total Crimes')
        plt.grid()
        st.pyplot(plt)

elif menu == "Prediction":
    st.header("Crime Prediction")
    
    # Input year for prediction
    year = st.number_input("Enter Year for Prediction", min_value=int(df['Year'].max()) + 1, max_value=2050, value=int(df['Year'].max()) + 1)
    
    # Input crime category values
    st.subheader("Enter Crime Category Values (Normalized)")
    inputs = {}
    for col in crime_columns[:-1]:  # Exclude 'Total_Crimes'
        inputs[col] = st.number_input(f"{col}", value=0.0, step=0.01)
    
    # Prepare input for prediction
    input_data = [[year] + list(inputs.values())]
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Total Crimes in {year}: {int(prediction[0])}")
    
    # Compare with previous years
    st.subheader("Comparison with Previous Years")
    previous_years_data = df[df['Year'] >= year - 5][['Year', 'Total_Crimes']].sort_values(by='Year')
    if not previous_years_data.empty:
        previous_years_data.loc[len(previous_years_data)] = [year, int(prediction[0])]  # Add predicted year
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Year', y='Total_Crimes', data=previous_years_data, marker='o')
        plt.title('Comparison of Predicted Crimes with Previous Years')
        plt.xlabel('Year')
        plt.ylabel('Total Crimes')
        plt.grid()
        st.pyplot(plt)
