import matplotlib.pyplot as plt
import seaborn as sns

def plot_crime_trends(df):
    """Plot trends in crimes over the years."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Total_Crimes', data=df, ci=None)
    plt.title('Trends in Crimes Against Women Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Crimes')
    plt.grid()
    return plt

def plot_crime_distribution(df):
    """Plot distribution of different types of crimes."""
    crime_columns = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
    crime_counts = df[crime_columns].sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=crime_counts.index, y=crime_counts.values)
    plt.xticks(rotation=45)
    plt.title('Distribution of Different Types of Crimes')
    plt.xlabel('Crime Type')
    plt.ylabel('Number of Cases')
    plt.grid()
    return plt