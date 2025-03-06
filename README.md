# Crime Prediction and Analysis

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)

This project focuses on analyzing and predicting crimes against women using data mining techniques. It leverages machine learning models to predict future trends in crime rates and provides a user-friendly Streamlit application for visualization and analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The goal of this project is to analyze historical crime data against women and predict future trends using machine learning. The application provides insights into crime patterns, visualizes trends over time, and allows users to make predictions based on input features.

---

## Features

1. **Dataset Overview**:
   - View the raw dataset with filtering options by state or year.
   - Display statistics such as the number of rows, unique states, and years covered.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize crime trends over the years.
   - Analyze the distribution of different types of crimes.
   - Compare crime trends across multiple states.

3. **Crime Prediction**:
   - Predict total crimes for a given year based on normalized crime category values.
   - Compare predicted values with historical data.

4. **Interactive Interface**:
   - Built using Streamlit for an intuitive and interactive user experience.

---

## Dataset

The dataset used in this project contains the following columns:

| Column Name | Explanation                     |
|-------------|---------------------------------|
| State       | Name of the state              |
| Year        | Year of the record             |
| Rape        | Number of rape cases           |
| K&A         | Kidnap and assault cases       |
| DD          | Dowry deaths                   |
| AoW         | Assault against women          |
| AoM         | Assault against modesty of women |
| DV          | Domestic violence cases        |
| WT          | Women trafficking cases        |

The dataset is stored in `datasets/CrimesOnWomenData.csv`.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Divith123/Crime-Predication-Analysis.git
   cd Crime-Predication-Analysis
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:
   ```bash
   python train.py
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Dataset Overview
- View the dataset with filtering options for state or year.
- Understand the structure and contents of the dataset.

### Exploratory Data Analysis
- Analyze crime trends over the years.
- Compare crime trends across multiple states.
- Visualize the distribution of different types of crimes.

### Crime Prediction
- Input a year and normalized crime category values to predict total crimes.
- Compare the predicted value with historical data.

---

## Project Structure

```
Crime-Predication-Analysis/
│
├── app.py                     # Streamlit application file
├── train.py                   # Training script
├── requirements.txt           # Python dependencies
├── datasets/                  # Folder for datasets
│   ├── CrimesOnWomenData.csv  # Dataset file
│   └── description.csv        # Column descriptions
├── models/                    # Folder for trained ML models
│   └── crime_prediction_model.pkl
├── utils/                     # Utility functions
│   └── preprocessing.py       # Data preprocessing functions
└── visuals/                   # Visualization functions
    └── plots.py               # Plotting functions
```

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m "Add YourFeatureName"`).
4. Push to the branch (`git push origin feature/"YourFeatureName"`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, feel free to reach out:

- **GitHub**: [@Divith123](https://github.com/Divith123) | [@dilanmelvin](https://github.com/dilanmelvin)
- **Email**: divithselvam23@gmail.com | dilan4524melvin@gmail.com

---

## Acknowledgments

- Inspired by real-world applications of data mining in crime analysis.
- Thanks to open-source libraries like Scikit-learn, Streamlit, and Pandas.

---
