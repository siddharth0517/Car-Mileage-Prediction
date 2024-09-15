# Car Mileage Prediction Using Linear and Polynomial Regression

This project aims to predict the fuel efficiency (measured in miles per gallon, MPG) of cars based on features such as horsepower, weight, and engine displacement. Both **linear regression** and **polynomial regression** models are used for prediction, and their performances are compared.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

## Project Overview
Fuel efficiency is a critical factor when purchasing a vehicle. This project explores the use of regression techniques to predict the MPG of cars based on features such as:
- **Horsepower**
- **Weight**
- **Displacement**
- **Cylinders**

The performance of linear and polynomial regression models is compared using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**.

## Dataset
The dataset used in this project is the **Auto MPG Dataset**, which contains information on car features and fuel efficiency. You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) or use any similar dataset.

### Dataset Features:
- `mpg`: Miles per gallon (target variable)
- `cylinders`: Number of cylinders
- `displacement`: Engine displacement (cubic inches)
- `horsepower`: Engine horsepower
- `weight`: Vehicle weight (lbs)
- `acceleration`: Time to accelerate from 0 to 60 mph (seconds)
- `model year`: Year of the car model
- `origin`: Origin of the car (categorical)
- `car name`: Car name (irrelevant for modeling)

## Preprocessing
- Replaced missing values in the `horsepower` column with the mean value.
- Normalized features such as `weight`, `horsepower`, and `displacement` using standardization.
- Removed irrelevant columns (e.g., `car name`).

## Modeling
Two types of regression models were used:
1. **Linear Regression**: The relationship between features and MPG is assumed to be linear.
2. **Polynomial Regression**: Higher-degree polynomial features were added to capture the nonlinear relationships between features and MPG.

### Steps:
- Data was split into training and testing sets.
- Linear regression and polynomial regression models were trained on the training set.
- Predictions were made on the test set for both models.

## Evaluation
The models were evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted MPG values.
- **R-squared (R²)**: Indicates how well the model explains the variability of the target variable.

| Model                | MSE        | R²          |
|----------------------|------------|-------------|
| Linear Regression     | *19.0035*    | *0.69*     |
| Polynomial Regression | *28.37*    | *0.54*     |

## Visualization
The predictions from both models were visualized and compared using scatter plots and line charts. Tableau was used for interactive data visualization.

![image](https://github.com/user-attachments/assets/9db8af10-578f-4da7-bb72-a2aa47508e0f)


## Technologies Used
- **Python**
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
- **Tableau** for visualization

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/car-mileage-prediction.git
    cd car-mileage-prediction
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script to train and evaluate the models:
    ```bash
    jupyter notebook car_mileage_prediction.ipynb
    ```

4. Export the results to CSV for Tableau visualization:
    ```bash
    python export_results.py
    ```

## Conclusion
This project demonstrates how polynomial regression can capture nonlinear relationships between car features and fuel efficiency better than linear regression. The comparison of model performance highlights the strengths of each approach for predictive tasks.

