# DataAnalysis-Cleaning-and-LinearRegression
E-commerce Expense Analysis

# Data Loading:
Loads a dataset from a CSV file named 'Ecom Expense.csv' located in a specific directory.
# Data Analysis:
Displays the first 3 rows of the dataset using head().
Checks the shape of the dataset using shape.
Displays column names using columns.values.
Checks data types of columns using dtypes.
Identifies missing values for each column.
# Data Cleaning:
Creates dummy variables for two categorical columns: 'Gender' and 'City Tier'.
Removes the original categorical variables from the dataset.
Normalizes the dataset using a custom normalization function.
Displays the first two records of the cleaned and normalized dataset.
Generates histograms for all variables.
Creates scatter plots to visualize the relationships between 'Age', 'Monthly Income', 'Transaction Time', and 'Total Spend'.
# Linear Regression:
Builds a linear regression model with two variations.
* Variation 1:
Defines predictor columns ('Monthly Income', 'Transaction Time', 'Gender_Female', 'Gender_Male', 'City Tier_Tier 1', 'City Tier_Tier 2', 'City Tier_Tier 3').
Splits the dataset into training and testing sets using 'train_test_split'.
Sets a random seed for reproducibility.
Fits a linear regression model using the training data.
Displays the model coefficients.
Calculates and displays the model score (R-squared) on the training data.
* Variation 2:
Adds the 'Record' feature to the predictor columns.
Splits the dataset again and fits a new linear regression model.
Displays the model coefficients and the model score.
# Model Evaluation:
Predicts 'Total Spend' for the test data using both models.
Compares the predictions with the actual 'Total Spend' values in the test data.

