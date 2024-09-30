import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

def train_and_evaluate_model():
    # Load dataset
    df_loaded = pd.read_csv('hr_employee.csv')

    # Handle missing values
    numerical_cols = df_loaded.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_loaded.select_dtypes(include=['object']).columns
    
    # Replace null values in numerical columns with mean
    for col in numerical_cols:
        mean_value = df_loaded[col].mean()
        df_loaded[col].fillna(mean_value, inplace=True)

    # Replace null values in categorical columns with mode
    for col in categorical_cols:
        mode_value = df_loaded[col].mode()[0]
        df_loaded[col].fillna(mode_value, inplace=True)

    # Label encoding for categorical variables
    le = preprocessing.LabelEncoder()
    label_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    df_loaded[label_cols] = df_loaded[label_cols].apply(le.fit_transform)

    # Split data into features (X) and target variable (y)
    X = df_loaded.drop('Attrition', axis=1)
    y = df_loaded['Attrition']

    # Split into training and test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    # Make predictions
    y_pred_lin = lin_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_lin)
    r2 = r2_score(y_test, y_pred_lin)

    # Return evaluation metrics
    return mse, r2

if __name__ == "__main__":
    mse, r2 = train_and_evaluate_model()
    print(f"Linear Regression - Mean Squared Error: {mse}")
    print(f"Linear Regression - R^2 Score: {r2}")
