
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


def train_and_evaluate_RFmodel():
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

    df = df_loaded

    l = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    df[l] = df[l].apply(le.fit_transform)

    #Outliers using Standard Deviation
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Split data into features (X) and target variable (y)
    X = df_loaded.drop('Attrition', axis=1)
    y = df_loaded['Attrition']
    

    # Split into training and test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train the random forest regression
    model0 = RandomForestRegressor(n_estimators=300, random_state=42)
    model0.fit(X_train, y_train)

    # Make predictions
    y_pred_reg = model0.predict(X_test)
            
    # Evaluate the model
    mse_rf = mean_squared_error(y_test, y_pred_reg)
    r2_rf = r2_score(y_test, y_pred_reg)
    
    # Train the random forest classification 

    model1 = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)
    model1.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_class = model1.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_class)

    print(accuracy_rf)

    return mse_rf, r2_rf, accuracy_rf


    

if __name__ == "__main__":
    mse_rf, r2_rf,accuracy_rf = train_and_evaluate_RFmodel()
    print(f" random forest regression - R^2 Score: {r2_rf}")
    print(f" random forest regression - Mean Squared Error: {mse_rf}")
    print(f" random forest clasification - Accuracy_rf: {accuracy_rf * 100:.2f}%")
  
