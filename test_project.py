import pytest
from finalproject_v1_2_labelencoding import train_and_evaluate_model

def test_linear_model_accuracy():
   
    mse, r2 = train_and_evaluate_model()

    # Assert that the R² score is above 0.5, a reasonable threshold for accuracy
    assert r2 > 0.5, f"R² score too low: {r2}"

    print(f"Test passed with R² score: {r2}, MSE: {mse}")

