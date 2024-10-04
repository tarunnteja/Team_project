import pytest
from Train_linear import train_and_evaluate_RFmodel

def test_RFmodel_accuracy():
    # Call the function to get model evaluation metrics
    mse_rf, r2_rf, accuracy_rf = train_and_evaluate_RFmodel()

    # Print the accuracy and R² score for debugging purposes
    print(f"Model Accuracy: {accuracy_rf:.2f}")
    print(f"R² Score: {r2_rf:.2f}")

    # Assert that the R² score is above 0.5, a reasonable threshold for accuracy
    assert r2_rf > 0.5, f"R² score too low: {r2_rf}"

    # Assert that the accuracy is above the threshold
    assert accuracy_rf >= 0.75, f"Accuracy should be at least 0.75, but got {accuracy_rf:.2f}."
