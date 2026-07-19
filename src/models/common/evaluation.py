import json

import pandas as pd


def evaluate_and_save(model, X_test, y_test, metrics_file, predictions_file):
    """Evaluate model on test set, save metrics.json and predictions.csv.
    Returns (loss, mae, rmse, predictions_flat)."""

    loss, mae, rmse = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Loss : {loss:.4f}")
    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")

    metrics = {"loss": float(loss), "mae": float(mae), "rmse": float(rmse)}

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics saved.")

    predictions = model.predict(X_test, verbose=0).flatten()

    prediction_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions,
        "Error": predictions - y_test
    })

    prediction_df.to_csv(predictions_file, index=False)

    return loss, mae, rmse, predictions
