import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(history, figure_dir, show=False):

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_mae_curve(history, figure_dir, show=False):

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Training vs Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_dir / "mae_curve.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_forecast_example(y_test, predictions, figure_dir, window_index=None, show=False):
    """Plots one test window's output_hours-long forecast against the true
    trajectory. y_test/predictions are (n, output_hours).

    If window_index is None, picks the window whose per-sample MAE is
    closest to the median (mirrors src/models/lstm/train.py's save_plots),
    so the example isn't a lucky or unlucky outlier. Returns the window
    index used."""
    per_sample_mae = np.mean(np.abs(predictions - y_test), axis=1)
    if window_index is None:
        median_mae = np.median(per_sample_mae)
        window_index = int(np.argmin(np.abs(per_sample_mae - median_mae)))

    truth = y_test[window_index]
    forecast = predictions[window_index]
    hours = np.arange(1, len(truth) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(hours, truth, marker="o", label="Actual Temperature")
    plt.plot(hours, forecast, marker="o", label="Predicted Temperature")
    plt.title(f"Forecast vs Actual (sample MAE {per_sample_mae[window_index]:.2f} °C)")
    plt.xlabel("Forecast Hour")
    plt.ylabel("Temperature (°C)")
    plt.xticks(hours)
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_dir / "forecast_example.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return window_index


def plot_mae_by_horizon(model_metrics, baseline_metrics, figure_dir, model_label="Model", show=False):
    """Plots MAE by forecast hour for the model against one or more
    baselines. `model_metrics`/each value in `baseline_metrics` is a
    calculate_metrics() dict; `baseline_metrics` maps a display label
    (e.g. "Persistence", "Previous day") to its metrics dict.

    This is the primary output figure of the retrain -- it's the one plot
    that makes the LSTM/Transformer comparison and the baseline ablation
    visible at a glance."""
    hours = np.arange(1, len(model_metrics["mae_by_hour"]) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(hours, model_metrics["mae_by_hour"], marker="o", label=model_label)
    for label, metrics in baseline_metrics.items():
        plt.plot(hours, metrics["mae_by_hour"], marker="o", linestyle="--", label=label)
    plt.title("MAE by Forecast Horizon")
    plt.xlabel("Forecast Hour")
    plt.ylabel("Mean Absolute Error (°C)")
    plt.xticks(hours)
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_dir / "mae_by_horizon.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
