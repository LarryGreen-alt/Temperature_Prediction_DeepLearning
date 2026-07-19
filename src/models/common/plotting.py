import matplotlib.pyplot as plt


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


def plot_prediction_curve(y_test, predictions, figure_dir, n=200, show=False):

    plt.figure(figsize=(14, 6))
    plt.plot(y_test[:n], label="Actual Temperature")
    plt.plot(predictions[:n], label="Predicted Temperature")
    plt.title("Predicted vs Actual Temperatures")
    plt.xlabel("Sample")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_dir / "prediction_curve.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
