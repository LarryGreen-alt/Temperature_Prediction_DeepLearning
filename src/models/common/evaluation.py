import json

import numpy as np
import pandas as pd

from src.models.common.data import CITY_VOCAB, FEATURE_COLUMNS, STRIDE


def calculate_metrics(y_true, y_pred):
    """y_true/y_pred are (n, output_hours). Returns overall_mae/rmse/bias,
    a handful of per-sample summary stats, and per-horizon mae_by_hour/
    rmse_by_hour/bias_by_hour lists.

    Uses the same key names as src.models.lstm.train.calculate_metrics, so
    a metrics.json from either track can be read by the same comparison
    code."""
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    per_sample_mae = np.mean(absolute_errors, axis=1)

    return {
        "overall_mae": float(np.mean(absolute_errors)),
        "overall_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "overall_bias": float(np.mean(errors)),
        "max_absolute_error": float(np.max(absolute_errors)),
        "median_sample_mae": float(np.median(per_sample_mae)),
        "p90_sample_mae": float(np.percentile(per_sample_mae, 90)),
        "p95_sample_mae": float(np.percentile(per_sample_mae, 95)),
        "p99_sample_mae": float(np.percentile(per_sample_mae, 99)),
        "endpoint_mae": float(np.mean(absolute_errors[:, -1])),
        "mae_by_hour": np.mean(absolute_errors, axis=0).astype(float).tolist(),
        "rmse_by_hour": np.sqrt(np.mean(np.square(errors), axis=0)).astype(float).tolist(),
        "bias_by_hour": np.mean(errors, axis=0).astype(float).tolist(),
    }


def calculate_baseline_predictions(temperature_history, output_hours):
    """temperature_history is (n, input_hours) of the target column's own
    past, aligned to the same windows as X/y. Callers build it with
    create_multistep_sequences(df, feature_columns=[TARGET_COLUMN])[0][:, :, 0],
    which is safe because segmentation and stride depend only on the frame
    and the window sizes, not on the feature set."""
    last_observed = temperature_history[:, -1]
    persistence = np.repeat(last_observed[:, np.newaxis], output_hours, axis=1)
    previous_day = temperature_history[:, -output_hours:]
    return persistence, previous_day


def save_predictions_sample(y_test, predictions, predictions_file, sample_size=300, seed=0):
    """Writes a subsampled, long-format predictions CSV (window, hour,
    actual, predicted) for spot-check plots. A full (n, output_hours)
    predictions table is not written -- for the full test set that's on
    the order of 5.8M values."""
    num_windows = len(y_test)
    sample_size = min(sample_size, num_windows)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(num_windows, size=sample_size, replace=False))

    output_hours = y_test.shape[1]
    sample_df = pd.DataFrame({
        "window": np.repeat(indices, output_hours),
        "hour": np.tile(np.arange(1, output_hours + 1), sample_size),
        "actual": y_test[indices].reshape(-1),
        "predicted": predictions[indices].reshape(-1),
    })
    sample_df.to_csv(predictions_file, index=False)


def calculate_group_metrics(y_test, predictions, city_ids):
    """Per-city breakdown, keyed by city name (data.CITY_VOCAB inverted).
    Each value is a full calculate_metrics() dict plus a samples count,
    matching Larry's test_metrics_by_group structure."""
    city_id_to_name = {city_index: city for city, city_index in CITY_VOCAB.items()}

    group_metrics = {}
    for city_id in sorted(np.unique(city_ids).tolist()):
        mask = city_ids == city_id
        city_metrics = calculate_metrics(y_test[mask], predictions[mask])
        city_metrics["samples"] = int(np.sum(mask))
        group_metrics[city_id_to_name[city_id]] = city_metrics

    return group_metrics


def evaluate_and_save(model, X_test, y_test, temperature_history, metrics_file, predictions_file,
                       model_input=None, sample_size=300, city_ids=None,
                       feature_columns=FEATURE_COLUMNS, stride=STRIDE):
    """Evaluates `model` on the test set, scores persistence and
    previous-day baselines on the same windows, and saves everything to
    `metrics_file` nested as improved_model/persistence_baseline/
    previous_day_baseline (Larry's schema), plus a subsampled
    predictions_sample.csv alongside it for plots.

    `model_input` is what gets passed to model.predict -- pass e.g.
    [X_test, city_ids] for a two-input city-aware model. Defaults to
    X_test itself.

    `temperature_history` is (n, input_hours) of the target column's own
    past (see calculate_baseline_predictions) -- required because the
    weather channel isn't guaranteed to be present in X_test (exp00/exp02
    drop it entirely).

    `city_ids` is an optional (n,) array of per-window city ids (see
    data.create_multistep_sequences(..., with_city_ids=True)). When given,
    a test_metrics_by_group breakdown is added and num_cities is filled in;
    otherwise both are omitted/null.

    Also writes a top-level provenance block (num_windows, num_cities,
    input_hours, output_hours, stride, feature_columns) describing how the
    test windows were built.

    Returns (metrics, predictions), where metrics is the full dict that was
    written to metrics_file."""
    assert len(temperature_history) == len(y_test), (
        f"temperature_history has {len(temperature_history)} rows, "
        f"expected {len(y_test)} to match y_test."
    )

    if model_input is None:
        model_input = X_test

    predictions = np.asarray(model.predict(model_input, verbose=0), dtype=np.float32)
    model_metrics = calculate_metrics(y_test, predictions)

    output_hours = y_test.shape[1]
    persistence_predictions, previous_day_predictions = calculate_baseline_predictions(
        temperature_history, output_hours
    )
    persistence_metrics = calculate_metrics(y_test, persistence_predictions)
    previous_day_metrics = calculate_metrics(y_test, previous_day_predictions)

    print(f"Test MAE  (model)        : {model_metrics['overall_mae']:.4f}")
    print(f"Test MAE  (persistence)  : {persistence_metrics['overall_mae']:.4f}")
    print(f"Test MAE  (previous day) : {previous_day_metrics['overall_mae']:.4f}")

    metrics = {"improved_model": model_metrics}
    if city_ids is not None:
        metrics["test_metrics_by_group"] = calculate_group_metrics(y_test, predictions, city_ids)
    metrics["persistence_baseline"] = persistence_metrics
    metrics["previous_day_baseline"] = previous_day_metrics
    metrics["num_windows"] = int(len(y_test))
    metrics["num_cities"] = int(len(np.unique(city_ids))) if city_ids is not None else None
    metrics["input_hours"] = int(X_test.shape[1])
    metrics["output_hours"] = int(output_hours)
    metrics["stride"] = int(stride)
    metrics["feature_columns"] = list(feature_columns)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    save_predictions_sample(y_test, predictions, predictions_file, sample_size)

    return metrics, predictions
