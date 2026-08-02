# Experiment 4: is the head dropout distorting the output scale?

**Hypothesis (H1)**: the Transformer's error is not noise, it is a systematic
compression of the output, and the `Dropout` in the dense head causes it.

Analysis of exp03's 241,504 saved test predictions found
`pred = 0.875 * actual + 0.336` at correlation 0.9935, a mean error of
-1.667 °C, and a predicted standard deviation of 9.92 against an actual 11.26.
46% of the test MSE (6.039) is pure squared bias, and undoing that affine
distortion by hand drops test MAE from 2.043 to 0.923. The LSTM (slope 0.837)
and the Transformer baseline (slope 0.856) show the same distortion, and all
three share the same dense head.

Re-scoring the saved exp03 model with dropout off (see
[`diagnose_output_scale.py`](../../../../../diagnose_output_scale.py) and
[`models/diagnostics/exp03_output_scale.json`](../../../../../models/diagnostics/exp03_output_scale.json))
gives MSE 6.011, bias -1.700 and slope 0.875 **on the training split** -- against
the 1.610 Keras reported during training. Keras reports training loss with dropout
on and validation loss with dropout off, so the usual train-vs-validation gap
cannot tell these two regimes apart. Measured on equal footing, the compression
is already there on the data the model was fitted to, so it is not a
generalisation failure.

The head is
`GlobalAveragePooling1D -> Dense(32, relu) -> Dropout(0.20) -> Dense(16, relu) -> Dense(1)`.
Dropout sits directly before a ReLU. Inverted dropout preserves the *mean*
entering `Dense(16)` but inflates its *variance* during training, and because
ReLU is convex, `E[relu(noisy)] > relu(E[clean])`. More signal reaches the
output during training than at inference, the output weights get calibrated to
the inflated regime, and at inference the prediction comes out multiplicatively
too small. On an unscaled target averaging 16.6 °C, a 12.5% shrink is a 1.67 °C
offset, which is what is observed.

**Prediction if H1 is true**: setting the head dropout to zero, changing nothing
else, removes the bias and brings the slope to roughly 1.0.

**What changed vs. exp03**: one value. `CONFIG` in [`config.py`](config.py) is
`TransformerConfig(head_dropout_rate=0.0)`. `head_dropout_rate` was added to
`build_model()` in [`architecture.py`](../../architecture.py) and to
`TransformerConfig` in [`training.py`](../../training.py); it controls only the
single `Dropout` in the dense head, and the four inside the encoder blocks keep
using `dropout_rate=0.20`. Left at its default of `None` it resolves to
`dropout_rate`, so every earlier experiment builds the identical graph. Feature
columns, `city_aware=True`, `city_embed_dim=8`, `d_model=64`, `num_heads=4`,
`num_encoder_layers=2`, `ff_dim=128`, `batch_size=32`, `epochs=25` and the
early-stopping patience of 5 are all exp03's.

**The control**: exp03 is rerun alongside this experiment rather than compared
against its July numbers. Both entrypoints now begin with
`tf.keras.utils.set_random_seed(42)`. Without a matched control, any
exp04-vs-exp03 difference would be confounded with run-to-run variation.

**Acceptance criteria**, judged against the exp03 control:

| | Threshold |
|---|---|
| Primary: bias | `abs(bias) < 0.5` °C |
| Primary: slope | `> 0.95` |
| Secondary: test MAE | below 1.5 (recalibrating exp03 by hand gives 0.92) |
| Guardrail: correlation | stays above 0.99 |

Both primary criteria must hold to accept H1. If accuracy improves but the
correlation drops, the model traded structure for calibration and the result
needs a closer look before any claim is made.

**How to run**

Locally (smoke test, CPU is fine for a quick sanity check):
```
python -m src.models.transformer.experiments.exp04_head_dropout.train
```

On Colab (full GPU training): open [`experiment.ipynb`](experiment.ipynb) in
Colab and run all cells -- see the notebook's first cell for one-time setup
(a GitHub PAT stored in Colab's Secrets manager). The notebook runs the exp03
control first, then exp04.

**Verdict**: [`check_exp04.py`](../../../../../check_exp04.py) in the repository
root reads `predictions.csv` from the newest exp03 and exp04 experiment
directories and prints n, MAE, RMSE, bias, slope, intercept and correlation for
each.

**Results**: saved under
`models/Transformer/exp04_head_dropout/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions.csv`, `training_history.csv`,
`model_summary.txt`, `figures/*.png`).
