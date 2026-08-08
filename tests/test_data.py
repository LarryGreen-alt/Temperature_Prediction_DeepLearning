"""Tests for src/models/common/data.py's windowing, the shared building
block both the LSTM and Transformer tracks are supposed to use identically.

Run with: python -m unittest tests.test_data
"""

import unittest

import numpy as np
import pandas as pd

from src.models.common.data import (
    FEATURE_COLUMNS,
    INPUT_HOURS,
    OUTPUT_HOURS,
    STRIDE,
    TARGET_COLUMN,
    CITY_VOCAB,
    create_multistep_sequences,
    split_into_contiguous_groups,
)

MINIMUM_ROWS = INPUT_HOURS + OUTPUT_HOURS  # 96


def make_hourly_dataframe(city, num_hours, start="2024-01-01", seed=0):
    """A synthetic, single-segment (no gaps) hourly frame for one city, with
    every FEATURE_COLUMNS filled with deterministic pseudo-random values."""
    rng = np.random.default_rng(seed)
    data = {column: rng.uniform(-10.0, 40.0, size=num_hours) for column in FEATURE_COLUMNS}
    data["time"] = pd.date_range(start=start, periods=num_hours, freq="h")
    data["city"] = city
    return pd.DataFrame(data)


class TestCreateMultistepSequences(unittest.TestCase):
    def test_shapes_and_target_alignment(self):
        city = "seattle"
        df = make_hourly_dataframe(city, num_hours=200)

        X, y = create_multistep_sequences(df)

        num_windows = len(range(0, len(df) - MINIMUM_ROWS + 1, STRIDE))
        self.assertEqual(X.shape, (num_windows, INPUT_HOURS, len(FEATURE_COLUMNS)))
        self.assertEqual(y.shape, (num_windows, OUTPUT_HOURS))

        # Window 0 starts at row 0, so its forecast covers rows
        # [INPUT_HOURS, INPUT_HOURS + OUTPUT_HOURS). Compared as float32,
        # matching the dtype create_multistep_sequences casts to internally.
        self.assertEqual(y[0, 0], np.float32(df[TARGET_COLUMN].iloc[INPUT_HOURS]))
        self.assertEqual(y[0, -1], np.float32(df[TARGET_COLUMN].iloc[INPUT_HOURS + OUTPUT_HOURS - 1]))

    def test_window_count_matches_formula(self):
        city = "seattle"
        for num_hours in (96, 100, 150, 200, 233):
            df = make_hourly_dataframe(city, num_hours=num_hours, seed=num_hours)
            X, y = create_multistep_sequences(df)
            expected = len(range(0, num_hours - MINIMUM_ROWS + 1, STRIDE))
            self.assertEqual(len(X), expected)
            self.assertEqual(len(y), expected)

    def test_below_minimum_rows_yields_no_windows(self):
        city = "seattle"
        df = make_hourly_dataframe(city, num_hours=MINIMUM_ROWS - 1)
        X, y = create_multistep_sequences(df)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)

    def test_with_city_ids_assigns_correct_vocab_id(self):
        city = "seattle"
        df = make_hourly_dataframe(city, num_hours=150)
        X, y, city_ids = create_multistep_sequences(df, with_city_ids=True)
        self.assertTrue(np.all(city_ids == CITY_VOCAB[city]))

    def test_matches_reference_loop_mirroring_lstm_train(self):
        """Independently reimplements the plain-Python window loop from
        src/models/lstm/train.py:344-349 and asserts byte-identical output
        against data.py's vectorized sliding_window_view implementation --
        the executable proof that the LSTM and Transformer tracks build
        identical windows from the same data."""
        city = "seattle"
        df = make_hourly_dataframe(city, num_hours=250, seed=42)

        features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        last_start = len(df) - MINIMUM_ROWS + 1

        reference_X = []
        reference_y = []
        for start_index in range(0, last_start, STRIDE):
            input_end = start_index + INPUT_HOURS
            target_end = input_end + OUTPUT_HOURS
            reference_X.append(features[start_index:input_end])
            reference_y.append(target[input_end:target_end])

        reference_X = np.asarray(reference_X, dtype=np.float32)
        reference_y = np.asarray(reference_y, dtype=np.float32)

        X, y = create_multistep_sequences(df)

        np.testing.assert_array_equal(X, reference_X)
        np.testing.assert_array_equal(y, reference_y)


class TestSplitIntoContiguousGroups(unittest.TestCase):
    def test_time_gap_splits_segments(self):
        city = "seattle"
        first = make_hourly_dataframe(city, num_hours=50, start="2024-01-01", seed=1)
        # A 5-hour gap between the two halves -- not adjacent hourly rows.
        gap_start = first["time"].iloc[-1] + pd.Timedelta(hours=5)
        second = make_hourly_dataframe(city, num_hours=50, start=gap_start, seed=2)
        df = pd.concat([first, second], ignore_index=True)

        segments = list(split_into_contiguous_groups(df))

        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0][1]), 50)
        self.assertEqual(len(segments[1][1]), 50)

        # Neither 50-hour half alone reaches the 96-row minimum, so no
        # window can be built -- proving windows never straddle the gap.
        X, y = create_multistep_sequences(df)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)

    def test_city_change_splits_segments(self):
        first = make_hourly_dataframe("seattle", num_hours=50, start="2024-01-01", seed=3)
        second = make_hourly_dataframe("boston", num_hours=50, start="2024-01-03", seed=4)
        df = pd.concat([first, second], ignore_index=True)

        segments = list(split_into_contiguous_groups(df))

        self.assertEqual({city for city, _ in segments}, {"seattle", "boston"})


if __name__ == "__main__":
    unittest.main()
