"""Run with: python -m unittest discover -s . -p test_integrity.py -v"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import features as ft
import final_holdout as hold
import ml_strategy_backtest as bt
import ml_walkforward as wf


def candles(start='2025-12-20', count=400):
    x = np.arange(count)
    close = 100 + x * .02 + np.sin(x / 5)
    return pd.DataFrame({'timestamp': pd.date_range(start, periods=count, freq='h', tz='UTC'),
                         'open': close, 'close': close, 'high': close + 1,
                         'low': close - 1, 'volume': 100 + x % 17})


def dataset(start='2025-12-20', count=400):
    return ft.create_target(ft.build_features(candles(start, count)))


class IntegrityTests(unittest.TestCase):
    def test_unknown_tail_is_not_negative(self):
        df = dataset()
        self.assertTrue(df['target'].tail(24).isna().all())
        self.assertTrue(df['label_end_time'].tail(24).isna().all())
        self.assertTrue(df['target'].iloc[:-24].notna().all())

    def test_exact_24_hour_outcomes(self):
        df = dataset()
        known = df['target'].notna()
        self.assertTrue((df.loc[known, 'label_end_time'] - df.loc[known, 'timestamp']).eq(pd.Timedelta(hours=24)).all())
        np.testing.assert_allclose(df['future_close_24h'].iloc[:-24], df['close'].iloc[24:])

    def test_gap_duplicate_and_reversed_rejected(self):
        raw = candles()
        for invalid in (raw.drop(index=100), pd.concat([raw, raw.tail(1)]), raw.iloc[::-1]):
            with self.subTest(kind=len(invalid)), self.assertRaises(ValueError):
                ft.create_target(invalid)

    def test_features_are_causal(self):
        raw = candles()
        changed = raw.copy()
        changed.loc[250:, ['close', 'high', 'low', 'open', 'volume']] *= 10
        pd.testing.assert_frame_equal(ft.build_features(raw).iloc[:250], ft.build_features(changed).iloc[:250])

    def test_holdout_cutoff_strict_and_inference_tail_kept(self):
        df = dataset()
        train, test = hold.prepare_holdout(df)
        self.assertTrue((train.label_end_time < ft.DEVELOPMENT_END).all())
        self.assertEqual(train.timestamp.max(), ft.DEVELOPMENT_END - pd.Timedelta(hours=25))
        self.assertEqual(test.timestamp.max(), df.timestamp.max())
        self.assertTrue(test.target.tail(24).isna().all())

    def test_holdout_changes_cannot_change_training(self):
        raw = candles()
        first = hold.prepare_holdout(ft.create_target(ft.build_features(raw)))[0]
        raw.loc[raw.timestamp >= ft.DEVELOPMENT_END, ['close', 'high', 'low', 'open', 'volume']] *= 20
        second = hold.prepare_holdout(ft.create_target(ft.build_features(raw)))[0]
        pd.testing.assert_frame_equal(first, second)

    def test_legacy_and_invalid_provenance_rejected(self):
        df = dataset()
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            hold.prepare_holdout(df.drop(columns='label_end_time'))
        df.loc[100, 'label_end_time'] += pd.Timedelta(hours=1)
        with self.assertRaises(ValueError):
            hold.prepare_holdout(df)

    def test_direct_prediction_call_rejects_leakage_before_fit(self):
        df = dataset()
        train = df[(df.timestamp < ft.DEVELOPMENT_END)].dropna()
        test = df[df.timestamp >= ft.DEVELOPMENT_END]
        with self.assertRaisesRegex(ValueError, 'boundary'):
            hold.generate_predictions(train, test)

    def test_walkforward_actual_fit_rows_and_evaluation_horizon(self):
        df = dataset('2025-12-01', 800)
        seen = []
        class SpyModel:
            def fit(self, X, y):
                seen.append(df.loc[X.index])
            def predict_proba(self, X):
                return np.tile([.5, .5], (len(X), 1))
        windows = [('2025-12-01', '2025-12-15 23:00', '2025-12-16', '2026-01-02')]
        metrics = dict(roc_auc=.5, pr_auc=.5, precision=.5, recall=.5)
        with patch.object(wf, 'create_walkforward_windows', return_value=windows), patch.object(wf, 'calculate_metrics', return_value=metrics), contextlib.redirect_stdout(io.StringIO()):
            _, predicted = wf.run_walkforward(df, 'spy', SpyModel())
        self.assertEqual(len(seen), 1)
        self.assertTrue((seen[0].label_end_time < pd.Timestamp('2025-12-16', tz='UTC')).all())
        self.assertLess(seen[0].label_end_time.max(), predicted.timestamp.min())
        self.assertTrue((predicted.label_end_time < ft.DEVELOPMENT_END).all())

    def test_robustness_all_combinations_and_no_global_mutation(self):
        prices = candles('2025-01-01', 200)
        predictions = prices[['timestamp']].copy()
        predictions['probability'] = .9
        original = bt.TRADING_FEE, bt.SLIPPAGE
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = bt.robustness_test(predictions, prices)
                self.assertEqual(len(result), 15)
                self.assertEqual(len(result[['scenario', 'threshold']].drop_duplicates()), 15)
                self.assertEqual(len(list(Path('results').glob('ml_trades_*.csv'))), 5)
                for _, group in result.groupby('threshold'):
                    self.assertTrue(np.all(np.diff(group['final']) < 0))
            finally:
                os.chdir(cwd)
        self.assertEqual((bt.TRADING_FEE, bt.SLIPPAGE), original)

    def test_robustness_exception_preserves_costs(self):
        original = bt.TRADING_FEE, bt.SLIPPAGE
        with patch.object(bt, 'simulate_threshold', side_effect=RuntimeError('test')), patch.object(bt.Path, 'mkdir'):
            with self.assertRaises(RuntimeError):
                bt.robustness_test(None, None)
        self.assertEqual((bt.TRADING_FEE, bt.SLIPPAGE), original)

    def test_execution_cannot_use_2026_prices(self):
        prices = candles('2025-12-30', 100)
        predictions = prices[['timestamp']].copy()
        predictions['probability'] = .9
        _, trades = bt.simulate_threshold(predictions, prices, .75)
        self.assertGreater(len(trades), 0)
        self.assertTrue((trades.exit_time < ft.DEVELOPMENT_END).all())
        self.assertTrue((trades.entry_time == trades.signal_time + pd.Timedelta(hours=1)).all())
        self.assertTrue((trades.exit_time == trades.entry_time + pd.Timedelta(hours=24)).all())
        self.assertTrue((trades.entry_time.iloc[1:].reset_index(drop=True) >= trades.exit_time.iloc[:-1].reset_index(drop=True)).all())

    def test_holdout_execution_normalizes_naive_price_timestamps(self):
        prices = candles('2026-01-01', 80)
        predictions = prices[['timestamp']].copy()
        predictions['probability'] = .9
        prices['timestamp'] = prices.timestamp.dt.tz_localize(None)
        _, trades = hold.simulate(predictions, prices)
        self.assertGreater(len(trades), 0)
        self.assertTrue((trades.entry_time == trades.signal_time + pd.Timedelta(hours=1)).all())

    def test_frozen_configuration_and_feature_allowlist(self):
        self.assertEqual(hold.THRESHOLD, .75)
        self.assertEqual(hold.FEATURES, wf.FEATURES)
        self.assertFalse(set(hold.FEATURES) & {'target', 'label_end_time', 'future_return_24h', 'future_close_24h'})

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn unavailable')
    def test_real_model_fit_and_fixed_holdout_threshold(self):
        train, test = hold.prepare_holdout(dataset())
        prediction = hold.generate_predictions(train, test)
        self.assertTrue(prediction.probability.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
