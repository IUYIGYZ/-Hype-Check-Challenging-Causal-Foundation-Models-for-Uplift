import unittest

import numpy as np

try:
    from baseline_benchmark.neural import DragonNetEstimator
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"PyTorch is unavailable in this environment: {exc}") from exc


def _neural_data(seed=7, n=160, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    treatment = np.tile([0, 1], n // 2).astype(np.int8)
    probability = np.clip(0.2 + 0.1 * treatment + 0.05 * X[:, 0], 0.01, 0.99)
    outcome = rng.binomial(1, probability).astype(np.int8)
    return X, treatment, outcome


class DragonNetEstimatorTests(unittest.TestCase):
    def test_rejects_invalid_hyperparameters(self):
        invalid = [
            ("epochs", 0),
            ("batch_size", 0),
            ("hidden_dim", 1),
            ("patience", 0),
            ("learning_rate", 0),
            ("weight_decay", -1),
            ("seed", 1.5),
        ]
        for parameter, value in invalid:
            with self.subTest(parameter=parameter, value=value):
                with self.assertRaises(ValueError):
                    DragonNetEstimator(**{parameter: value})

    def test_fit_predict_and_input_checks(self):
        X, treatment, outcome = _neural_data()
        model = DragonNetEstimator(
            seed=7,
            epochs=2,
            batch_size=32,
            hidden_dim=16,
            patience=2,
            device="cpu",
        )
        model.fit(
            X[:120],
            treatment[:120],
            outcome[:120],
            X[120:],
            treatment[120:],
            outcome[120:],
        )
        cate = model.predict_cate(X[120:])
        self.assertEqual(cate.shape, (40,))
        self.assertTrue(np.isfinite(cate).all())
        self.assertTrue(np.all((-1.0 <= cate) & (cate <= 1.0)))

        with self.assertRaisesRegex(ValueError, "features"):
            model.predict_cate(np.ones((5, 3), dtype=np.float32))

    def test_rejects_invalid_training_labels(self):
        X, _, outcome = _neural_data()
        model = DragonNetEstimator(epochs=1, batch_size=32, hidden_dim=8, device="cpu")
        with self.assertRaisesRegex(ValueError, "both 0 and 1"):
            model.fit(
                X[:120],
                np.zeros(120, dtype=np.int8),
                outcome[:120],
                X[120:],
                np.zeros(40, dtype=np.int8),
                outcome[120:],
            )


if __name__ == "__main__":
    unittest.main()
