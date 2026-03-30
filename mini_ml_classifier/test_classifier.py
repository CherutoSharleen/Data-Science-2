import unittest
from pathlib import Path

import json

from classifier import evaluate, predict, predict_with_score, train


class MiniClassifierTests(unittest.TestCase):
    def test_prediction(self) -> None:
        model = train(Path(__file__).resolve().parent / "dataset.json")
        label = predict(model, "helpful support response")
        self.assertEqual(label, "positive")

    def test_predict_with_score_shape(self) -> None:
        model = train(Path(__file__).resolve().parent / "dataset.json")
        label, score = predict_with_score(model, "slow unresolved issue")
        self.assertIn(label, {"positive", "negative"})
        self.assertIsInstance(score, int)

    def test_evaluate_in_bounds(self) -> None:
        base = Path(__file__).resolve().parent
        model = train(base / "dataset.json")
        dataset = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
        acc = evaluate(model, dataset)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
