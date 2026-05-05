import unittest
from pathlib import Path

import json

from classifier import confusion_matrix, evaluate, label_counts, predict, predict_with_score, train


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

    def test_label_counts(self) -> None:
        base = Path(__file__).resolve().parent
        dataset = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
        counts = label_counts(dataset)
        self.assertIn("positive", counts)
        self.assertIn("negative", counts)

    def test_confusion_matrix(self) -> None:
        matrix = confusion_matrix(["positive", "negative"], ["positive", "positive"])
        self.assertEqual(matrix["positive"]["positive"], 1)
        self.assertEqual(matrix["negative"]["positive"], 1)


if __name__ == "__main__":
    unittest.main()
