import unittest
from pathlib import Path

from classifier import predict, train


class MiniClassifierTests(unittest.TestCase):
    def test_prediction(self) -> None:
        model = train(Path(__file__).resolve().parent / "dataset.json")
        label = predict(model, "helpful support response")
        self.assertEqual(label, "positive")


if __name__ == "__main__":
    unittest.main()
