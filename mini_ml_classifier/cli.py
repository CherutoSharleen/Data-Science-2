from __future__ import annotations

from pathlib import Path

from classifier import predict, train


def main() -> None:
    base = Path(__file__).resolve().parent
    model = train(base / "dataset.json")
    text = input("Enter message text: ").strip()
    print(predict(model, text))


if __name__ == "__main__":
    main()
