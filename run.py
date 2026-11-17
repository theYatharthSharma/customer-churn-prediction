"""One-command launcher.

    python run.py                 check data, train if needed, open the dashboard
    python run.py --retrain       force retraining before launching
    python run.py --train-only    train and save the model, do not launch the app
    python run.py --port 8502     use a different port
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer Churn Prediction System launcher")
    parser.add_argument("--retrain", action="store_true", help="retrain even if a model exists")
    parser.add_argument("--train-only", action="store_true", help="train and exit without starting the app")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (default 8501)")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print("Python 3.10+ is required (3.12 recommended).")
        return 1
    try:
        from src import data_loader, predict, train
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nRun:  pip install -r requirements.txt")
        return 1

    status, reason = predict.model_status()
    needs_training = args.retrain or args.train_only or status != "ok"

    if needs_training:
        if not data_loader.dataset_exists():
            print(config.DATASET_MISSING_MESSAGE)
            print(f"Expected path: {config.DATASET_PATH}")
            if args.train_only or args.retrain:
                return 1
            print("The app will still start, and you can upload a CSV from its sidebar.\n")
        else:
            try:
                df, notes = data_loader.load_dataset_with_notes()
            except data_loader.DatasetValidationError as exc:
                print(f"The dataset is not valid: {exc}")
                return 1
            for line in notes:
                print(f"note: {line}")
            print(f"Training models ({reason if not args.retrain else 'retrain requested'})...")
            train.train_all_models(df, progress_callback=train.print_progress)
            print("Model and reports saved to models/ and reports/.")
    else:
        print("A compatible trained model was found - skipping training.")

    if args.train_only:
        return 0
    print(f"Starting the dashboard on http://localhost:{args.port} (press Ctrl+C to stop)")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.port", str(args.port)])


if __name__ == "__main__":
    sys.exit(main())
