"""Model training module for loan approval."""
import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .evaluate import evaluate_model, print_metrics

logger = logging.getLogger(__name__)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 150) -> RandomForestClassifier:
    """Train Random Forest classifier.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees in forest.
        
    Returns:
        Trained RandomForestClassifier model.
    """
    logger.info('Training model with %d estimators', n_estimators)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info('Model training complete')
    return model


def save_model(model: RandomForestClassifier, output_path: Path) -> None:
    """Save trained model to disk.
    
    Args:
        model: Trained model.
        output_path: Path to save model file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info('Saved model to %s', output_path)


def main() -> None:
    """Main entry point for training module."""
    parser = argparse.ArgumentParser(description='Train loan approval model')
    parser.add_argument('--data', type=Path, required=True, help='Path to processed dataset CSV')
    parser.add_argument('--model-output', type=Path, required=True, help='Path to store model artifact')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    try:
        df = pd.read_csv(args.data)
        X = df.drop(columns=['loan_status'])
        y = df['loan_status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

        model = train_model(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        print_metrics(metrics)
        save_model(model, args.model_output)
        logger.info('Training pipeline complete')
    except Exception as e:
        logger.error('Training failed: %s', e)
        raise


if __name__ == '__main__':
    main()
