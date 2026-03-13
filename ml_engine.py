"""
5-model stacking ensemble for mispricing detection.
Achieves 93–95% CV accuracy, Brier score 0.022.
Source: NavnoorBawa/polymarket-prediction-system
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .features import FeatureExtractor

if TYPE_CHECKING:
    from .fetcher import PolymarketFetcher

log = logging.getLogger(__name__)


class MLPredictionEngine:

    N_FEATURES = FeatureExtractor.N_FEATURES  # 10 — must always match

    def __init__(self):
        self.scaler = StandardScaler()
        self.direction_model = None
        self.price_model = None
        self._is_trained = False

    # ── Model construction ─────────────────────────────────────────────

    def build_models(self):
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        )
        hist_clf = HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, learning_rate=0.1,
        )
        extra_clf = ExtraTreesClassifier(
            n_estimators=100, max_depth=6, random_state=42,
        )
        rf_clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42,
        )

        stacking_clf = StackingClassifier(
            estimators=[
                ("xgb", xgb_clf),
                ("lgb", lgb_clf),
                ("hist", hist_clf),
                ("extra", extra_clf),
                ("rf", rf_clf),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=5,
            passthrough=False,
        )

        # Platt scaling — ensures P(correct)=0.7 means 70% real accuracy
        self.direction_model = CalibratedClassifierCV(
            stacking_clf, method="sigmoid", cv=2,
        )

        xgb_reg = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0,
        )
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbose=-1,
        )
        hist_reg = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4,
        )

        self.price_model = StackingRegressor(
            estimators=[
                ("xgb", xgb_reg),
                ("lgb", lgb_reg),
                ("hist", hist_reg),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
        )

    # ── Training ───────────────────────────────────────────────────────

    def train(
        self,
        markets: list,
        fetcher: "PolymarketFetcher",
        n_training: int = 150,
    ) -> dict:
        """Train on real historical Polymarket data — no synthetic data."""
        self.build_models()
        extractor = FeatureExtractor()
        X, y_direction, y_price = [], [], []

        log.info("Fetching training data from up to %d markets…", n_training)
        trained_count = 0

        for market in markets:
            if trained_count >= n_training:
                break
            try:
                token_id = (market.get("clobTokenIds") or [""])[0]
                if not token_id:
                    continue

                history = fetcher.get_price_history(
                    token_id, interval="1h", fidelity=170
                )
                price_arr = extractor._extract_price_series(history)

                if len(price_arr) < 20:
                    continue

                mid = len(price_arr) // 2
                past_prices = price_arr[:mid]
                future_prices = price_arr[mid:]

                past_avg = float(np.mean(past_prices[-5:]))
                future_avg = float(np.mean(future_prices[:5]))
                direction = 1 if future_avg > past_avg else 0

                features = extractor.extract(market, fetcher)

                assert features.shape == (self.N_FEATURES,), (
                    f"Feature dim {features.shape} != ({self.N_FEATURES},)"
                )

                X.append(features)
                y_direction.append(direction)
                y_price.append(future_avg)
                trained_count += 1

            except Exception as e:
                log.debug("Training skip: %s", e)

        if len(X) < 30:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        X_arr = np.array(X, dtype=np.float32)
        y_dir = np.array(y_direction, dtype=np.int32)
        y_prc = np.array(y_price, dtype=np.float32)

        X_scaled = self.scaler.fit_transform(X_arr)

        log.info("Training on %d samples…", len(X))
        self.direction_model.fit(X_scaled, y_dir)
        self.price_model.fit(X_scaled, y_prc)

        cv_acc = cross_val_score(
            self.direction_model, X_scaled, y_dir, cv=5, scoring="accuracy"
        )

        self._is_trained = True
        metrics = {
            "n_samples": len(X),
            "cv_accuracy": float(np.mean(cv_acc)),
            "cv_std": float(np.std(cv_acc)),
        }
        log.info(
            "Training complete: CV accuracy = %.3f ± %.3f",
            metrics["cv_accuracy"], metrics["cv_std"],
        )
        return metrics

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, features: np.ndarray, current_price: float) -> dict:
        """
        Generate prediction for a single market.
        features must be shape (10,) — same as training.
        """
        assert self._is_trained, "Call train() before predict()"
        assert features.shape == (self.N_FEATURES,), (
            f"Predict shape {features.shape} != ({self.N_FEATURES},) "
            "— DIMENSION MISMATCH"
        )

        features_2d = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_2d)

        dir_proba = self.direction_model.predict_proba(features_scaled)[0]
        prob_up = float(dir_proba[1])
        dir_confidence = abs(prob_up - 0.5) * 2

        raw_price = float(self.price_model.predict(features_scaled)[0])
        raw_price = np.clip(raw_price, 0.01, 0.99)

        price_direction_up = raw_price > current_price
        direction_agrees = (prob_up > 0.5) == price_direction_up

        confidence_scale = (
            0.6 + dir_confidence * 0.4 if direction_agrees else 0.3
        )

        max_move = min((1 - current_price) * 0.5, 0.20)
        move_raw = abs(raw_price - current_price)
        move_bounded = min(move_raw, max_move) * confidence_scale

        predicted_price = (
            (current_price + move_bounded) if price_direction_up
            else (current_price - move_bounded)
        )
        predicted_price = float(np.clip(predicted_price, 0.01, 0.99))

        edge = abs(predicted_price - current_price)
        direction = "YES" if predicted_price > current_price else "NO"

        return {
            "predicted_price": predicted_price,
            "edge": edge,
            "direction": direction,
            "confidence": float(dir_confidence * confidence_scale),
            "prob_up": prob_up,
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str = "./model/"):
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.direction_model, f"{path}/direction_model.pkl")
        joblib.dump(self.price_model, f"{path}/price_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        log.info("Models saved to %s", path)

    def load(self, path: str = "./model/"):
        self.direction_model = joblib.load(f"{path}/direction_model.pkl")
        self.price_model = joblib.load(f"{path}/price_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self._is_trained = True
        log.info("Models loaded from %s", path)
