"""
models/lstm_model.py

Small PyTorch LSTM/GRU sequence regressor. Kept OPTIONAL: torch is a heavy
dependency, and it isn't installed in every environment (including the
sandbox this project was scaffolded in) — so this module only imports torch
inside `__init__`, not at module load time. Everything else in the project
(walk-forward harness, other models) works fine without torch installed;
this class simply raises a clear ImportError if you try to use it without
torch present.

Not wired into the default Phase 5 walk-forward run — refitting an LSTM
every single day of an expanding-window loop is slow. Use it standalone,
or thread it into backtest/walk_forward.py's model set once you're ready to
pay that training cost (e.g. by refitting every N days instead of daily).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LSTMModel:
    def __init__(
        self,
        feature_cols: list[str],
        sequence_length: int = 10,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 30,
        lr: float = 1e-3,
        cell_type: str = "lstm",  # "lstm" or "gru"
    ):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LSTMModel requires PyTorch ('pip install torch'), which is "
                "not installed in this environment. Every other part of the "
                "pipeline (ARIMA, XGBoost, walk-forward harness) works "
                "without it."
            ) from e

        self.feature_cols = list(feature_cols)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.cell_type = cell_type
        self._net = None
        self._mean = None
        self._std = None

    def _build_net(self, n_features: int):
        import torch.nn as nn

        cell_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU

        class _SeqNet(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, cell_cls):
                super().__init__()
                self.rnn = cell_cls(
                    input_size=n_features, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _SeqNet(n_features, self.hidden_size, self.num_layers, cell_cls)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        n = len(X)
        seqs = []
        targets = []
        for i in range(self.sequence_length, n + 1):
            seqs.append(X[i - self.sequence_length:i])
            if y is not None:
                targets.append(y[i - 1])
        return np.stack(seqs), (np.array(targets) if y is not None else None)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        import torch
        import torch.nn as nn

        X_arr = X[self.feature_cols].to_numpy(dtype=np.float32)
        y_arr = y.to_numpy(dtype=np.float32)

        self._mean = X_arr.mean(axis=0)
        self._std = X_arr.std(axis=0) + 1e-8
        X_scaled = (X_arr - self._mean) / self._std

        if len(X_scaled) <= self.sequence_length:
            raise ValueError(
                f"LSTMModel: need more than sequence_length={self.sequence_length} "
                f"training rows, got {len(X_scaled)}."
            )

        seqs, targets = self._make_sequences(X_scaled, y_arr)
        seqs_t = torch.tensor(seqs, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        self._net = self._build_net(n_features=X_scaled.shape[1])
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self._net(seqs_t)
            loss = loss_fn(pred, targets_t)
            loss.backward()
            optimizer.step()

        # Keep the last `sequence_length` rows of TRAINING data so predict()
        # can build a sequence ending at the new test row(s) without needing
        # the caller to pass history explicitly.
        self._last_train_window = X_scaled[-self.sequence_length:]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("LSTMModel.predict called before fit().")

        X_arr = X[self.feature_cols].to_numpy(dtype=np.float32)
        X_scaled = (X_arr - self._mean) / self._std

        preds = []
        history = self._last_train_window.copy()
        self._net.eval()
        with torch.no_grad():
            for row in X_scaled:
                window = np.vstack([history[1:], row[None, :]])
                seq_t = torch.tensor(window[None, :, :], dtype=torch.float32)
                pred = self._net(seq_t).item()
                preds.append(pred)
                history = window
        return np.asarray(preds)
