import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── ManualMLP (same as notebook) ─────────────────────────────────────────────
class ManualMLP:
    def __init__(self, layer_sizes, learning_rate=0.001,
                 max_iter=500, batch_size=256, random_state=42):
        np.random.seed(random_state)
        self.layer_sizes  = layer_sizes
        self.lr           = learning_rate
        self.max_iter     = max_iter
        self.batch_size   = batch_size
        self.loss_history = []
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) \
                * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):      return np.maximum(0, z)
    def _relu_grad(self, z): return (z > 0).astype(float)

    def _forward(self, X):
        self._activations = [X]
        self._z_vals = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self._activations[-1] @ w + b
            self._z_vals.append(z)
            a = self._relu(z) if i < len(self.weights) - 1 else z
            self._activations.append(a)
        return self._activations[-1]

    def _backward(self, y_true):
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)
        delta = 2 * (self._activations[-1] - y_true) / m
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.weights)
        for i in reversed(range(len(self.weights))):
            grad_w[i] = self._activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_grad(self._z_vals[i-1])
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i]  -= self.lr * grad_b[i]

    def fit(self, X, y):
        for epoch in range(self.max_iter):
            perm = np.random.permutation(X.shape[0])
            X_s, y_s = X[perm], y[perm]
            for start in range(0, X.shape[0], self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]
                self._forward(Xb)
                self._backward(yb)
            loss = np.mean((self._forward(X).flatten() - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        return self._forward(X).flatten()


# ── Load pre-trained weights (instant startup, no training needed) ────────────
@st.cache_resource
def load_and_train():
    ck = np.load("model_weights.npz")

    model = ManualMLP(layer_sizes=[5, 64, 32, 1])
    model.weights = [ck["w0"], ck["w1"], ck["w2"]]
    model.biases  = [ck["b0"], ck["b1"], ck["b2"]]

    X_mean = ck["X_mean"]
    X_std  = ck["X_std"]
    y_all  = ck["y_all"]
    y_test = ck["y_test"]
    y_pred = ck["y_pred"]
    rmse, mae, r2 = ck["metrics"]

    return model, X_mean, X_std, y_all, y_test, y_pred, float(rmse), float(mae), float(r2)


# ── App config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

with st.spinner("Training model... (only happens once)"):
    model, X_mean, X_std, y_all, y_test, y_pred, rmse, mae, r2 = load_and_train()

page = st.sidebar.radio("Navigate", ["Landing Page", "Interactive Demo"])


# ── Page 1: Landing Page ──────────────────────────────────────────────────────
if page == "Landing Page":
    st.title("Predicting Spotify Track Popularity")
    st.subheader("DS4420 Final Project")
    st.markdown("---")

    st.markdown("""
    ## Project Overview
    Can audio features of a song predict how popular it becomes on Spotify?
    This project applies two machine learning methods to 114,000 Spotify tracks
    to explore the relationship between sound and commercial success.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Dataset
        - **Source**: Spotify Tracks Dataset
        - **Size**: 114,000 songs
        - **Target variable**: `popularity` (0–100)
        - **Features used**:
            - Danceability
            - Energy
            - Tempo
            - Loudness
            - Valence
        """)

    with col2:
        st.markdown("### Neural Network Performance")
        st.dataframe(pd.DataFrame({
            "Metric": ["RMSE", "MAE", "R²"],
            "Value":  [f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
        }), hide_index=True)

    st.markdown("---")

    st.markdown("""
    ### Models
    | Model | Language | Method |
    |-------|----------|--------|
    | Multilayer Perceptron (MLP) | Python | Implemented from scratch with NumPy |
    | Bayesian Regression | R | Posterior inference with credible intervals |

    ### Key Finding
    Audio features explain approximately **7% of variance** in popularity (R² = 0.071).
    Popularity is also driven by factors outside audio content — artist reputation,
    playlist placement, and social media — which this dataset does not capture.
    """)


# ── Page 2: Interactive Demo ──────────────────────────────────────────────────
elif page == "Interactive Demo":
    st.title("Interactive Popularity Predictor")
    st.markdown("Adjust the audio features below and see what popularity score our MLP predicts.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Set Audio Features")
        danceability = st.slider("Danceability",    0.0,   1.0,  0.57, 0.01)
        energy       = st.slider("Energy",          0.0,   1.0,  0.64, 0.01)
        tempo        = st.slider("Tempo (BPM)",    60.0, 220.0, 122.0, 1.0)
        loudness     = st.slider("Loudness (dB)", -50.0,   5.0,  -8.3, 0.1)
        valence      = st.slider("Valence",         0.0,   1.0,  0.47, 0.01)

        x_input  = np.array([[danceability, energy, tempo, loudness, valence]])
        x_scaled = (x_input - X_mean) / X_std
        pred     = float(np.clip(model.predict(x_scaled)[0], 0, 100))

        st.markdown("---")
        st.metric("Predicted Popularity", f"{pred:.1f} / 100")

    with col2:
        st.subheader("Where Does Your Track Fall?")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Distribution with user's prediction marked
        axes[0].hist(y_all, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].axvline(pred, color="red", linewidth=2, linestyle="--",
                        label=f"Your track: {pred:.1f}")
        axes[0].set_xlabel("Popularity")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Popularity Distribution")
        axes[0].legend()

        # Prediction vs Actual (test set)
        axes[1].scatter(y_test, y_pred, alpha=0.2, s=8, color="steelblue")
        axes[1].plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Ideal")
        axes[1].scatter([pred], [pred], color="red", s=80, zorder=5, label="Your track")
        axes[1].set_xlabel("Actual Popularity")
        axes[1].set_ylabel("Predicted Popularity")
        axes[1].set_title("Prediction vs Actual (Test Set)")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
