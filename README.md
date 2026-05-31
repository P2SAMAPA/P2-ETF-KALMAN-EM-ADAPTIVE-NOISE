# Adaptive Kalman Filter with EM (Shumway & Stoffer)

Implements a univariate Kalman filter with Expectation-Maximization for online estimation of process noise Q and measurement noise R. The model adapts to non-stationary regimes where noise levels shift over time. The filtered state estimate provides a denoised trend signal used for ETF ranking.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- State-space model: random walk latent state, direct observation
- EM algorithm (forward filtering, backward smoothing) to update Q, R
- Score = last filtered state estimate
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-kalman-em-adaptive-noise-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The Kalman filter extracts the underlying trend (latent state) from noisy returns.
- EM adapts Q (process noise) and R (measurement noise) to the current window, capturing changing volatility regimes.
- A high final filtered state indicates a strong upward trend after noise reduction.
- This method is critical for non-stationary markets where fixed noise levels fail.

## Requirements

See `requirements.txt`.
