import numpy as np

class AdaptiveKalmanEM:
    """
    Univariate Kalman filter with EM for adaptive noise covariances Q and R.
    State-space model:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = x_t + v_t,       v_t ~ N(0, R)
    """
    def __init__(self, Q_init=1.0, R_init=1.0, em_iter=10):
        self.Q = Q_init
        self.R = R_init
        self.em_iter = em_iter

    def fit(self, observations):
        """
        Run Kalman filter + EM to estimate Q, R.
        Returns filtered state estimates (x_t|t).
        """
        T = len(observations)
        if T < 3:
            return np.zeros(T)
        # Initial state
        x_pred = observations[0]
        P_pred = 1.0
        # Storage
        x_filt = np.zeros(T)
        P_filt = np.zeros(T)
        # EM loop
        for em in range(self.em_iter):
            # Forward pass (Kalman filter)
            x_pred = observations[0]
            P_pred = 1.0
            for t in range(T):
                # Update
                K = P_pred / (P_pred + self.R)
                x_filt[t] = x_pred + K * (observations[t] - x_pred)
                P_filt[t] = (1 - K) * P_pred
                # Predict next
                if t < T-1:
                    x_pred = x_filt[t]
                    P_pred = P_filt[t] + self.Q
            # Backward smoothing (Rauch-Tung-Striebel) to get smoothed states
            x_smooth = np.zeros(T)
            P_smooth = np.zeros(T)
            x_smooth[-1] = x_filt[-1]
            P_smooth[-1] = P_filt[-1]
            for t in range(T-2, -1, -1):
                C = P_filt[t] / (P_filt[t] + self.Q)
                x_smooth[t] = x_filt[t] + C * (x_smooth[t+1] - x_filt[t])
                P_smooth[t] = P_filt[t] + C**2 * (P_smooth[t+1] - (P_filt[t] + self.Q))
            # EM update of Q and R
            # R = mean of (y_t - x_smooth[t])^2
            R_new = np.mean((observations - x_smooth)**2)
            # Q = mean of (x_smooth[t+1] - x_smooth[t])^2
            diff = np.diff(x_smooth)
            Q_new = np.mean(diff**2)
            self.Q = max(Q_new, 1e-6)
            self.R = max(R_new, 1e-6)
        return x_filt

def kalman_em_score(returns, em_iter=10):
    """
    Compute per-ETF score = final filtered state estimate (last value).
    """
    returns_clean = returns.dropna().values
    if len(returns_clean) < 3:
        return 0.0
    # Initial Q,R from data variance
    Q_init = np.var(returns_clean) * 0.01
    R_init = np.var(returns_clean) * 0.5
    kf = AdaptiveKalmanEM(Q_init, R_init, em_iter)
    filtered = kf.fit(returns_clean)
    return float(filtered[-1])   # last filtered state as score
