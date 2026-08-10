import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import detrend


def generate_series(n_points, trend_slope=0.0, seasonal_amplitude=1.0, seasonal_period=50,noise_std=0.1, phase=0.0, start_level=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_points)
    trend = start_level + trend_slope * t
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * (t + phase) / seasonal_period)
    noise = rng.normal(0.0, noise_std, size=n_points)

    return t, trend + seasonality + noise


def generate_series_ar(n_points, phi=0.0, trend_slope=0.0, seasonal_amplitude=1.0, seasonal_period=50, noise_std=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    e = rng.normal(0.0, noise_std, size=n_points)
    u = np.zeros(n_points)
    for k in range(1, n_points):
        u[k] = phi * u[k - 1] + e[k]

    t = np.arange(n_points)
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    return t, trend_slope * t + seasonality + u


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def estimate_period(series, pmax=400):
    detrended = detrend(series, order=1)
    a = acf(detrended, nlags=pmax)
    peaks, _ = find_peaks(a, height=0.1)
    return int(peaks[0])
