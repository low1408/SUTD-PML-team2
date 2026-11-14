"""Hidden Markov model features: state probabilities that use only past and current hours."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Vital signs used as HMM observations: measured most hours at both hospitals.
VITALS = ("HR", "MAP", "Resp", "O2Sat")
# Readings outside these ranges are treated as recording errors and skipped (set to NaN).
PLAUSIBLE = {"HR": (20, 250), "MAP": (20, 200), "Resp": (4, 70), "O2Sat": (50, 100)}


def forward_filter(start: np.ndarray, transition: np.ndarray, emission: np.ndarray) -> np.ndarray:
    """Return P(state at hour t | observations up to hour t) for every hour.

    start:      shape (N,)   probability of each state at the first hour (pi)
    transition: shape (N, N) transition[i, j] = P(next state j | current state i) (A)
    emission:   shape (T, N) emission[t, i] = likelihood of hour t's observation in state i (b)
    """
    hours, states = emission.shape
    filtered = np.zeros((hours, states))
    predicted = start
    for t in range(hours):
        joint = predicted * emission[t]          # update with hour t's observation
        total = joint.sum()
        if total <= 0:
            raise ValueError(f"hour {t}: observation has zero likelihood in every state")
        filtered[t] = joint / total              # normalise so the row sums to 1
        predicted = filtered[t] @ transition     # predict the next hour
    return filtered


def log_gaussian_emissions(observations: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Return log emission[t, i] before any rescaling. Missing measurements add 0."""
    if np.any(stds <= 0):
        raise ValueError("standard deviations must be positive")
    x = observations[:, None, :]                       # (T, 1, D)
    z = (x - means[None, :, :]) / stds[None, :, :]     # (T, N, D)
    log_density = -0.5 * z**2 - np.log(stds)[None, :, :] - 0.5 * np.log(2 * np.pi)
    log_density = np.where(np.isnan(x), 0.0, log_density)   # skip missing measurements
    return log_density.sum(axis=2)                     # multiply = add the logs


def gaussian_emissions(observations: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Return emission[t, i]: how well hour t's measurements fit state i.

    observations: shape (T, D) measured values, NaN where not measured
    means:        shape (N, D) mean of each measurement in each state
    stds:         shape (N, D) standard deviation of each measurement in each state

    Each measurement uses a normal (Gaussian) density, and the measurements of
    one hour are multiplied together. A missing measurement is skipped, so an
    hour with nothing measured gives every state the same value. Each hour is
    rescaled so its largest value is 1; forward_filter normalises every hour,
    so this does not change its result but stops very small numbers rounding to 0.
    """
    log_emission = log_gaussian_emissions(observations, means, stds)
    log_emission -= log_emission.max(axis=1, keepdims=True)
    return np.exp(log_emission)


@dataclass
class GaussianHMM:
    """Fitted parameters. Use `filter` for features: it never looks ahead."""

    start: np.ndarray       # (N,)    pi
    transition: np.ndarray  # (N, N)  A
    means: np.ndarray       # (N, D)  mu
    stds: np.ndarray        # (N, D)  sigma

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """P(state at hour t | observations up to hour t), shape (T, N)."""
        emission = gaussian_emissions(observations, self.means, self.stds)
        return forward_filter(self.start, self.transition, emission)


def forward_backward(start: np.ndarray, transition: np.ndarray, log_emission: np.ndarray):
    """E-step for one stay. Uses the WHOLE stay, so it is for training only.

    Returns gamma (T, N), the summed xi (N, N), and the stay's log-likelihood.
    """
    hours, states = log_emission.shape
    shift = log_emission.max(axis=1, keepdims=True)
    emission = np.exp(log_emission - shift)

    # Forward pass, normalised every hour. scale[t] is the total before normalising.
    alpha = np.zeros((hours, states))
    scale = np.zeros(hours)
    predicted = start
    for t in range(hours):
        joint = predicted * emission[t]
        scale[t] = joint.sum()
        alpha[t] = joint / scale[t]
        predicted = alpha[t] @ transition

    # Backward pass, divided by the same totals so the numbers stay in range.
    beta = np.ones((hours, states))
    for t in range(hours - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    # xi summed over hours: alpha[t, i] * A[i, j] * b_j(x[t+1]) * beta[t+1, j] / scale[t+1]
    ahead = emission[1:] * beta[1:] / scale[1:, None]
    xi_sum = (alpha[:-1].T @ ahead) * transition
    log_likelihood = np.log(scale).sum() + shift.sum()
    return gamma, xi_sum, log_likelihood


def fit_hmm(stays: list[np.ndarray], n_states: int, n_iter: int = 30, tol: float = 1e-4,
            min_std_fraction: float = 0.01) -> tuple[GaussianHMM, list[float]]:
    """Baum-Welch on training stays only. Each stay is (hours, measurements), NaN = missing.

    Starts from quantiles of the data so the result is the same on every run.
    Stops when the total log-likelihood improves by less than `tol` per hour.
    Returns the fitted model and the log-likelihood after each iteration.
    """
    stays = [np.asarray(s, dtype=float) for s in stays if len(s) > 0]
    if not stays:
        raise ValueError("need at least one stay")
    pooled = np.vstack(stays)
    total_hours = len(pooled)
    overall_mean = np.nanmean(pooled, axis=0)
    overall_std = np.nanstd(pooled, axis=0)
    if np.any(np.isnan(overall_mean)) or np.any(overall_std <= 0):
        raise ValueError("every measurement needs at least two different observed values")
    min_std = min_std_fraction * overall_std

    # Initial guess: state 0 low on every measurement, the last state high.
    levels = (np.arange(n_states) + 0.5) / n_states
    means = np.nanquantile(pooled, levels, axis=0)
    stds = np.tile(overall_std, (n_states, 1))
    start = np.full(n_states, 1.0 / n_states)
    transition = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
    np.fill_diagonal(transition, 0.9 if n_states > 1 else 1.0)

    history = []
    for _ in range(n_iter):
        start_sum = np.zeros(n_states)
        xi_total = np.zeros((n_states, n_states))
        weight = np.zeros_like(means)            # sum of gamma where each measurement is observed
        weighted_x = np.zeros_like(means)
        weighted_x2 = np.zeros_like(means)
        log_likelihood = 0.0
        for stay in stays:                       # E-step, summed over every training stay
            gamma, xi_sum, ll = forward_backward(start, transition, log_gaussian_emissions(stay, means, stds))
            start_sum += gamma[0]
            xi_total += xi_sum
            observed = ~np.isnan(stay)
            values = np.where(observed, stay, 0.0)
            weight += gamma.T @ observed
            weighted_x += gamma.T @ values
            weighted_x2 += gamma.T @ values**2
            log_likelihood += ll
        history.append(log_likelihood)

        # M-step: weighted averages.
        start = start_sum / start_sum.sum()
        transition = xi_total / xi_total.sum(axis=1, keepdims=True)
        has_weight = weight > 0
        new_means = np.where(has_weight, weighted_x / np.where(has_weight, weight, 1), means)
        variance = np.where(has_weight, weighted_x2 / np.where(has_weight, weight, 1) - new_means**2, stds**2)
        means = new_means
        stds = np.maximum(np.sqrt(np.maximum(variance, 0)), min_std)

        if len(history) > 1 and (history[-1] - history[-2]) / total_hours < tol:
            break
    return GaussianHMM(start, transition, means, stds), history


def vital_observations(frame: pd.DataFrame, columns=VITALS) -> np.ndarray:
    """One stay's vitals as (hours, measurements), with implausible readings set to NaN."""
    values = frame[list(columns)].to_numpy(dtype=float).copy()
    for k, name in enumerate(columns):
        low, high = PLAUSIBLE.get(name, (-np.inf, np.inf))
        column = values[:, k]
        column[(column < low) | (column > high)] = np.nan
    return values


def hmm_state_features(frame: pd.DataFrame, model: GaussianHMM, columns=VITALS) -> pd.DataFrame:
    """hmm_state_0 ... hmm_state_{N-1} for every hour of ONE stay, forward-filtered only."""
    probabilities = model.filter(vital_observations(frame, columns))
    names = [f"hmm_state_{k}" for k in range(probabilities.shape[1])]
    return pd.DataFrame(probabilities, index=frame.index, columns=names)


def save_hmm(model: GaussianHMM, path: Path, columns=VITALS, extra: dict | None = None) -> None:
    """Save parameters as JSON so they can be read without pickle."""
    content = {key: value.tolist() for key, value in asdict(model).items()}
    content["columns"] = list(columns)
    content.update(extra or {})
    Path(path).write_text(json.dumps(content, indent=2))


def load_hmm(path: Path) -> tuple[GaussianHMM, list[str]]:
    content = json.loads(Path(path).read_text())
    model = GaussianHMM(*(np.array(content[key]) for key in ("start", "transition", "means", "stds")))
    return model, content["columns"]
