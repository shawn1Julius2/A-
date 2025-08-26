"""Compute dragon dance chain positions and speeds along an Archimedean spiral.

This module implements the rigid chain model described in the paper. It
simulates the dragon with 223 benches moving on a spiral at a constant
head speed and outputs the position and velocity tables required for
``result1.xlsx``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Basic parameters of the dragon chain and spiral
# ---------------------------------------------------------------------------
N = 223  # total number of benches (1 head, 221 body, 1 tail)

# distance between the head bench front handle and the next bench
D_head = 2.860  # metres

# distance between subsequent body benches
D_body = 1.650  # metres

# Archimedean spiral pitch (m)
p = 0.55

# spiral coefficient a = p / (2*pi)
a = p / (2 * np.pi)

# head bench translation speed along the spiral (m/s)
v_head = 1.0

# simulation time step (s)
DT = 1.0

# total simulation duration (s)
T_total = 300

# time sequence used throughout the simulation
times = np.arange(0, T_total + DT, DT)

# expose upper-case aliases for backward compatibility
D_HEAD = D_head
D_BODY = D_body
PITCH = p
A = a
V_HEAD = v_head
T_TOTAL = T_total


def spiral_length(theta: float) -> float:
    """Return arc length of ``r = a * theta`` from 0 to ``theta``."""
    return 0.5 * A * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))


def invert_length(s_target: float, theta_guess: float) -> float:
    """Invert :func:`spiral_length` using Newton iteration."""
    theta = theta_guess
    for _ in range(20):
        f = spiral_length(theta) - s_target
        if abs(f) < 1e-10:
            break
        theta -= f / (A * np.sqrt(theta ** 2 + 1))
    return theta


# initial angle of the head (16 full turns)
theta_head0 = 16 * 2 * np.pi

# corresponding arc length position of the head at t=0
s_head0 = spiral_length(theta_head0)


def generate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate position and velocity tables for ``times`` seconds."""
    cols = [f"{t} s" for t in times]

    pos_index = ["龙头x (m)", "龙头y (m)"]
    vel_index = ["龙头 (m/s)"]
    for i in range(1, N - 1):
        pos_index.extend([f"第{i}节龙身x (m)", f"第{i}节龙身y (m)"])
        vel_index.append(f"第{i}节龙身  (m/s)")
    pos_index.extend(["龙尾x (m)", "龙尾y (m)", "龙尾（后）x (m)", "龙尾（后）y (m)"])
    vel_index.extend(["龙尾  (m/s)", "龙尾（后） (m/s)"])

    pos_df = pd.DataFrame(index=pos_index, columns=cols, dtype=float)
    vel_df = pd.DataFrame(index=vel_index, columns=cols, dtype=float)

    # Theta values for head, body benches and tail rear handle
    thetas = np.zeros(N + 1)

    # initialise all segments at t=0
    thetas[0] = theta_head0
    for seg in range(1, N):
        # Benches are located behind the head along the spiral, so their
        # arc-length parameter is larger than the head's by the fixed
        # bench spacing.
        s_i = s_head0 + (D_HEAD + (seg - 1) * D_BODY)
        if s_i < 0:
            s_i = 0.0
        thetas[seg] = invert_length(s_i, thetas[seg - 1])
    s_tail_rear = s_head0 + (D_HEAD + (N - 1) * D_BODY)
    if s_tail_rear < 0:
        s_tail_rear = 0.0
    thetas[N] = invert_length(s_tail_rear, thetas[N - 1])

    x = np.zeros((N + 1, len(times)))
    y = np.zeros((N + 1, len(times)))

    for t_idx, t in enumerate(times):
        s_head = s_head0 - v_head * t
        thetas[0] = invert_length(s_head, thetas[0])
        x[0, t_idx] = A * thetas[0] * np.cos(thetas[0])
        y[0, t_idx] = A * thetas[0] * np.sin(thetas[0])

        for seg in range(1, N):
            # Each subsequent bench follows the head at a fixed arc-length
            # offset determined by the chain spacing.
            s_i = s_head + (D_HEAD + (seg - 1) * D_BODY)
            # When the rear segments move past the spiral origin their
            # theoretical arc length becomes negative.  In reality they
            # would remain at the origin instead of continuing along the
            # spiral with a negative radius, so clamp the value here.
            if s_i < 0:
                s_i = 0.0
            thetas[seg] = invert_length(s_i, thetas[seg])
            x[seg, t_idx] = A * thetas[seg] * np.cos(thetas[seg])
            y[seg, t_idx] = A * thetas[seg] * np.sin(thetas[seg])

        s_tail_rear = s_head + (D_HEAD + (N - 1) * D_BODY)
        if s_tail_rear < 0:
            s_tail_rear = 0.0
        thetas[N] = invert_length(s_tail_rear, thetas[N])
        x[N, t_idx] = A * thetas[N] * np.cos(thetas[N])
        y[N, t_idx] = A * thetas[N] * np.sin(thetas[N])

    # Fill position DataFrame
    for seg in range(len(pos_index) // 2):
        pos_df.loc[pos_index[2 * seg], :] = x[seg, :]
        pos_df.loc[pos_index[2 * seg + 1], :] = y[seg, :]

    # Compute velocities
    vel = np.zeros((N + 1, len(times)))
    diff_x = np.diff(x, axis=1)
    diff_y = np.diff(y, axis=1)
    vel[:, 1:] = np.sqrt(diff_x ** 2 + diff_y ** 2) / DT

    for idx, name in enumerate(vel_index):
        vel_df.loc[name, :] = vel[idx, :]

    return pos_df, vel_df


# Pre-compute tables so that other modules can reuse them directly
output, velocity = generate_data()


if __name__ == "__main__":
    # Only execute when run as a script.  Save the computed tables to an Excel
    # workbook so that they can be inspected or reused by other problems.
    with pd.ExcelWriter("result1.xlsx") as writer:
        output.to_excel(writer, sheet_name="位置")
        velocity.to_excel(writer, sheet_name="速度")


# ---------------------------------------------------------------------------
# Unsupervised maintenance clustering utilities (problem 1 auxiliary code)
# ---------------------------------------------------------------------------

from typing import Sequence

from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import PowerTransformer, StandardScaler
import matplotlib as mpl


WHITELIST_FEATURES = [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
]

LABEL_COL = "Failure_Within_7_Days"


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy keeping only whitelisted columns.

    Parameters
    ----------
    df:
        Raw input DataFrame.
    """

    cols = [c for c in WHITELIST_FEATURES if c in df.columns]
    keep = cols + ([LABEL_COL] if LABEL_COL in df.columns else [])
    if "Machine_ID" in df.columns:
        keep = ["Machine_ID"] + keep
    return df[keep].copy()


def setup_fonts() -> None:
    """Configure fonts with graceful degradation."""

    preferred = ["Hiragino Sans GB", "SimHei", "Arial Unicode MS"]
    existing = list(mpl.rcParams.get("font.sans-serif", []))
    for font in preferred:
        if font not in existing:
            existing.append(font)
    mpl.rcParams["font.sans-serif"] = existing


def name_clusters_by_risk(
    means: np.ndarray, feature_names: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    """Assign simple textual names to clusters based on mean values.

    The clusters are ordered by the mean of their features, with lower means
    interpreted as representing higher maintenance demand.
    """

    order = np.argsort(means.mean(axis=1))
    names = [f"Cluster {i}" for i in order]
    return names, order


def cluster_machines(df: pd.DataFrame) -> dict:
    """Perform GMM clustering with BIC-based model selection.

    This routine follows the modelling approach described in the project
    documentation.  It implements data-driven boundary detection, internal
    validity metrics and robust handling of optional labels.
    """

    setup_fonts()
    df = ensure_features(df)

    has_label = LABEL_COL in df.columns
    if has_label:
        y = df[LABEL_COL].fillna(0).astype(int).values
        X = df.drop(columns=[LABEL_COL])
    else:
        X = df
        y = None

    id_col = "Machine_ID" if "Machine_ID" in df.columns else None
    ids = df[id_col].values if id_col else np.arange(len(df))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pt = PowerTransformer(method="yeo-johnson")
    Xn = pt.fit_transform(Xs)

    bic_scores: list[float] = []
    models: list[GaussianMixture] = []
    for k in range(2, 11):
        init = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=0)
        init.fit(Xn)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=0,
            means_init=init.cluster_centers_,
        )
        gmm.fit(Xn)
        bic_scores.append(gmm.bic(Xn))
        models.append(gmm)

    best_idx = int(np.argmin(bic_scores))
    best_model = models[best_idx]

    gamma = best_model.predict_proba(Xn)
    max_gamma = gamma.max(axis=1)
    hard = gamma.argmax(axis=1)

    alpha = float(np.quantile(max_gamma, 0.70))
    boundary = (max_gamma < alpha).astype(int)

    # internal validity metrics (sampled)
    sample_idx = np.random.choice(len(Xn), size=min(20000, len(Xn)), replace=False)
    sil = float(silhouette_score(Xn[sample_idx], hard[sample_idx]))
    dbi = float(davies_bouldin_score(Xn[sample_idx], hard[sample_idx]))
    with open("q1_internal_validity.txt", "w", encoding="utf-8") as f:
        f.write(f"Silhouette(sampled): {sil:.4f}\nDBI(sampled): {dbi:.4f}\n")

    # output assignments
    out_assign = pd.DataFrame(
        {
            (id_col or "Machine_Index"): ids,
            "Cluster": hard,
            "max_gamma": max_gamma,
            "Boundary": boundary,
        }
    )
    out_assign.to_csv("q1_assignments.csv", index=False, encoding="utf-8-sig")

    # cluster naming
    names, order = name_clusters_by_risk(best_model.means_, X.columns)
    pd.DataFrame({"Cluster": list(order), "Name": names}).to_csv(
        "q1_cluster_names.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "best_k": int(best_model.n_components),
        "boundary_alpha_data_driven": alpha,
        "silhouette_sampled": sil,
        "dbi_sampled": dbi,
    }
    if y is not None:
        ari = float(adjusted_rand_score(y, hard))
        summary["ari_full"] = ari

    return summary

