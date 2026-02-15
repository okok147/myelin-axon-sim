"""Disease-specific phenomenological integrity field generators.

The generated fields are educational abstractions, not clinical models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .axon_builder import IntegrityProfile

PRESET_NAMES = [
    "MS",
    "NMOSD",
    "MOGAD",
    "ADEM",
    "PML",
    "ALD",
    "KRABBE",
    "MLD",
    "GBS",
    "CIDP",
    "CMT",
]


@dataclass
class DiseaseTrajectory:
    """Space-time disease fields over an axon."""

    preset: str
    times_norm: np.ndarray
    m: np.ndarray
    a: np.ndarray
    r: np.ndarray
    p: np.ndarray
    s: np.ndarray
    metadata: Dict[str, float | str]


def _gaussian_lesion_profile(x_um: np.ndarray, centers_um: np.ndarray, sigma_um: float) -> np.ndarray:
    if centers_um.size == 0:
        return np.zeros_like(x_um)
    profile = np.zeros_like(x_um)
    for c in centers_um:
        profile += np.exp(-0.5 * ((x_um - c) / sigma_um) ** 2)
    return np.clip(profile, 0.0, 1.0)


def _segment_lesion_profile(x_um: np.ndarray, centers_um: np.ndarray, length_um: float) -> np.ndarray:
    if centers_um.size == 0:
        return np.zeros_like(x_um)
    half = 0.5 * length_um
    profile = np.zeros_like(x_um)
    for c in centers_um:
        profile = np.maximum(profile, ((x_um >= c - half) & (x_um <= c + half)).astype(float))
    return np.clip(profile, 0.0, 1.0)


def _pick_centers(
    x_um: np.ndarray,
    lesion_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if lesion_count <= 0:
        return np.zeros(0, dtype=float)
    x_min = float(x_um.min() + 0.1 * (x_um.max() - x_um.min()))
    x_max = float(x_um.max() - 0.1 * (x_um.max() - x_um.min()))
    return rng.uniform(x_min, x_max, size=lesion_count)


def _clip_field(arr: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.clip(arr, low, high)


def generate_disease_trajectory(
    preset: str,
    x_um: np.ndarray,
    is_node: np.ndarray,
    axon_type: str,
    demyelination_severity: float,
    lesion_count: int,
    lesion_len_um: float,
    seed: int,
    n_timepoints: int = 60,
) -> DiseaseTrajectory:
    """Generate space-time fields m/a/r/p/s for a disease preset.

    Parameters
    ----------
    demyelination_severity:
        Severity scalar in [0, 1] (values >1 are clipped).
    """

    preset_u = preset.upper()
    if preset_u not in PRESET_NAMES:
        raise ValueError(f"Unknown preset '{preset}'.")

    sev = float(np.clip(demyelination_severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_timepoints)
    n_comp = x_um.size

    m = np.ones((n_timepoints, n_comp), dtype=float)
    a = np.ones((n_timepoints, n_comp), dtype=float)
    r = np.zeros((n_timepoints, n_comp), dtype=float)
    p = np.ones((n_timepoints, n_comp), dtype=float)
    s = np.ones((n_timepoints, n_comp), dtype=float)

    centers = _pick_centers(x_um, max(1, lesion_count), rng)
    sigma_um = max(lesion_len_um * 0.4, 40.0)
    segment_profile = _segment_lesion_profile(x_um, centers, lesion_len_um)
    gaussian_profile = _gaussian_lesion_profile(x_um, centers, sigma_um)

    diffuse_gradient = (x_um - x_um.min()) / max(float(np.ptp(x_um)), 1e-9)
    mild_noise = rng.normal(0.0, 0.03, size=(n_timepoints, n_comp))

    if preset_u == "MS":
        relapse_peaks = np.array([0.22, 0.55, 0.82])
        relapse = np.zeros_like(t)
        for pk in relapse_peaks:
            relapse += np.exp(-0.5 * ((t - pk) / 0.09) ** 2)
        relapse = relapse / max(relapse.max(), 1e-9)

        progressive = np.clip((t - 0.35) / 0.65, 0.0, 1.0)
        total_attack = 0.72 * relapse + 0.45 * progressive

        for i, attack in enumerate(total_attack):
            local_damage = sev * attack * gaussian_profile
            remyel = 0.28 * np.exp(-0.5 * ((t[i] - 0.9) / 0.18) ** 2) * gaussian_profile
            m[i] = 1.0 - 0.92 * local_damage + remyel
            p[i] = 1.0 - 0.75 * local_damage
            r[i] = 0.15 + 0.8 * local_damage
            a[i] = 1.0 - 0.18 * local_damage * progressive[i]

    elif preset_u == "NMOSD":
        severe_centers = _pick_centers(x_um, max(1, lesion_count // 2), rng)
        severe_profile = _segment_lesion_profile(x_um, severe_centers, lesion_len_um * 1.2)
        for i, ti in enumerate(t):
            attack = sev * (0.3 + 0.7 * np.exp(-0.5 * ((ti - 0.48) / 0.18) ** 2))
            s_drop = attack * severe_profile
            s[i] = 1.0 - 0.95 * s_drop
            m[i] = 1.0 - 0.82 * s_drop
            a[i] = 1.0 - 0.35 * s_drop
            p[i] = 1.0 - 0.68 * s_drop
            r[i] = 0.1 + 0.65 * s_drop

    elif preset_u == "MOGAD":
        episode1 = np.exp(-0.5 * ((t - 0.3) / 0.11) ** 2)
        episode2 = np.exp(-0.5 * ((t - 0.62) / 0.1) ** 2)
        attack = np.clip(episode1 + 0.75 * episode2, 0.0, 1.4)
        for i, att in enumerate(attack):
            local = sev * att * segment_profile
            recovery = 0.35 * np.exp(-0.5 * ((t[i] - 0.88) / 0.16) ** 2)
            m[i] = 1.0 - 0.88 * local + recovery * segment_profile
            p[i] = 1.0 - 0.62 * local
            r[i] = 0.08 + 0.55 * local
            a[i] = 1.0 - 0.12 * local

    elif preset_u == "ADEM":
        wide = 0.45 + 0.55 * (0.6 * gaussian_profile + 0.4 * diffuse_gradient)
        pulse = np.exp(-0.5 * ((t - 0.42) / 0.16) ** 2)
        for i, amp in enumerate(pulse):
            local = sev * amp * wide
            recovery = 0.6 * np.clip((t[i] - 0.55) / 0.45, 0.0, 1.0)
            m[i] = 1.0 - 0.75 * local + recovery * 0.25
            p[i] = 1.0 - 0.5 * local
            r[i] = 0.05 + 0.5 * local
            a[i] = 1.0 - 0.08 * local

    elif preset_u == "PML":
        start_center = _pick_centers(x_um, 1, rng)[0]
        base_radius = max(lesion_len_um * 0.35, 80.0)
        radius_growth = max(lesion_len_um * 1.8, 300.0)
        for i, ti in enumerate(t):
            radius = base_radius + radius_growth * ti
            front = np.exp(-0.5 * ((x_um - start_center) / radius) ** 2)
            damage = sev * (0.25 + 0.75 * ti) * front
            m[i] = 1.0 - 0.95 * damage
            p[i] = 1.0 - 0.8 * damage
            a[i] = 1.0 - 0.3 * damage
            s[i] = 1.0 - 0.5 * damage
            r[i] = 0.12 + 0.55 * damage

    elif preset_u == "ALD":
        step = (t > 0.35).astype(float) * 0.2 + (t > 0.7).astype(float) * 0.3
        for i, ti in enumerate(t):
            prog = sev * np.clip(0.2 + 0.65 * ti + step[i], 0.0, 1.2)
            field = 0.35 * gaussian_profile + 0.65 * diffuse_gradient
            local = prog * field
            m[i] = 1.0 - 0.88 * local
            p[i] = 1.0 - 0.66 * local
            a[i] = 1.0 - 0.24 * local
            s[i] = 1.0 - 0.22 * local
            r[i] = 0.1 + 0.58 * local

    elif preset_u == "KRABBE":
        cns_pns_factor = 1.0 if axon_type.upper() == "PNS" else 0.85
        for i, ti in enumerate(t):
            prog = sev * cns_pns_factor * np.clip(0.3 + 0.9 * ti, 0.0, 1.3)
            field = 0.6 + 0.4 * diffuse_gradient
            local = prog * field
            m[i] = 1.0 - 0.95 * local
            a[i] = 1.0 - 0.55 * local
            p[i] = 1.0 - 0.72 * local
            s[i] = 1.0 - 0.35 * local
            r[i] = 0.14 + 0.65 * local

    elif preset_u == "MLD":
        for i, ti in enumerate(t):
            prog = sev * np.clip(0.2 + 0.85 * ti, 0.0, 1.2)
            field = 0.45 + 0.55 * (0.5 * gaussian_profile + 0.5 * diffuse_gradient)
            local = prog * field
            m[i] = 1.0 - 0.9 * local
            p[i] = 1.0 - 0.64 * local
            a[i] = 1.0 - 0.2 * local
            s[i] = 1.0 - 0.28 * local
            r[i] = 0.1 + 0.55 * local

    elif preset_u == "GBS":
        # Acute segmental PNS demyelination with intermittent block-like fluctuations.
        segmental = _segment_lesion_profile(x_um, centers, lesion_len_um * 0.9)
        flicker = 0.75 + 0.25 * np.sin(2.0 * np.pi * (t[:, None] * 3.0 + diffuse_gradient[None, :] * 0.6))
        local = sev * segmental[None, :] * flicker
        m = 1.0 - 0.95 * local
        p = 1.0 - 0.88 * local
        a = 1.0 - 0.2 * local
        r = 0.12 + 0.65 * local

    elif preset_u == "CIDP":
        segmental = _segment_lesion_profile(x_um, centers, lesion_len_um)
        chronic = 0.55 + 0.45 * t[:, None]
        remission = 0.15 * np.sin(2.0 * np.pi * (t[:, None] * 1.2))
        local = sev * segmental[None, :] * np.clip(chronic + remission, 0.25, 1.25)
        m = 1.0 - 0.9 * local
        p = 1.0 - 0.82 * local
        a = 1.0 - 0.18 * local
        r = 0.1 + 0.62 * local

    elif preset_u == "CMT":
        # Uniform hereditary slowing pattern: diffuse mild-moderate myelin deficit.
        base = sev * np.clip(0.35 + 0.45 * t[:, None], 0.0, 0.95)
        spatial = 0.9 + 0.1 * np.sin(2.0 * np.pi * diffuse_gradient)[None, :]
        local = base * spatial
        m = 1.0 - 0.6 * local
        p = 1.0 - 0.3 * local
        a = 1.0 - 0.08 * local
        r = 0.05 + 0.25 * local

    # Apply small bounded heterogeneity for realism.
    m = _clip_field(m + mild_noise, 0.02, 1.0)
    a = _clip_field(a + 0.5 * mild_noise, 0.05, 1.0)
    p = _clip_field(p + 0.4 * mild_noise, 0.02, 1.0)
    s = _clip_field(s + 0.3 * mild_noise, 0.02, 1.0)
    r = _clip_field(r + 0.2 * mild_noise, 0.0, 1.0)

    # Keep node remodeling concentrated near nodes.
    node_weight = np.where(is_node, 1.0, 0.65)
    r *= node_weight[None, :]

    metadata: Dict[str, float | str] = {
        "severity": float(sev),
        "lesion_count": float(max(1, lesion_count)),
        "lesion_len_um": float(lesion_len_um),
        "axon_type": axon_type.upper(),
    }

    return DiseaseTrajectory(
        preset=preset_u,
        times_norm=t,
        m=m,
        a=a,
        r=r,
        p=p,
        s=s,
        metadata=metadata,
    )


def snapshot_from_trajectory(traj: DiseaseTrajectory, fraction: float) -> IntegrityProfile:
    """Select one disease stage (0..1 progression fraction)."""

    f = float(np.clip(fraction, 0.0, 1.0))
    idx = int(round(f * (traj.times_norm.size - 1)))
    return IntegrityProfile(
        m=traj.m[idx].copy(),
        a=traj.a[idx].copy(),
        r=traj.r[idx].copy(),
        p=traj.p[idx].copy(),
        s=traj.s[idx].copy(),
    )


def recommended_stage_fraction(preset: str) -> float:
    preset_u = preset.upper()
    if preset_u in {"MS", "MOGAD", "NMOSD"}:
        return 0.72
    if preset_u == "ADEM":
        return 0.45
    if preset_u == "PML":
        return 0.8
    if preset_u in {"ALD", "KRABBE", "MLD", "CIDP", "CMT"}:
        return 0.85
    if preset_u == "GBS":
        return 0.6
    return 0.75
