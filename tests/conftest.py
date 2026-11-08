import numpy as np
import pandas as pd
import pytest

from fusion.features import MODALITIES, feature_columns
from fusion.synthetic import make_recording

SHORT = [(0, 30.0), (1, 540.0), (0, 30.0), (2, 200.0), (0, 30.0), (3, 120.0), (0, 30.0)]


@pytest.fixture(scope="session")
def rec():
    return make_recording("S0", seed=0, protocol=SHORT, chest_fs=350.0)


def fake_table(n_subjects=6, n_per_class=12, seed=0, mods=None, shift=1.5) -> pd.DataFrame:
    """Window table with class-dependent features (no signal processing), for fast model tests."""
    rng = np.random.default_rng(seed)
    mods = list(MODALITIES) if mods is None else mods
    rows = []
    for s in range(n_subjects):
        for lab in (1, 2, 3):
            for _ in range(n_per_class):
                r = {"subject": f"S{s}", "label": lab}
                for m in mods:
                    for c in feature_columns([m]):
                        r[c] = rng.normal(shift * (lab == 2) + 0.3 * (lab == 3) + 0.2 * s, 1.0)
                    r[f"q__{m}"] = 1.0
                rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def table():
    return fake_table()
