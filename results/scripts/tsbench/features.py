"""Sequence-derived features for the surrogate models (no language-model embeddings).

F1: amino-acid composition (20) + dipeptide composition (400)            -> 420 dims, shift-invariant
F2: F1 + 10 physicochemical summaries                                    -> 430 dims
terminal(): composition of the first / last 3 residues (20 dims each)
"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
      "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
EXTRA_NAMES = ["mean_hydrophobicity", "net_charge", "frac_positive", "frac_negative", "frac_polar", "frac_aromatic",
               "frac_gly_pro", "hydrophobic_moment_180", "hydrophobic_moment_100", "hydrophobicity_autocorr_lag1"]


def encode(seqs):
    """Integer-encode equal-length sequences -> (n, L) array of indices into AA."""
    L = len(seqs[0])
    assert all(len(s) == L for s in seqs), "sequences must have equal length"
    table = np.full(256, -1, dtype=np.int16)
    for i, a in enumerate(AA):
        table[ord(a)] = i
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), L)
    idx = table[arr]
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} non-standard residues found")
    return idx.astype(np.int64)


def composition(idx):
    n, L = idx.shape
    aa = np.stack([(idx == a).sum(1) for a in range(20)], axis=1) / L
    pair = (idx[:, :-1] * 20 + idx[:, 1:]).ravel()
    rows = np.repeat(np.arange(n), L - 1)
    di = np.zeros((n, 400))
    np.add.at(di, (rows, pair), 1.0)
    return np.hstack([aa, di / (L - 1)]).astype(np.float32)


def physchem(idx):
    n, L = idx.shape
    kd = np.array([KD[a] for a in AA])
    h = kd[idx]
    frac = lambda letters: np.isin(idx, [AA.index(c) for c in letters]).mean(1)  # noqa: E731
    pos, neg = frac("KR"), frac("DE")
    j = np.arange(L)

    def moment(deg):
        ang = np.deg2rad(deg) * j
        return np.hypot((h * np.cos(ang)).sum(1), (h * np.sin(ang)).sum(1)) / L

    hc = h - h.mean(1, keepdims=True)
    num = (hc[:, :-1] * hc[:, 1:]).sum(1)
    den = (hc ** 2).sum(1) + 1e-9
    cols = [h.mean(1), pos - neg, pos, neg, frac("STNQ"), frac("FWY"), frac("GP"), moment(180), moment(100), num / den]
    return np.stack(cols, axis=1).astype(np.float32)


def build(seqs):
    idx = encode(list(seqs))
    f1 = composition(idx)
    f2 = np.hstack([f1, physchem(idx)])
    return f1, f2


def terminal(seqs, which):
    """Fractions of the 20 amino acids in the first (which='N') or last ('C') 3 residues -> (n, 20) float32."""
    X = np.zeros((len(seqs), 20), np.float32); lut = np.full(256, -1, np.int16); lut[[ord(c) for c in AA]] = np.arange(20)
    for i, s in enumerate(seqs):
        t = s[:3] if which == 'N' else s[-3:]
        for ch in t: X[i, lut[ord(ch)]] += 1.0 / 3.0
    return X
