"""SimulacraBench submission: a time-aware stacked ensemble for P(answer | GIVEN).

For a held-out respondent only the GIVEN block is visible, so every PREDICT
cell is a probabilistic classification of that item from the GIVEN block.  We
fit, per instrument and at run time (the only place the data exist):

  marg     smoothed crowd marginal
  bo*      hierarchical Dirichlet back-off over the item's most informative
           GIVEN columns (two concentrations)
  lr*      multinomial logistic regression on one-hot GIVEN (+ pairwise
           interactions of the strongest columns), all items in one batched
           torch model, several L2 strengths
  mlp      multi-task MLP (shared trunk, one softmax head per item)
  gbdt     histogram gradient boosting (only when time allows)

Out-of-fold predictions from K-fold cross-fitting feed a per-item stacker
(convex linear pool, fitted by EM on the floored log score).  Gates whose
parent is GIVEN and that the visible rows satisfy exactly are enforced.

Everything is wrapped: any failing component is dropped, a failure of the
whole pipeline falls back to gate-aware marginals, and a failure of that falls
back to uniform vectors, so predict() always returns legal output.
"""

import math
import os
import time

_T_IMPORT = time.time()

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

FLOOR = 1e-3

CONFIG = {
    "folds": 5,
    "seed": 20261019,
    "models": ["marg", "bo", "mlp", "lr", "bol", "hb", "hbw", "mlpx", "gbdt"],
    "bo_alphas": [3.0, 20.0],
    "bo_depth": 3,
    "bol_alphas": [10.0, 50.0],
    "lr_l2": [1e-3, 3e-3, 1e-2, 3e-2],
    "lr_iters": 100,
    "lr_pairs": 3,
    "mlp_hidden": 256,
    "mlp_dropout": 0.2,
    "mlp_wd": 1e-5,
    "mlp_lr": 2e-3,
    "mlp_epochs": 200,
    "mlp_patience": 12,
    "mlp_batch": 512,
    "mlp_seeds": 1,
    "mlp_layers": 2,
    "mlp_wide": True,
    "mlp_featdrop": 0.0,
    "mlp_item_es": False,
    # order = priority when time is short (Development phase): the mixture model
    # first (what v2 and v2+hbw ran first, and the World-Bank-like gain), then the
    # thermometer MLP (UNHCR-like gain), then the wide MLP
    "mlp_variants": [{"name": "", "mlp_seeds": 3},
                     {"name": "mix", "mlp_mix": "auto", "mlp_hidden": 128, "mlp_dropout": 0.1, "mlp_seeds": 2},
                     {"name": "t", "mlp_thermo": True, "mlp_seeds": 3},
                     {"name": "w", "mlp_hidden": 512, "mlp_dropout": 0.3, "mlp_seeds": 2}],
    "ordinal_bases": ["mlp", "mlpw"],
    # latent classes of the mixture model: 48 only where the proxies showed a gain
    # (UNHCR-like: UNHCR itself, or an unknown instrument with >= 150 items), else
    # 24 (World Bank, UNICEF: 48 was neutral on the proxies and the dev board's
    # World Bank bin dropped with it)
    "mix_classes": 24,
    "mix_classes_big": 48,
    "mix_big_min_items": 150,
    "hb_strong_params": [[1.0, 1.0], [1.0, 0.3, 1.0]],
    "gbdt_max_items": 400,
    "hb_params": [[0.0, 4.0], [0.3, 4.0], [0.0, 16.0]],
    "hb_radius": 1,
    "budget_frac": 0.55,      # share of the phase wall-clock we allow ourselves
    "budget_frac_long": 0.75,  # the same in the 3600 s Test phase (more GBDT items)
    # share of an instrument's model budget kept for boosting. Development: none
    # (with ~80 s per instrument a reserve buys boosting for 1-28 items, worth ~0,
    # and starves the MLP variants: v3.0 ran no MLP variant on World Bank and lost
    # a dev bin); Test: 30 %, enough for every item
    "gbdt_reserve": 0.0,
    "gbdt_reserve_long": 0.3,
    "gbdt_lr": 0.08,          # boosting schedule (0.08 x 200 trees, early stopping); when
    "gbdt_max_iter": 200,     # the budget cannot cover every item with it, switch to the
    "gbdt_fast": [0.15, 100],  # ~1.5x cheaper 0.15 x 100 schedule (-0.00005 skill, more items)
    # Test phase: cost shares per instrument (World Bank gets more than in v2)
    "known_cost_long": {"unicef": 2.0e6, "world_bank": 3.0e6, "unhcr": 6.0e6},
    "budget_override": None,  # seconds per instrument (local experiments)
    "enforce_gates": True,
    "gate_consistency": 0.0,
    "ordinal_smooth": True,
    "stack_prior": 2.0,
    "stack_method": "both",
    "stack_l2": 1e-3,
    "stack_iters": 60,
    "stack_max_rows": 20000,
    "verbose": False,
    "threads": None,
}

# extra work for the Test phase (3600 s budget); applied on top of CONFIG when
# the run budget is detected as the long one
# Optional extra work for the 3600 s Test phase, applied on top of CONFIG. Left
# empty on purpose: on the proxies more folds / seeds / variants did not help
# (phase-2 mean skill 0.43861 with extras vs 0.43868 without), and keeping
# both phases on one configuration means the Development-phase runs exercise
# exactly the code the single Test run will execute. The longer budget still
# lets the time-boxed GBDT cover more items.
PHASE2_EXTRA = {}

_STATE = {"done": [], "spent": 0.0}

# rough relative cost of the three known instruments (rows x sum K), used to
# split the run budget when we do not yet know which instrument comes next
_KNOWN_COST = {"unicef": 2.0e6, "world_bank": 2.0e6, "unhcr": 6.0e6}


def _log(*a):
    if CONFIG.get("verbose"):
        print("[sb %.1fs]" % (time.time() - _T_IMPORT), *a, flush=True)


# ===================================================================== utils

def _options(schema, rec):
    vals = [v for v in (rec.get("values") or [])]
    if rec.get("gate"):
        vals = vals + [schema.get("gated_value", "NA_GATED")]
    return vals


def _isna_array(series):
    """(missing mask, object values) for a frame column, robust to dtype."""
    arr = series.astype(object).to_numpy()
    try:
        na = np.asarray(pd.isna(series), dtype=bool).reshape(-1)
        if na.shape[0] != arr.shape[0]:
            raise ValueError
    except Exception:
        na = np.array([_is_missing(v) for v in arr], dtype=bool)
    return na, arr


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return False
    try:
        r = pd.isna(v)
        return bool(r) if np.ndim(r) == 0 else False
    except Exception:
        return False


def _encode(arr, levels):
    """Map raw values to indices of `levels`; missing or unknown -> -1."""
    lut = {}
    for i, lv in enumerate(levels):
        lut.setdefault(str(lv), i)
    out = np.full(len(arr), -1, dtype=np.int64)
    for i, v in enumerate(arr):
        if _is_missing(v):
            continue
        j = lut.get(v if isinstance(v, str) else str(v))
        if j is None and isinstance(v, float) and v.is_integer():
            j = lut.get(str(int(v)))
        if j is not None:
            out[i] = j
    return out


def _floor_mix(p, k):
    return FLOOR + (1.0 - k * FLOOR) * p


def _normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True)
    k = p.shape[-1]
    bad = ~(s > 0)
    p = np.where(bad, 1.0 / k, p / np.where(bad, 1.0, s))
    return p


def _blank_cells(frame, schema):
    """Canonical list of (row, item) blanks: rows top to bottom, schema item order."""
    items = [n for n, r in schema["items"].items()
             if isinstance(r, dict) and r.get("class") in ("GIVEN", "PREDICT") and n in frame.columns]
    masks = {}
    for n in items:
        masks[n], _ = _isna_array(frame[n])
    n_rows = len(frame)
    cells = []
    if not items:
        return cells, items
    M = np.stack([masks[n] for n in items], axis=1)
    rr, cc = np.nonzero(M)
    for r, c in zip(rr.tolist(), cc.tolist()):
        cells.append((r, items[c]))
    return cells, items


# ================================================================ data prep

class Data:
    pass


def _prepare(frame, schema):
    d = Data()
    recs = schema["items"]
    items = [n for n, r in recs.items() if isinstance(r, dict)
             and r.get("class") in ("GIVEN", "PREDICT") and n in frame.columns]
    d.given = [n for n in items if recs[n]["class"] == "GIVEN"]
    d.targets = [n for n in items if recs[n]["class"] == "PREDICT"]
    d.options = {n: _options(schema, recs[n]) for n in items}
    d.N = len(frame)
    gated_value = schema.get("gated_value", "NA_GATED")
    # GIVEN codes: levels + sentinel (if gated) + unknown bucket
    xs, cards = [], []
    d.given_levels = {}
    for g in d.given:
        lv = list(d.options[g])
        _, arr = _isna_array(frame[g])
        code = _encode(arr, lv)
        k = len(lv)
        code = np.where(code < 0, k, code)
        xs.append(code)
        cards.append(k + 1)
        d.given_levels[g] = lv
    d.X = np.stack(xs, axis=1) if xs else np.zeros((d.N, 0), dtype=np.int64)
    d.cards = cards
    # ordinal GIVEN columns (numeric bands in increasing schema order): their
    # cumulative "level >= m" indicators are extra features for LR / MLP
    d.thermo = []
    for f, g in enumerate(d.given):
        try:
            o = _given_ordinal(d.given_levels[g])
        except Exception:
            o = None
        if o:
            d.thermo.append((f, o))
    # PREDICT labels
    d.Y = np.full((d.N, len(d.targets)), -1, dtype=np.int64)
    d.test_mask = np.zeros((d.N, len(d.targets)), dtype=bool)
    for j, t in enumerate(d.targets):
        na, arr = _isna_array(frame[t])
        code = _encode(arr, d.options[t])
        code[na] = -1
        d.Y[:, j] = code
        d.test_mask[:, j] = na
    d.K = [len(d.options[t]) for t in d.targets]
    # training rows: every PREDICT label known; test rows: any PREDICT blank
    known = (d.Y >= 0)
    d.train_rows = np.nonzero(known.all(axis=1))[0] if d.targets else np.arange(d.N)
    if len(d.train_rows) < 0.5 * max(1, (known.any(axis=1)).sum()):
        d.train_rows = np.nonzero(known.any(axis=1))[0]
    d.test_rows = np.nonzero(d.test_mask.any(axis=1))[0] if d.targets else np.zeros(0, dtype=np.int64)
    # gates
    d.gate = {}
    for t in d.targets:
        g = recs[t].get("gate")
        if g and isinstance(g, dict) and g.get("parent") in recs:
            d.gate[t] = (g.get("parent"), list(g.get("observed_if") or []))
    d.gated_value = gated_value
    try:
        d.fp = _fingerprint(schema)
    except Exception:
        d.fp = "other"
    return d


# =================================================================== models

def _marginal(Ytr, K, alpha=0.5):
    out = []
    for j, k in enumerate(K):
        y = Ytr[:, j]
        y = y[y >= 0]
        c = np.bincount(y, minlength=k).astype(np.float64)[:k] + alpha
        out.append(c / c.sum())
    return out


def model_marg(Xtr, Ytr, Xev, ctx):
    m = _marginal(Ytr, ctx.K, 0.5)
    return [np.tile(p, (len(Xev), 1)) for p in m]


def _mutual_info(x, y, nx, ny):
    if len(x) < 10:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(np.float64)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _make_bo(alpha, depth):
    """Hierarchical Dirichlet back-off along the item's most informative GIVEN
    columns; the bottom of the ladder is the marginal or a base model."""
    def model_bo(Xtr, Ytr, Xev, ctx, base_ev=None):
        cards = ctx.cards
        out = []
        F = Xtr.shape[1]
        for j, k in enumerate(ctx.K):
            y = Ytr[:, j]
            ok = y >= 0
            xt, yt = Xtr[ok], y[ok]
            if base_ev is not None:
                run_ev = _normalize(base_ev[j]).copy()
            else:
                base = np.bincount(yt, minlength=k).astype(np.float64)[:k] + 0.5
                run_ev = np.tile(base / base.sum(), (len(Xev), 1))
            if F == 0 or len(yt) == 0:
                out.append(run_ev)
                continue
            mi = [_mutual_info(xt[:, f], yt, cards[f], k) for f in range(F)]
            order = list(np.argsort(mi)[::-1][:depth])
            key_tr = np.zeros(len(xt), dtype=np.int64)
            key_ev = np.zeros(len(Xev), dtype=np.int64)
            size = 1
            for f in order:
                if size * cards[f] > 2_000_000_000:
                    break
                key_tr = key_tr * cards[f] + xt[:, f]
                key_ev = key_ev * cards[f] + Xev[:, f]
                size *= cards[f]
                uk, inv_tr = np.unique(key_tr, return_inverse=True)
                cnt = np.zeros((len(uk), k))
                np.add.at(cnt, (inv_tr, yt), 1.0)
                pos = np.clip(np.searchsorted(uk, key_ev), 0, len(uk) - 1)
                hit = uk[pos] == key_ev
                c_ev = np.where(hit[:, None], cnt[pos], 0.0)
                run_ev = (c_ev + alpha * run_ev) / (c_ev.sum(1, keepdims=True) + alpha)
            out.append(run_ev)
        return out
    return model_bo


def _row_keys(X, mult):
    return (X.astype(np.int64) * mult[None, :]).sum(axis=1)


def _group_sums(keys_tr, Yoh, keys_ev):
    """Sum of one-hot labels over training rows sharing a key, gathered for eval rows."""
    uk, inv = np.unique(keys_tr, return_inverse=True)
    G = len(uk)
    S = np.zeros((G, Yoh.shape[1]), dtype=np.float32)
    np.add.at(S, inv, Yoh) if Yoh.shape[0] * Yoh.shape[1] < 2e5 else _add_rows(S, inv, Yoh)
    pos = np.clip(np.searchsorted(uk, keys_ev), 0, G - 1)
    hit = uk[pos] == keys_ev
    out = S[pos]
    out[~hit] = 0.0
    return out


def _add_rows(S, inv, Yoh):
    try:
        from scipy import sparse
        A = sparse.csr_matrix((np.ones(len(inv), dtype=np.float32), (inv, np.arange(len(inv)))),
                              shape=(S.shape[0], len(inv)))
        S += np.asarray(A @ Yoh, dtype=np.float32)
    except Exception:
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        starts = np.concatenate([[0], bounds])
        sums = np.add.reduceat(Yoh[order], starts, axis=0)
        S[inv[order][starts]] += sums


def _make_hb(lam1, alpha, radius, lam2=0.0):
    """Hot-deck / Hamming-ball donors: exact GIVEN matches, plus donors differing
    in one column (weight lam1) or two columns (weight lam2), backed off onto a
    base model with concentration alpha."""
    def model_hb(Xtr, Ytr, Xev, ctx, base_ev):
        K = ctx.K
        offs = np.concatenate([[0], np.cumsum(K)])
        n = len(Xtr)
        Yoh = np.zeros((n, int(offs[-1])), dtype=np.float32)
        for j in range(len(K)):
            y = Ytr[:, j]
            ok = y >= 0
            Yoh[np.nonzero(ok)[0], offs[j] + y[ok]] = 1.0
        F = Xtr.shape[1]
        rng = np.random.default_rng(12345)
        mult = rng.integers(1, 2 ** 40, size=F, dtype=np.int64) * 2 + 1
        ktr = _row_keys(Xtr, mult)
        kev = _row_keys(Xev, mult)
        dtr = Xtr.astype(np.int64) * mult[None, :]
        dev_ = Xev.astype(np.int64) * mult[None, :]
        N0 = _group_sums(ktr, Yoh, kev)
        C = N0.copy()
        if radius >= 1 and (lam1 > 0 or lam2 > 0) and F > 1:
            S1 = np.zeros_like(N0)
            for f in range(F):
                S1 += _group_sums(ktr - dtr[:, f], Yoh, kev - dev_[:, f])
            N1 = S1 - F * N0
            C += lam1 * N1
            if radius >= 2 and lam2 > 0 and F > 2:
                S2 = np.zeros_like(N0)
                for f in range(F):
                    for g in range(f + 1, F):
                        S2 += _group_sums(ktr - dtr[:, f] - dtr[:, g], Yoh, kev - dev_[:, f] - dev_[:, g])
                N2 = S2 - (F * (F - 1) / 2.0) * N0 - (F - 1) * N1
                C += lam2 * np.clip(N2, 0.0, None)
        out = []
        for j, k in enumerate(K):
            c = C[:, offs[j]:offs[j + 1]].astype(np.float64)
            b = base_ev[j] if base_ev is not None else np.full((len(Xev), k), 1.0 / k)
            b = _normalize(b)
            out.append((c + alpha * b) / (c.sum(1, keepdims=True) + alpha))
        return out
    return model_hb


def _block_sums(M, offs):
    """(rows x sum K) -> (rows x J): per-item sums of a flat one-hot layout."""
    return np.add.reduceat(M, offs[:-1], axis=1) if M.shape[1] else np.zeros((M.shape[0], 0))


def _make_hbw(n_fit=1500, iters=60):
    """Hot-deck with LEARNED per-column mismatch weights.

    Donor weight 1 for an exact GIVEN match and lambda_f for a donor that
    differs from the respondent only in column f; backed off onto the item
    marginal with concentration alpha:
        P_j(y) = (N0[y] + sum_f lambda_f N1_f[y] + alpha m_j[y]) / (n0 + sum_f lambda_f n1_f + alpha).
    lambda (in (0, 1)) and alpha are shared by all items and fitted by
    leave-one-out on a subsample of the training rows, on the floored log score.
    A column that does not matter gets lambda -> 1 (bigger donor pools); one
    that matters gets lambda -> 0."""
    def model_hbw(Xtr, Ytr, Xev, ctx):
        K = ctx.K
        J = len(K)
        offs = np.concatenate([[0], np.cumsum(K)]).astype(np.int64)
        n, F = Xtr.shape
        Yoh = np.zeros((n, int(offs[-1])), dtype=np.float32)
        for j in range(J):
            y = Ytr[:, j]
            ok = y >= 0
            Yoh[np.nonzero(ok)[0], offs[j] + y[ok]] = 1.0
        marg = _marginal(Ytr, K, 0.5)
        if F < 2 or n < 50 or not _HAS_TORCH:
            return [np.tile(m, (len(Xev), 1)) for m in marg]
        rng = np.random.default_rng(int(ctx.seed) + 3)
        mult = np.random.default_rng(12345).integers(1, 2 ** 40, size=F, dtype=np.int64) * 2 + 1
        ktr = _row_keys(Xtr, mult)
        kev = _row_keys(Xev, mult)
        dtr = Xtr.astype(np.int64) * mult[None, :]
        dev_ = Xev.astype(np.int64) * mult[None, :]
        # --- leave-one-out statistics on a subsample of training rows
        S = rng.choice(n, min(n, n_fit), replace=False)
        ylab = Ytr[S]
        okc = ylab >= 0
        cols = offs[:-1][None, :] + np.clip(ylab, 0, None)
        rr = np.arange(len(S))[:, None]
        N0 = _group_sums(ktr, Yoh, ktr[S])
        N0l = N0 - Yoh[S]
        A0 = N0l[rr, cols]
        T0 = _block_sums(N0l, offs)
        A1 = np.zeros((F, len(S), J), dtype=np.float32)
        T1 = np.zeros((F, len(S), J), dtype=np.float32)
        for f in range(F):
            G = _group_sums(ktr - dtr[:, f], Yoh, ktr[S] - dtr[S, f]) - N0
            A1[f] = G[rr, cols]
            T1[f] = _block_sums(G, offs)
        Mlab = np.stack([marg[j][np.clip(ylab[:, j], 0, None)] for j in range(J)], axis=1)
        kvec = np.asarray(K, dtype=np.float64)[None, :]
        dev = torch.device("cpu")
        tA0 = torch.tensor(A0, dtype=torch.float64, device=dev)
        tT0 = torch.tensor(T0, dtype=torch.float64, device=dev)
        tA1 = torch.tensor(A1, dtype=torch.float64, device=dev)
        tT1 = torch.tensor(T1, dtype=torch.float64, device=dev)
        tM = torch.tensor(Mlab, dtype=torch.float64, device=dev)
        tok = torch.tensor(okc, dtype=torch.float64, device=dev)
        tk = torch.tensor(1.0 - kvec * FLOOR, dtype=torch.float64, device=dev)
        u = torch.full((F,), math.log(0.3 / 0.7), dtype=torch.float64, requires_grad=True)
        v = torch.tensor([math.log(4.0)], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lam = torch.sigmoid(u)
            al = torch.exp(v)
            num = tA0 + torch.einsum("f,fsj->sj", lam, tA1) + al * tM
            den = tT0 + torch.einsum("f,fsj->sj", lam, tT1) + al
            lp = torch.log(FLOOR + tk * num / den)
            loss = -(lp * tok).sum() / torch.clamp(tok.sum(), min=1.0)
            loss.backward()
            return loss
        try:
            opt.step(closure)
            lam = torch.sigmoid(u).detach().numpy()
            al = float(torch.exp(v).detach())
            if not (np.all(np.isfinite(lam)) and np.isfinite(al)):
                raise ValueError
        except Exception:
            lam, al = np.full(F, 0.3), 4.0
        ctx.hbw_params = (lam, al)
        # --- predict the evaluation rows
        C = _group_sums(ktr, Yoh, kev).astype(np.float64)
        N0e = C.copy()
        for f in range(F):
            if lam[f] < 1e-4:
                continue
            C += lam[f] * (_group_sums(ktr - dtr[:, f], Yoh, kev - dev_[:, f]) - N0e)
        out = []
        for j, k in enumerate(K):
            c = np.clip(C[:, offs[j]:offs[j + 1]], 0.0, None)
            out.append((c + al * marg[j][None, :]) / (c.sum(1, keepdims=True) + al))
        return out
    return model_hbw


# ---------------------------------------------------------------- torch nets

_DEVICE = []


def _device():
    """CUDA if a tiny probe op works there, else CPU (cached)."""
    if _DEVICE:
        return _DEVICE[0]
    dev = torch.device("cpu")
    try:
        if _HAS_TORCH and torch.cuda.is_available():
            t = torch.ones(8, device="cuda")
            emb = torch.nn.EmbeddingBag(4, 3, mode="sum").to("cuda")
            _ = float((emb(torch.zeros((2, 2), dtype=torch.long, device="cuda")).sum() + t.sum()).item())
            dev = torch.device("cuda")
    except Exception:
        dev = torch.device("cpu")
    _DEVICE.append(dev)
    return dev


def _given_ordinal(levels):
    """Indices, in increasing order, of the levels of a GIVEN column that are
    numeric bands ('0', '1-1999', '25000+', '18 to 24 years old', 'Under 18');
    None when the column is not such a scale.  Levels that do not start with a
    number (e.g. 'Not answered') are left out; enumerated labels such as
    '2. Married' are codes, not quantities, and disqualify the column."""
    import re
    pat = re.compile(r"^\s*(under|less than|fewer than|below|<|>|more than|over)?\s*(\d[\d,]*(?:\.\d+)?)",
                     re.IGNORECASE)
    idx, keys = [], []
    for i, v in enumerate(levels):
        if re.match(r"^\s*\d+\s*[\.\)]\s*[A-Za-z]", str(v)):
            return None
        m = pat.match(str(v))
        if not m:
            continue
        x = float(m.group(2).replace(",", ""))
        pre = (m.group(1) or "").lower()
        if pre in ("under", "less than", "fewer than", "below", "<"):
            x -= 0.5
        elif pre:
            x += 0.5
        idx.append(i)
        keys.append(x)
    if len(idx) < 4 or len(idx) < 0.6 * len(levels) or len(set(keys)) != len(keys):
        return None
    return [i for _, i in sorted(zip(keys, idx))]


def _onehot_design(X, cards, pairs, thermo=None):
    """One-hot main effects plus one-hot of selected feature pairs (as index lists).

    `thermo` [(column, ordered level indices)]: each threshold m of an ordinal
    column adds one 3-state slot (below m / at or above m / not on the scale)."""
    offs = 0
    idx = []
    for f, c in enumerate(cards):
        idx.append(X[:, f] + offs)
        offs += c
    for (a, b) in pairs:
        idx.append(X[:, a] * cards[b] + X[:, b] + offs)
        offs += cards[a] * cards[b]
    for (f, order) in (thermo or []):
        rank = np.full(cards[f], -1, dtype=np.int64)
        rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
        r = rank[np.clip(X[:, f], 0, cards[f] - 1)]
        for m in range(1, len(order)):
            state = np.where(r < 0, 2, (r >= m).astype(np.int64))
            idx.append(state + offs)
            offs += 3
    return np.stack(idx, axis=1) if idx else np.zeros((len(X), 0), dtype=np.int64), offs


class _Heads:
    """Flat (sum K) logits; softmax per item, computed group-wise over items that
    share an option count so no padding is materialised."""

    def __init__(self, K, device):
        self.K = list(K)
        self.J = len(K)
        self.S = int(sum(K))
        offs = np.concatenate([[0], np.cumsum(K)]).astype(np.int64)
        self.offs = offs
        groups = {}
        for j, k in enumerate(K):
            groups.setdefault(int(k), []).append(j)
        self.groups = []
        for k, js in sorted(groups.items()):
            cols = np.concatenate([np.arange(offs[j], offs[j] + k) for j in js])
            self.groups.append((k, np.array(js, dtype=np.int64),
                                torch.tensor(cols, device=device)))

    def nll(self, logits, Y):
        tot = logits.new_zeros(())
        cnt = 0
        for k, js, cols in self.groups:
            z = logits[:, cols].view(-1, len(js), k)
            lp = torch.log_softmax(z, dim=-1)
            yg = Y[:, js]
            ok = yg >= 0
            g = lp.gather(2, torch.clamp(yg, min=0).unsqueeze(-1)).squeeze(-1)
            tot = tot - (g * ok).sum()
            cnt = cnt + ok.sum()
        return tot / torch.clamp(cnt, min=1) if torch.is_tensor(cnt) else tot / max(cnt, 1)

    def nll_items(self, logits, Y):
        """Mean NLL per item (J,), +inf where an item has no observed label."""
        out = torch.full((self.J,), float("inf"), device=logits.device)
        for k, js, cols in self.groups:
            z = logits[:, cols].view(-1, len(js), k)
            lp = torch.log_softmax(z, dim=-1)
            yg = Y[:, js]
            ok = (yg >= 0).float()
            g = lp.gather(2, torch.clamp(yg, min=0).unsqueeze(-1)).squeeze(-1)
            cnt = ok.sum(0)
            v = -(g * ok).sum(0) / torch.clamp(cnt, min=1)
            v = torch.where(cnt > 0, v, torch.full_like(v, float("inf")))
            out[torch.as_tensor(js, device=logits.device)] = v
        return out

    def probs(self, logits):
        out = torch.empty_like(logits)
        for k, js, cols in self.groups:
            z = logits[:, cols].view(-1, len(js), k)
            out[:, cols] = torch.softmax(z, dim=-1).reshape(logits.shape[0], -1)
        return out

    def split(self, P):
        return [P[:, self.offs[j]:self.offs[j + 1]].astype(np.float64) for j in range(self.J)]


class _LRNet(torch.nn.Module if _HAS_TORCH else object):
    def __init__(self, D, out):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(D, out, mode="sum")
        torch.nn.init.zeros_(self.emb.weight)
        self.bias = torch.nn.Parameter(torch.zeros(out))

    def forward(self, idx):
        return self.emb(idx) + self.bias


def _fit_lr(idx_tr, Ytr, idx_ev, D, K, l2_list, ctx):
    """Batched multinomial LR for all items, full-batch L-BFGS along a warm-started
    L2 path (strongest penalty first)."""
    dev = ctx.device
    heads = _Heads(K, dev)
    it = torch.tensor(idx_tr, device=dev)
    ie = torch.tensor(idx_ev, device=dev)
    Y = torch.tensor(Ytr, device=dev)
    res = {}
    torch.manual_seed(0)
    net = _LRNet(D, heads.S).to(dev)
    with torch.no_grad():   # start from the log marginals
        b0 = []
        for j, k in enumerate(K):
            y = Ytr[:, j]
            c = np.bincount(y[y >= 0], minlength=k).astype(np.float64)[:k] + 0.5
            b0.append(np.log(c / c.sum()))
        net.bias.copy_(torch.tensor(np.concatenate(b0), dtype=net.bias.dtype, device=dev))
    nobs_item = float(max(1, Ytr.shape[1]))
    for step, l2 in enumerate(sorted(l2_list, reverse=True)):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=int(CONFIG["lr_iters"]) if step == 0
                                else int(CONFIG["lr_iters"]) // 2 + 10, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

        hard = getattr(ctx, "hard_deadline", None)

        def closure():
            if hard is not None and time.time() > hard:
                raise TimeoutError("lr over budget")
            opt.zero_grad()
            loss = heads.nll(net(it), Y) + l2 * (net.emb.weight ** 2).sum() / nobs_item
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except TimeoutError:
            raise
        except Exception:
            pass
        with torch.no_grad():
            P = heads.probs(net(ie)).cpu().numpy()
        res[l2] = heads.split(P)
    return res


def _choose_pairs(Xtr, Ytr, cards, K, npairs):
    if npairs <= 0 or Xtr.shape[1] < 2:
        return []
    F = Xtr.shape[1]
    score = np.zeros(F)
    for j, k in enumerate(K):
        y = Ytr[:, j]
        ok = y >= 0
        for f in range(F):
            score[f] += _mutual_info(Xtr[ok, f], y[ok], cards[f], k)
    top = list(np.argsort(score)[::-1][:npairs])
    pairs = []
    for a in range(len(top)):
        for b in range(F):
            fa, fb = top[a], b
            if fb == fa or (fb in top[:a]):
                continue
            if cards[fa] * cards[fb] > 4000:
                continue
            pairs.append((fa, fb))
    return pairs


def model_lr_all(Xtr, Ytr, Xev, ctx, thermo=False):
    """Returns {name: preds} for several L2 strengths (main effects + pairs)."""
    pairs = ctx.pairs
    l2s = list(CONFIG["lr_l2"])
    if ctx.device.type == "cpu" and not CONFIG.get("full_on_cpu"):
        # no GPU: a cheaper path so the MLP still fits in time
        l2s = [v for v in l2s if 2e-3 <= v <= 2e-2] or l2s[:2]
        if len(ctx.K) > 60:
            pairs = pairs[:max(0, len(pairs) // 4)]
    th = getattr(ctx, "thermo", None) if thermo else None
    idx_tr, D = _onehot_design(Xtr, ctx.cards, pairs, th)
    idx_ev, _ = _onehot_design(Xev, ctx.cards, pairs, th)
    res = _fit_lr(idx_tr, Ytr, idx_ev, D, ctx.K, l2s, ctx)
    return {("lrt%g" if thermo else "lr%g") % l2: v for l2, v in res.items()}


class _MLP(torch.nn.Module if _HAS_TORCH else object):
    def __init__(self, D, H, out, p, layers=2, wide=True, featdrop=0.0):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(D, H, mode="sum")
        self.b0 = torch.nn.Parameter(torch.zeros(H))
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(max(0, layers - 1))])
        self.out = torch.nn.Linear(H, out)
        self.use_wide = wide
        if wide:
            self.wide = torch.nn.EmbeddingBag(D, out, mode="sum")
            torch.nn.init.zeros_(self.wide.weight)
        self.drop = torch.nn.Dropout(p)
        self.featdrop = featdrop

    def forward(self, idx):
        w = None
        if self.training and self.featdrop > 0:
            keep = (torch.rand(idx.shape, device=idx.device) > self.featdrop).float()
            w = keep / (1.0 - self.featdrop)
        h = torch.nn.functional.gelu(self.emb(idx, per_sample_weights=w) + self.b0)
        h = self.drop(h)
        for lin in self.hidden:
            h = torch.nn.functional.gelu(lin(h))
            h = self.drop(h)
        z = self.out(h)
        if self.use_wide:
            z = z + self.wide(idx, per_sample_weights=w)
        return z


def model_mlp(Xtr, Ytr, Xev, ctx, cfg=None):
    C = dict(CONFIG)
    if cfg:
        C.update(cfg)
    dev = ctx.device
    th = getattr(ctx, "thermo", None) if C.get("mlp_thermo") else None
    idx_tr, D = _onehot_design(Xtr, ctx.cards, [], th)
    idx_ev, _ = _onehot_design(Xev, ctx.cards, [], th)
    if C.get("mlp_mix"):
        return _fit_mix(idx_tr, Ytr, idx_ev, D, ctx, C)
    heads = _Heads(ctx.K, dev)
    n = len(idx_tr)
    J = heads.J
    item_es = bool(C.get("mlp_item_es"))
    item_of_col = torch.tensor(np.repeat(np.arange(J), ctx.K), device=dev)
    rng = np.random.default_rng(ctx.seed + 7 + int(C.get("mlp_seed_offset", 0)))
    preds = None
    Ie = torch.tensor(idx_ev, device=dev)

    def predict_ev(net):
        outs = []
        for b in range(0, len(idx_ev), 4096):
            outs.append(heads.probs(net(Ie[b:b + 4096])))
        return torch.cat(outs, 0) if outs else torch.zeros((0, heads.S), device=dev)

    n_seeds = max(1, int(C["mlp_seeds"]))
    fold_budget = float(getattr(ctx, "fold_budget", 0.0) or 0.0)
    for s in range(n_seeds):
        seed_deadline = time.time() + fold_budget / n_seeds if fold_budget > 0 else None
        perm = rng.permutation(n)
        nval = max(1, int(0.1 * n))
        va, tr = perm[:nval], perm[nval:]
        torch.manual_seed(ctx.seed + s + int(C.get("mlp_seed_offset", 0)))
        net = _MLP(D, C["mlp_hidden"], heads.S, C["mlp_dropout"], C["mlp_layers"],
                   C["mlp_wide"], C["mlp_featdrop"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=C["mlp_lr"], weight_decay=C["mlp_wd"])
        Itr = torch.tensor(idx_tr[tr], device=dev)
        Ytr_t = torch.tensor(Ytr[tr], device=dev)
        Iva = torch.tensor(idx_tr[va], device=dev)
        Yva = torch.tensor(Ytr[va], device=dev)
        bs = int(C["mlp_batch"])
        best, best_state, bad = 1e18, None, 0
        best_item = torch.full((J,), float("inf"), device=dev)
        bestP = None
        for ep in range(int(C["mlp_epochs"])):
            net.train()
            order = torch.randperm(len(tr), device=dev)
            for b in range(0, len(tr), bs):
                bi = order[b:b + bs]
                opt.zero_grad()
                loss = heads.nll(net(Itr[bi]), Ytr_t[bi])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                zv = net(Iva)
                if item_es:
                    li = heads.nll_items(zv, Yva)
                    imp = li < best_item - 1e-6
                    if bool(imp.any()):
                        Pev = predict_ev(net)
                        if bestP is None:
                            bestP = Pev.clone()
                        else:
                            cm = imp[item_of_col]
                            bestP[:, cm] = Pev[:, cm]
                        best_item = torch.where(imp, li, best_item)
                        bad = 0
                    else:
                        bad += 1
                    if bad >= C["mlp_patience"]:
                        break
                else:
                    vl = float(heads.nll(zv, Yva))
                    if vl < best - 1e-5:
                        best, bad = vl, 0
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= C["mlp_patience"]:
                            break
            if ctx.deadline is not None and time.time() > ctx.deadline:
                break
            if seed_deadline is not None and time.time() > seed_deadline:
                break
        net.eval()
        with torch.no_grad():
            if item_es and bestP is not None:
                P = bestP.cpu().numpy()
            else:
                if best_state is not None:
                    net.load_state_dict(best_state)
                P = predict_ev(net).cpu().numpy()
        sp = heads.split(P)
        preds = sp if preds is None else [a + b for a, b in zip(preds, sp)]
    S = max(1, int(C["mlp_seeds"]))
    return [p / S for p in preds]


class _MixNet(torch.nn.Module if _HAS_TORCH else object):
    """Mixture of softmaxes over the PREDICT block: P(y_j | x) = sum_c pi_c(x) theta_jc,
    pi from a one-hidden-layer net on the GIVEN one-hot, theta free per class
    (a latent-class model with covariates, fitted on the per-item log score)."""

    def __init__(self, D, H, C, S, p, theta0):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(D, H, mode="sum")
        self.b0 = torch.nn.Parameter(torch.zeros(H))
        self.gate = torch.nn.Linear(H, C)
        self.drop = torch.nn.Dropout(p)
        self.theta = torch.nn.Parameter(theta0)

    def log_pi(self, idx):
        h = self.drop(torch.nn.functional.gelu(self.emb(idx) + self.b0))
        return torch.log_softmax(self.gate(h), dim=-1)


def _mix_nll(net, heads, I, Y):
    lpi = net.log_pi(I)
    C = lpi.shape[1]
    tot = lpi.new_zeros(())
    cnt = 0
    for k, js, cols in heads.groups:
        lt = torch.log_softmax(net.theta[:, cols].view(C, len(js), k), dim=-1).permute(1, 2, 0)
        yg = Y[:, js]
        ok = yg >= 0
        yc = torch.clamp(yg, min=0)
        jidx = torch.arange(len(js), device=yc.device).unsqueeze(0).expand_as(yc)
        g = lt[jidx, yc]                                   # B x Jg x C
        lp = torch.logsumexp(g + lpi.unsqueeze(1), dim=-1)  # B x Jg
        tot = tot - (lp * ok).sum()
        cnt = cnt + ok.sum()
    return tot / torch.clamp(cnt, min=1)


def _mix_probs(net, heads, I):
    pi = torch.exp(net.log_pi(I))
    th = heads.probs(net.theta)
    return pi @ th


def _fit_mix(idx_tr, Ytr, idx_ev, D, ctx, C):
    dev = ctx.device
    heads = _Heads(ctx.K, dev)
    nC = int(C["mlp_mix"])
    n = len(idx_tr)
    off = int(C.get("mlp_seed_offset", 0))
    rng = np.random.default_rng(ctx.seed + 11 + off)
    Ie = torch.tensor(idx_ev, device=dev)
    b0 = []
    for j, k in enumerate(ctx.K):
        y = Ytr[:, j]
        c = np.bincount(y[y >= 0], minlength=k).astype(np.float64)[:k] + 0.5
        b0.append(np.log(c / c.sum()))
    b0 = np.concatenate(b0)
    n_seeds = max(1, int(C["mlp_seeds"]))
    fold_budget = float(getattr(ctx, "fold_budget", 0.0) or 0.0)
    preds = None
    for s in range(n_seeds):
        seed_deadline = time.time() + fold_budget / n_seeds if fold_budget > 0 else None
        perm = rng.permutation(n)
        nval = max(1, int(0.1 * n))
        va, tr = perm[:nval], perm[nval:]
        torch.manual_seed(ctx.seed + s + off)
        theta0 = torch.tensor(b0[None, :] + float(C.get("mix_init", 0.5)) * rng.standard_normal((nC, len(b0))),
                              dtype=torch.float32)
        net = _MixNet(D, int(C["mlp_hidden"]), nC, heads.S, float(C["mlp_dropout"]), theta0).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(C.get("mix_lr", 5e-3)), weight_decay=C["mlp_wd"])
        Itr = torch.tensor(idx_tr[tr], device=dev)
        Yt = torch.tensor(Ytr[tr], device=dev)
        Iva = torch.tensor(idx_tr[va], device=dev)
        Yva = torch.tensor(Ytr[va], device=dev)
        bs = int(C["mlp_batch"])
        best, best_state, bad = 1e18, None, 0
        for ep in range(int(C["mlp_epochs"])):
            net.train()
            order = torch.randperm(len(tr), device=dev)
            for b in range(0, len(tr), bs):
                bi = order[b:b + bs]
                opt.zero_grad()
                loss = _mix_nll(net, heads, Itr[bi], Yt[bi])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(_mix_nll(net, heads, Iva, Yva))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= C["mlp_patience"]:
                    break
            if ctx.deadline is not None and time.time() > ctx.deadline:
                break
            if seed_deadline is not None and time.time() > seed_deadline:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            outs = [_mix_probs(net, heads, Ie[b:b + 4096]) for b in range(0, len(idx_ev), 4096)]
            P = (torch.cat(outs, 0) if outs else torch.zeros((0, heads.S), device=dev)).cpu().numpy()
        sp = heads.split(P)
        preds = sp if preds is None else [a + b for a, b in zip(preds, sp)]
    return [p / n_seeds for p in preds]


def _gbdt_fit_predict(Xtr, y, Xev, k, catmask, lr=None, max_iter=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    lr = float(lr if lr is not None else CONFIG.get("gbdt_lr", 0.08))
    max_iter = int(max_iter if max_iter is not None else CONFIG.get("gbdt_max_iter", 200))
    ok = y >= 0
    cnt = np.bincount(y[ok], minlength=k)
    p = np.tile((cnt + 0.5) / (cnt + 0.5).sum(), (len(Xev), 1))
    keep = ok & (cnt[np.maximum(y, 0)] >= 8)   # classes too rare for a stratified split
    present = np.unique(y[keep])
    if len(present) < 2:
        return p
    try:
        clf = HistGradientBoostingClassifier(learning_rate=lr, max_iter=max_iter, max_leaf_nodes=15,
                                             min_samples_leaf=20, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.12,
                                             n_iter_no_change=12, categorical_features=catmask,
                                             random_state=0)
        clf.fit(Xtr[keep], y[keep])
        pp = clf.predict_proba(Xev)
        q = np.zeros((len(Xev), k))
        cls = clf.classes_.astype(np.int64)
        q[:, cls] = pp
        rare = np.setdiff1d(np.arange(k), cls)
        if len(rare):   # rare classes keep a sliver of their marginal mass
            q[:, rare] = p[:, rare]
        return _normalize(q)
    except Exception:
        return p


def model_gbdt(Xtr, Ytr, Xev, ctx, base_ev=None):
    """Per-item gradient boosting on the items in ctx.gbdt_items; every other
    item copies the base model so the stacker sees no difference there."""
    out = []
    catmask = np.array([c <= 250 for c in ctx.cards], dtype=bool)
    items = set(getattr(ctx, "gbdt_items", range(len(ctx.K))))
    for j, k in enumerate(ctx.K):
        if j not in items and base_ev is not None:
            out.append(_normalize(base_ev[j]))
            continue
        out.append(_gbdt_fit_predict(Xtr, Ytr[:, j], Xev, k, catmask,
                                     getattr(ctx, "gbdt_lr", None), getattr(ctx, "gbdt_max_iter", None)))
        if ctx.deadline is not None and time.time() > ctx.deadline:
            raise TimeoutError("gbdt over budget")
    return out


# ======================================================== ordinal smoothing

def _ordinal_index(values):
    """Indices of the values that look ordinal (numbers, bands, Likert digits),
    in schema order, or None when the item does not look ordinal."""
    import re
    idx = [i for i, v in enumerate(values) if re.search(r"\d", str(v))
           and not re.match(r"^\s*\d+\.\s+[A-Za-z]", str(v))]
    if len(idx) < 4 or len(idx) < 0.6 * len(values):
        return None

    def key(v):
        s = str(v).strip().lower()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        x = float(m.group(0)) if m else 0.0
        if s.startswith("<") or s.startswith("before") or s.startswith("under") or s.startswith("less"):
            x -= 1e-3
        if "year" in s and "month" not in s.split("year")[0]:
            x *= 12.0
        elif "week" in s:
            x /= 4.0
        if s.endswith("+") or s.startswith("more") or s.startswith(">") or s.startswith("over"):
            x += 1e-3
        return x
    keys = [key(values[i]) for i in idx]
    small = [k for k in keys if k < 90]
    if small and len(small) >= 4:   # drop 97/98/99-style non-response codes
        keep = [not (str(values[i]).strip() in ("96", "97", "98", "99") and max(small) < 50)
                for i in idx]
        idx = [i for i, kp in zip(idx, keep) if kp]
        keys = [k for k, kp in zip(keys, keep) if kp]
    if len(set(keys)) == len(keys):
        idx = [i for _, i in sorted(zip(keys, idx))]
    return idx


def _smooth_ordinal(P, idx, w=0.5):
    """Mass-preserving [w/2, 1-w, w/2] kernel along the ordinal block."""
    Q = P.copy()
    B = P[:, idx]
    left = np.concatenate([B[:, :1], B[:, :-1]], axis=1)
    right = np.concatenate([B[:, 1:], B[:, -1:]], axis=1)
    Q[:, idx] = (1 - w) * B + 0.5 * w * (left + right)
    return Q


def _add_ordinal_variants(d, oof, test, bases):
    ords = [_ordinal_index(d.options[t]) for t in d.targets]
    if not any(o is not None for o in ords):
        return
    for b in bases:
        if b not in oof:
            continue
        nm = b + "_os"
        oof[nm] = [(_smooth_ordinal(_normalize(oof[b][j]), ords[j]) if ords[j] else oof[b][j])
                   for j in range(len(d.K))]
        test[nm] = [(_smooth_ordinal(_normalize(test[b][j]), ords[j]) if ords[j] else test[b][j])
                    for j in range(len(d.K))]


# ================================================================== stacking

def _stack_weights(oof_list, y, k, prior_w, prior_strength, iters=200):
    """EM for convex mixture weights maximizing the floored log score."""
    M = len(oof_list)
    ok = y >= 0
    yy = y[ok]
    L = np.stack([_normalize(o[ok])[np.arange(ok.sum()), yy] for o in oof_list], axis=1)  # n x M
    L = _floor_mix(L, k)  # floor is linear, so mixing floored = flooring mixture
    w = np.array(prior_w, dtype=np.float64)
    w = w / w.sum()
    a = prior_strength * w
    for _ in range(iters):
        num = L * w
        den = num.sum(1, keepdims=True)
        r = num / np.maximum(den, 1e-300)
        nw = r.sum(0) + a
        nw = nw / nw.sum()
        if np.abs(nw - w).max() < 1e-6:
            w = nw
            break
        w = nw
    ll = float(np.log(np.maximum((L * w).sum(1), 1e-300)).mean())
    return w, ll


# ================================================================== pipeline

class Ctx:
    pass


def _budget_for(d, schema, frame):
    """Seconds this instrument may spend."""
    if CONFIG.get("budget_override"):
        return float(CONFIG["budget_override"])
    split = schema.get("split") or {}
    n_hidden = len(d.test_rows)
    phase_total = 900.0
    try:
        if int(split.get("n_test", -1)) == n_hidden and int(split.get("n_dev", -2)) != n_hidden:
            phase_total = 3600.0
    except Exception:
        pass
    _STATE["phase_total"] = phase_total
    frac = CONFIG["budget_frac"]
    if phase_total >= 3600.0 and CONFIG.get("budget_frac_long"):
        frac = float(CONFIG["budget_frac_long"])
    total = frac * phase_total
    used = time.time() - _T_IMPORT
    remaining = max(0.0, total - used)
    fp = _fingerprint(schema)
    known = _phase_cfg("known_cost") or _KNOWN_COST
    my_cost = known.get(fp, float(d.N) * float(sum(d.K) + 1))
    others = sum(c for nm, c in known.items() if nm != fp and nm not in _STATE["done"])
    if fp not in known:
        others = sum(c for nm, c in known.items() if nm not in _STATE["done"])
        others = others * 2.0 / 3.0
    share = my_cost / max(my_cost + others, 1.0)
    return max(20.0, remaining * share)


def _phase_cfg(key, default=None):
    """CONFIG[key], or CONFIG[key + '_long'] in the 3600 s Test phase when set."""
    if _STATE.get("phase_total", 900.0) >= 3600.0 and (key + "_long") in CONFIG:
        return CONFIG[key + "_long"]
    return CONFIG.get(key, default)


def _fingerprint(schema):
    names = set(schema.get("items", {}).keys())
    if "Q08" in names and "Country" in names:
        return "unicef"
    if "sc_nstd_extra" in names or "tr_job_full" in names:
        return "world_bank"
    if "s3_11" in " ".join(names) or "s1_2_country_asylum_lbl" in names:
        return "unhcr"
    return "other"


def _run_models(d, budget):
    t0 = time.time()
    deadline_all = t0 + budget
    tr = d.train_rows
    te = d.test_rows
    Xtr_all, Ytr_all, Xte = d.X[tr], d.Y[tr], d.X[te]
    n = len(tr)
    nf = int(max(2, min(CONFIG["folds"], n // 10)))
    rng = np.random.default_rng(CONFIG["seed"])
    fold = rng.permutation(n) % max(nf, 1)
    ctx = Ctx()
    ctx.K, ctx.cards, ctx.seed = d.K, d.cards, CONFIG["seed"]
    ctx.device = _device() if _HAS_TORCH else None
    ctx.deadline = None
    ctx.thermo = list(getattr(d, "thermo", []) or [])
    ctx.pairs = []
    try:
        ctx.pairs = _choose_pairs(Xtr_all, Ytr_all, d.cards, d.K, CONFIG["lr_pairs"])
    except Exception:
        ctx.pairs = []
    J = len(d.K)
    oof = {}
    test = {}

    def run_cv(name_fn, fn, multi=False, refit_full=True, base=None, until=None):
        """Cross-fit `fn`; fold models' predictions on test are averaged unless refit.

        `base` names an already cross-fitted model whose predictions for the
        evaluation rows are handed to `fn` (out-of-fold rows get that model's
        out-of-fold predictions, test rows its test predictions)."""
        o = {}
        tp = {}
        t_cv = time.time()
        for f in range(nf):
            if f == 1:   # after one fold we know the cost: give up early if it cannot finish
                per = time.time() - t_cv
                if time.time() + per * (nf - 1 + (1 if refit_full else 0)) > (until or deadline_all):
                    raise TimeoutError("budget (projected)")
            a = fold != f
            b = fold == f
            Xev = Xtr_all[b] if refit_full else np.concatenate([Xtr_all[b], Xte], axis=0)
            bev = None
            if base is not None:
                bev = [oof[base][j][b] if refit_full else
                       np.concatenate([oof[base][j][b], test[base][j]], axis=0) for j in range(J)]
            res = fn(Xtr_all[a], Ytr_all[a], Xev, ctx, bev) if base is not None else fn(Xtr_all[a], Ytr_all[a], Xev, ctx)
            if not multi:
                res = {name_fn: res}
            for nm, preds in res.items():
                if nm not in o:
                    o[nm] = [np.zeros((n, k), dtype=np.float32) for k in d.K]
                    tp[nm] = [np.zeros((len(te), k), dtype=np.float32) for k in d.K]
                nb = int(b.sum())
                for j in range(J):
                    o[nm][j][b] = preds[j][:nb]
                    if not refit_full:
                        tp[nm][j] += preds[j][nb:] / nf
            if time.time() > deadline_all:
                raise TimeoutError("budget")
        if refit_full:
            if base is not None:
                res = fn(Xtr_all, Ytr_all, Xte, ctx, [test[base][j] for j in range(J)])
            else:
                res = fn(Xtr_all, Ytr_all, Xte, ctx)
            if not multi:
                res = {name_fn: res}
            for nm, preds in res.items():
                tp[nm] = [np.asarray(p, dtype=np.float32) for p in preds]
        for nm in o:
            oof[nm] = o[nm]
            test[nm] = tp[nm]

    def best_of(prefix):
        best, bv = None, -1e18
        for nm in oof:
            if not nm.startswith(prefix) or (prefix in ("lr", "lrt") and not nm[len(prefix):][:1].isdigit()):
                continue
            tot = 0.0
            for j in range(J):
                y = Ytr_all[:, j]
                ok = y >= 0
                p = _normalize(oof[nm][j])[ok][np.arange(ok.sum()), y[ok]]
                tot += np.log(_floor_mix(p, d.K[j])).sum()
            if tot > bv:
                best, bv = nm, tot
        return best

    plan = [m for m in CONFIG["models"]]
    # marginal is cheap and the fallback of everything
    run_cv("marg", model_marg)
    _log("marg done", time.time() - t0)
    for m in plan:
        if m == "marg":
            continue
        left = deadline_all - time.time()
        if left <= 1.0:
            break
        try:
            ts = time.time()
            if m == "bo":
                for a in CONFIG["bo_alphas"]:
                    run_cv("bo%g" % a, _make_bo(a, CONFIG["bo_depth"]))
            elif m in ("lr", "lrt") and _HAS_TORCH:
                ctx.hard_deadline = deadline_all
                try:
                    if m == "lr":
                        run_cv("lr", model_lr_all, multi=True)
                    else:
                        run_cv("lrt", lambda a1, a2, a3, a4: model_lr_all(a1, a2, a3, a4, thermo=True),
                               multi=True)
                finally:
                    ctx.hard_deadline = None
            elif m == "bol":
                bname = best_of(CONFIG.get("lr_base", "lr")) or best_of("lr") or "marg"
                for a in CONFIG["bol_alphas"]:
                    run_cv("bol%g" % a, _make_bo(a, CONFIG["bo_depth"]), base=bname)
            elif m == "hb":
                bname = best_of(CONFIG.get("lr_base", "lr")) or best_of("lr") or "marg"
                for prm in CONFIG["hb_params"]:
                    l1, l2_, al = (prm[0], 0.0, prm[1]) if len(prm) == 2 else tuple(prm[:3])
                    if l2_ > 0 and deadline_all - time.time() < 0.5 * (deadline_all - t0):
                        continue
                    rad = 2 if l2_ > 0 else 1
                    run_cv("hb%g_%g_%g" % (l1, l2_, al), _make_hb(l1, al, rad, l2_), base=bname)
                # near-duplicate donors: hot-decks that trust Hamming-1/2 donors
                # strongly.  They only enter the stack when their out-of-fold
                # score beats every standard hot-deck, i.e. when the visible rows
                # hold near-copies of respondents (e.g. the same household in two
                # rounds); otherwise they would only add noise to the stack.
                strong = []
                for prm in CONFIG.get("hb_strong_params") or []:
                    if deadline_all - time.time() < 0.4 * (deadline_all - t0):
                        break
                    l1, l2_, al = (prm[0], 0.0, prm[1]) if len(prm) == 2 else tuple(prm[:3])
                    rad = 2 if l2_ > 0 else 1
                    nm_s = "hbs%g_%g_%g" % (l1, l2_, al)
                    run_cv(nm_s, _make_hb(l1, al, rad, l2_), base=bname)
                    strong.append(nm_s)
                if strong:
                    # complementarity test: how much does a per-item linear pool of
                    # {LR, best standard hot-deck} gain out of fold when the strong
                    # hot-decks join it?  (In-sample over-fit of two extra weights
                    # is ~1e-4 nats per cell; near-copies give ~1e-2.)
                    gmt = getattr(d, "gate_masks", None) or {}
                    std = [nm_ for nm_ in oof if nm_.startswith("hb") and not nm_.startswith("hbs")]
                    std_best = None
                    if std:
                        sc_ = {nm_: 0.0 for nm_ in std}
                        for j_ in range(J):
                            y_ = Ytr_all[:, j_]
                            ok_ = y_ >= 0
                            for nm_ in std:
                                P_ = oof[nm_][j_]
                                if j_ in gmt:
                                    P_ = _gate_apply(P_, gmt[j_][tr], d.K[j_] - 1)
                                p_ = _normalize(P_)[ok_][np.arange(ok_.sum()), y_[ok_]]
                                sc_[nm_] += np.log(_floor_mix(p_, d.K[j_])).sum()
                        std_best = max(sc_, key=sc_.get)
                    base_set = [x for x in (bname, std_best) if x]
                    gain, ncell = 0.0, 0
                    for j_ in range(J):
                        y_ = Ytr_all[:, j_]
                        ok_ = y_ >= 0
                        if ok_.sum() == 0:
                            continue
                        Ps = []
                        for nm_ in base_set + [x for x in strong if x in oof]:
                            P_ = oof[nm_][j_]
                            if j_ in gmt:
                                P_ = _gate_apply(P_, gmt[j_][tr], d.K[j_] - 1)
                            Ps.append(P_)
                        nb = len(base_set)
                        _, ll_a = _stack_weights(Ps[:nb], y_, d.K[j_], np.ones(nb), CONFIG["stack_prior"])
                        _, ll_b = _stack_weights(Ps, y_, d.K[j_], np.ones(len(Ps)), CONFIG["stack_prior"])
                        gain += (ll_b - ll_a) * ok_.sum()
                        ncell += ok_.sum()
                    gain /= max(1, ncell)
                    _log("strong hot-deck pool gain %.5f nats/cell" % gain)
                    if not gain > float(CONFIG.get("hb_strong_min_gain", 0.002)):
                        for nm_ in strong:
                            oof.pop(nm_, None)
                            test.pop(nm_, None)
            elif m == "hbw":
                run_cv("hbw", _make_hbw())
                _log("hbw lambda", np.round(getattr(ctx, "hbw_params", (None,))[0], 2),
                     "alpha", getattr(ctx, "hbw_params", (None, None))[1])
            elif m in ("mlp", "mlpx") and _HAS_TORCH:
                variants = CONFIG.get("mlp_variants") or [{"name": ""}]
                chosen = list(enumerate(variants))
                if "mlpx" in plan:   # "mlp" runs the first variant early, "mlpx" the rest
                    chosen = chosen[:1] if m == "mlp" else chosen[1:]
                elif m == "mlpx":
                    chosen = []
                # time kept back for the boosting stage that runs after the MLPs
                res_s = 0.0
                if m == "mlpx" and any(x in plan for x in ("gbdt", "gbdtf")):   # never starve the base MLP
                    res_s = float(_phase_cfg("gbdt_reserve", 0.0)) * (deadline_all - t0)
                for vi, var in chosen:
                    if ctx.device.type == "cpu" and vi > 0 and not CONFIG.get("full_on_cpu"):
                        continue
                    left = deadline_all - res_s - time.time()
                    if left < 5.0:
                        break
                    cfg = {k: v for k, v in var.items() if k != "name"}
                    cfg.setdefault("mlp_seed_offset", 1000 * vi)
                    if cfg.get("mlp_mix") == "auto":
                        fp_ = getattr(d, "fp", "other")
                        big = fp_ == "unhcr" or (fp_ not in ("unicef", "world_bank") and
                                                 J >= int(CONFIG.get("mix_big_min_items", 150)))
                        cfg["mlp_mix"] = int(CONFIG["mix_classes_big"] if big else CONFIG["mix_classes"])
                        _log("mixture model: %d latent classes" % cfg["mlp_mix"])
                    if ctx.device.type == "cpu" and not CONFIG.get("full_on_cpu"):
                        cfg["mlp_seeds"] = 1
                    ctx.deadline = time.time() + left * 0.8
                    ctx.fold_budget = left * 0.75 / max(1, nf)
                    nm = "mlp" + str(var.get("name", vi))
                    try:
                        run_cv(nm, (lambda c: (lambda a1, a2, a3, a4: model_mlp(a1, a2, a3, a4, c)))(cfg),
                               refit_full=False, until=deadline_all - res_s)
                    except Exception as e:
                        _log("mlp variant", nm, "failed:", type(e).__name__, str(e)[:200])
                    ctx.deadline = None
                    ctx.fold_budget = 0.0
            elif m in ("gbdt", "gbdtf"):
                if m == "gbdtf":   # research: a faster boosting schedule
                    ctx.gbdt_lr, ctx.gbdt_max_iter = 0.15, 100
                else:
                    ctx.gbdt_lr, ctx.gbdt_max_iter = None, None
                bname = best_of("mlp") or best_of("lr") or "marg"
                # rank items by how much loss the best model still leaves, probe
                # the fit time and keep as many items as the budget allows
                loss = []
                for j in range(J):
                    y = Ytr_all[:, j]
                    ok = y >= 0
                    pj = _normalize(oof[bname][j])[ok][np.arange(ok.sum()), y[ok]]
                    loss.append(-np.log(_floor_mix(pj, d.K[j])).sum())
                order = list(np.argsort(loss)[::-1])
                catmask = np.array([c <= 250 for c in d.cards], dtype=bool)
                tp0 = time.time()
                probe = order[:2]
                for j in probe:
                    _gbdt_fit_predict(Xtr_all, Ytr_all[:, j], Xte[:10], d.K[j], catmask,
                                      ctx.gbdt_lr, ctx.gbdt_max_iter)
                per_item = max(1e-3, (time.time() - tp0) / max(1, len(probe)))
                left = deadline_all - time.time()
                m_items = int(0.75 * left / (per_item * nf * 1.1))
                fast = CONFIG.get("gbdt_fast")
                if m == "gbdt" and fast and m_items < J:
                    # not every item fits: probe the cheaper schedule and use it
                    tp1 = time.time()
                    for j in probe:
                        _gbdt_fit_predict(Xtr_all, Ytr_all[:, j], Xte[:10], d.K[j], catmask,
                                          fast[0], fast[1])
                    pf = max(1e-3, (time.time() - tp1) / max(1, len(probe)))
                    left = deadline_all - time.time()
                    mf = int(0.75 * left / (pf * nf * 1.1))
                    if mf > m_items:
                        ctx.gbdt_lr, ctx.gbdt_max_iter = fast[0], int(fast[1])
                        per_item, m_items = pf, mf
                        _log("gbdt: switching to the fast schedule")
                m_items = min(J, m_items, int(CONFIG["gbdt_max_items"]))
                if m == "gbdt" and CONFIG.get("gbdt_extend", True) and m_items < J:
                    # the probe can be off by 2x: start with half and extend with
                    # the measured cost below, rather than lose the whole stage to
                    # the projected-overrun check
                    m_items = max(1, m_items // 2)
                _log("gbdt: %.2fs per item fit, %d of %d items" % (per_item, m_items, J))
                if m_items >= 1:
                    ctx.gbdt_items = order[:m_items]
                    ctx.gbdt_order = order
                    ctx.gbdt_base = bname
                    ctx.deadline = time.time() + left * 0.9
                    t_g = time.time()
                    run_cv(m, model_gbdt, refit_full=False, base=bname)
                    ctx.deadline = None
                    # the probe is noisy: with the measured cost, keep adding the
                    # next items by loss while the instrument's budget lasts
                    done_n = m_items
                    per_meas = max(1e-3, (time.time() - t_g) / max(1, done_n * nf))
                    cap = min(J, int(CONFIG["gbdt_max_items"]))
                    while done_n < cap and m == "gbdt" and CONFIG.get("gbdt_extend", True):
                        left = deadline_all - time.time()
                        more = int(0.8 * left / (per_meas * nf * 1.15))
                        if more < 3:
                            break
                        chunk = order[done_n:done_n + more]
                        ctx.gbdt_items = chunk
                        ctx.deadline = time.time() + left * 0.9
                        try:
                            t_g = time.time()
                            run_cv("gbdt_x", model_gbdt, refit_full=False, base=m)
                        except Exception as e:
                            _log("gbdt extension stopped:", type(e).__name__)
                            oof.pop("gbdt_x", None)
                            test.pop("gbdt_x", None)
                            break
                        finally:
                            ctx.deadline = None
                        for j in chunk:
                            oof[m][j] = oof["gbdt_x"][j]
                            test[m][j] = test["gbdt_x"][j]
                        oof.pop("gbdt_x", None)
                        test.pop("gbdt_x", None)
                        per_meas = max(1e-3, (time.time() - t_g) / max(1, len(chunk) * nf))
                        done_n += len(chunk)
                        _log("gbdt extended to %d of %d items" % (done_n, J))
            _log(m, "done in %.1fs" % (time.time() - ts))
        except Exception as e:  # drop the model
            ctx.deadline = None
            for nm in list(oof):
                if nm.startswith(m) and (nm not in test or len(test[nm]) != J):
                    oof.pop(nm, None)
                    test.pop(nm, None)
            _log("model", m, "failed:", type(e).__name__, str(e)[:200])
    _STATE["last_ctx"] = ctx
    return oof, test, Ytr_all


def _loglinear_fit(logP, yy, k, w0, l2, iters):
    """Geometric pool q ∝ exp(sum_m w_m log p_m), w >= 0, fitted on the floored log score."""
    from scipy.optimize import minimize
    n = len(yy)
    M = logP.shape[0]
    ar = np.arange(n)
    onehot = np.zeros((n, k))
    onehot[ar, yy] = 1.0
    c = 1.0 - k * FLOOR

    def f(w):
        z = np.tensordot(w, logP, 1)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        q = e / e.sum(1, keepdims=True)
        qy = q[ar, yy]
        py = FLOOR + c * qy
        loss = -np.log(py).mean() + l2 * ((w - w0) ** 2).sum()
        G = -(c * qy / py)[:, None] * (onehot - q)
        g = np.einsum("mnk,nk->m", logP, G) / n + 2 * l2 * (w - w0)
        return loss, g
    r = minimize(f, w0.copy(), jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * M,
                 options={"maxiter": iters})
    w = r.x if np.all(np.isfinite(r.x)) else w0
    return w, -float(r.fun)


def _stack(d, oof, test, Ytr):
    names = [nm for nm in oof if nm in test]
    J = len(d.K)
    final = []
    report = {}
    prior = np.ones(len(names))
    method = CONFIG.get("stack_method", "linear")
    for j in range(J):
        y = Ytr[:, j]
        k = d.K[j]
        try:
            w, ll = _stack_weights([oof[nm][j] for nm in names], y, k, prior,
                                   CONFIG["stack_prior"])
        except Exception:
            w = np.zeros(len(names))
            w[names.index("marg")] = 1.0
            ll = float("nan")
        p = _normalize(sum(w[m] * _normalize(test[names[m]][j]) for m in range(len(names))))
        if method in ("loglinear", "both") and time.time() < _STATE.get("stack_deadline", 1e18):
            try:
                ok = y >= 0
                mx = int(CONFIG.get("stack_max_rows", 6000))
                if ok.sum() > mx:
                    idx_ok = np.nonzero(ok)[0]
                    sub = np.random.default_rng(j).choice(idx_ok, mx, replace=False)
                    ok = np.zeros_like(ok)
                    ok[sub] = True
                Po = [_normalize(oof[nm][j][ok]) for nm in names]
                Pt = [_normalize(test[nm][j]) for nm in names]
                if method == "both":
                    Po.append(_normalize(sum(w[m] * Po[m] for m in range(len(names)))))
                    Pt.append(p)
                logPo = np.log(np.clip(np.stack(Po, 0), 1e-12, None))
                M = len(Po)
                w0 = np.full(M, 1.0 / M) if method == "loglinear" else np.eye(M)[M - 1]
                wl, ll2 = _loglinear_fit(logPo, y[ok], k, w0, CONFIG["stack_l2"], CONFIG["stack_iters"])
                logPt = np.log(np.clip(np.stack(Pt, 0), 1e-12, None))
                z = np.tensordot(wl, logPt, 1)
                z -= z.max(1, keepdims=True)
                q = np.exp(z)
                q = q / q.sum(1, keepdims=True)
                if np.all(np.isfinite(q)) and ll2 >= ll - 1e-9:
                    p, ll = q, ll2
            except Exception:
                pass
        final.append(_normalize(p))
        report[d.targets[j]] = (ll, dict(zip(names, np.round(w, 3))))
    return final, report, names


def _exact_gate_masks(d, frame):
    """{item index: gate-open mask over all rows} for GIVEN-parent gates that the
    visible rows satisfy exactly (sentinel iff the gate is closed)."""
    masks = {}
    tr = d.train_rows
    for j, t in enumerate(d.targets):
        if t not in d.gate:
            continue
        parent, obs = d.gate[t]
        if parent not in d.given:
            continue
        sent = d.K[j] - 1
        _, parr = _isna_array(frame[parent])
        pv = np.array([str(v) for v in parr], dtype=object)
        open_ = np.isin(pv, np.array([str(o) for o in obs], dtype=object))
        y = d.Y[tr, j]
        ok = y >= 0
        if ok.sum() < 30:
            continue
        if not np.all((y[ok] == sent) == ~open_[tr][ok]):
            continue
        masks[j] = open_
    return masks


def _gate_apply(P, open_rows, sent):
    P = _normalize(P).copy()
    P[~open_rows] = 0.0
    P[~open_rows, sent] = 1.0
    Q = P[open_rows]
    Q[:, sent] = 0.0
    P[open_rows] = _normalize(Q)
    return P


def _enforce_gates(d, final, masks):
    for j, m in masks.items():
        final[j] = _gate_apply(final[j], m[d.test_rows], d.K[j] - 1)
    return final


def _gate_all_models(d, oof, test, masks):
    for j, m in masks.items():
        sent = d.K[j] - 1
        for nm in list(oof):
            oof[nm][j] = _gate_apply(oof[nm][j], m[d.train_rows], sent)
            if nm in test:
                test[nm][j] = _gate_apply(test[nm][j], m[d.test_rows], sent)


def _gate_consistency(d, final, lam):
    """For PREDICT-parent gates that hold exactly on the visible rows, the child's
    sentinel probability and the parent's 'gate closed' probability estimate the
    same event; blend them (parents first, so chains propagate)."""
    pos = {t: j for j, t in enumerate(d.targets)}
    tr = d.train_rows

    def depth(t, seen=()):
        g = d.gate.get(t)
        if not g or g[0] not in pos or t in seen:
            return 0
        return 1 + depth(g[0], seen + (t,))
    order = sorted(range(len(d.targets)), key=lambda j: depth(d.targets[j]))
    for j in order:
        t = d.targets[j]
        g = d.gate.get(t)
        if not g or g[0] not in pos:
            continue
        jp = pos[g[0]]
        opts_p = d.options[g[0]]
        obs = set(str(o) for o in g[1])
        closed_idx = [i for i, o in enumerate(opts_p) if str(o) not in obs]
        if not closed_idx:
            continue
        sent = d.K[j] - 1
        yc = d.Y[tr, j]
        yp = d.Y[tr, jp]
        ok = (yc >= 0) & (yp >= 0)
        if ok.sum() < 30:
            continue
        if not np.all((yc[ok] == sent) == np.isin(yp[ok], closed_idx)):
            continue
        closed = final[jp][:, closed_idx].sum(1)
        P = _normalize(final[j]).copy()
        s_new = (1.0 - lam) * P[:, sent] + lam * np.clip(closed, 0.0, 1.0)
        rest = P[:, :sent]
        rs = rest.sum(1, keepdims=True)
        rest = np.where(rs > 0, rest / np.where(rs > 0, rs, 1.0), 1.0 / max(1, sent))
        P[:, :sent] = rest * (1.0 - s_new)[:, None]
        P[:, sent] = s_new
        final[j] = _normalize(P)
    return final


def _assemble(frame, schema, d, final, cells):
    """Vectors in canonical order; anything we cannot fill gets the marginal/uniform."""
    pos = {t: j for j, t in enumerate(d.targets)}
    te_index = {r: i for i, r in enumerate(d.test_rows.tolist())}
    marg = {}
    out = []
    for (r, item) in cells:
        rec = schema["items"][item]
        width = len(_options(schema, rec))
        v = None
        j = pos.get(item)
        if j is not None and r in te_index:
            v = final[j][te_index[r]]
        if v is None or len(v) != width:
            if item not in marg:
                marg[item] = _fallback_marginal(frame, schema, item)
            v = marg[item]
        out.append(v)
    return out


def _fallback_marginal(frame, schema, item):
    rec = schema["items"][item]
    opts = _options(schema, rec)
    try:
        na, arr = _isna_array(frame[item])
        code = _encode(arr[~na], opts)
        code = code[code >= 0]
        c = np.bincount(code, minlength=len(opts)).astype(np.float64)[:len(opts)] + 0.5
        return c / c.sum()
    except Exception:
        return np.full(len(opts), 1.0 / max(1, len(opts)))


def _fallback_gate_ok(frame, schema, item):
    """True when a GIVEN-parent gate holds exactly on the visible rows."""
    try:
        rec = schema["items"][item]
        g = rec.get("gate")
        if not g or schema["items"].get(g.get("parent"), {}).get("class") != "GIVEN":
            return False
        gv = schema.get("gated_value", "NA_GATED")
        na, arr = _isna_array(frame[item])
        _, parr = _isna_array(frame[g["parent"]])
        obs = set(str(o) for o in g.get("observed_if") or [])
        vis = ~na
        if vis.sum() < 30:
            return False
        is_sent = np.array([str(v) == gv for v in arr[vis]])
        closed = np.array([str(v) not in obs for v in parr[vis]])
        return bool(np.all(is_sent == closed))
    except Exception:
        return False


def _fallback_predict(frame, schema, cells):
    """Gate-aware marginals; cannot fail except on a broken schema."""
    out = []
    cache = {}
    gate_ok = {}
    for (r, item) in cells:
        rec = schema["items"][item]
        opts = _options(schema, rec)
        if item not in cache:
            cache[item] = _fallback_marginal(frame, schema, item)
            gate_ok[item] = _fallback_gate_ok(frame, schema, item)
        v = cache[item]
        if gate_ok[item]:
            try:
                g = rec.get("gate")
                pv = frame[g["parent"]].iloc[r]
                if not _is_missing(pv):
                    if str(pv) not in [str(o) for o in g.get("observed_if") or []]:
                        v = np.zeros(len(opts))
                        v[-1] = 1.0
                    else:
                        v = v.copy()
                        v[-1] = 0.0
                        v = v / v.sum() if v.sum() > 0 else np.full(len(opts), 1.0 / len(opts))
            except Exception:
                v = cache[item]
        out.append(v)
    return out


def _validate(vectors, cells, schema):
    out = []
    for v, (r, item) in zip(vectors, cells):
        width = len(_options(schema, schema["items"][item]))
        try:
            a = np.asarray(v, dtype=np.float64).reshape(-1)
            if a.shape[0] != width or not np.all(np.isfinite(a)) or np.any(a < 0) or not a.sum() > 0:
                raise ValueError
            a = a / a.sum()
        except Exception:
            a = np.full(width, 1.0 / max(1, width))
        out.append([float(x) for x in a])
    return out


def _predict_full(frame, schema, cells):
    d = _prepare(frame, schema)
    if not d.targets or len(d.test_rows) == 0:
        return _fallback_predict(frame, schema, cells)
    budget = _budget_for(d, schema, frame)
    saved = None
    if PHASE2_EXTRA and (CONFIG.get("force_phase2_extra") or
                         (_STATE.get("phase_total", 900.0) >= 3600.0 and not CONFIG.get("budget_override"))):
        saved = dict(CONFIG)
        CONFIG.update(PHASE2_EXTRA)
    try:
        return _predict_core(frame, schema, cells, d, budget)
    finally:
        if saved is not None:
            CONFIG.clear()
            CONFIG.update(saved)


def _predict_core(frame, schema, cells, d, budget):
    t0 = time.time()
    _log("instrument", _fingerprint(schema), "N=%d train=%d test=%d J=%d budget=%.0fs"
         % (d.N, len(d.train_rows), len(d.test_rows), len(d.targets), budget))
    if len(d.train_rows) < 30:
        return _fallback_predict(frame, schema, cells)
    hard_left = 0.85 * _STATE.get("phase_total", 900.0) - (time.time() - _T_IMPORT)
    if hard_left < 45.0 and not CONFIG.get("budget_override"):
        _log("hard stop: falling back to gate-aware marginals")
        return _fallback_predict(frame, schema, cells)
    if not CONFIG.get("budget_override"):
        budget = min(budget, max(20.0, hard_left - 30.0))
    try:
        d.gate_masks = _exact_gate_masks(d, frame) if CONFIG.get("enforce_gates") else {}
    except Exception:
        d.gate_masks = {}
    oof, test, Ytr = _run_models(d, budget * 0.8)
    _STATE["stack_deadline"] = time.time() + max(10.0, budget * 0.2)
    if CONFIG.get("ordinal_smooth"):
        try:
            bases = [nm for nm in (CONFIG.get("ordinal_bases") or []) if nm in oof] or \
                [nm for nm in list(oof) if nm.startswith("mlp")][:2]
            _add_ordinal_variants(d, oof, test, bases)
        except Exception:
            pass
    masks = {}
    if CONFIG.get("enforce_gates"):
        try:
            masks = _exact_gate_masks(d, frame)
            _gate_all_models(d, oof, test, masks)
        except Exception:
            masks = {}
    ts = time.time()
    try:
        final, report, names = _stack(d, oof, test, Ytr)
    except Exception as e:
        # stacking failed: use the single model with the best out-of-fold score
        _log("stacking failed:", type(e).__name__, str(e)[:200])
        best, bv = "marg", -1e18
        for nm in oof:
            if nm not in test:
                continue
            tot = 0.0
            for j in range(len(d.K)):
                y = Ytr[:, j]
                ok = y >= 0
                pj = _normalize(oof[nm][j])[ok][np.arange(ok.sum()), y[ok]]
                tot += np.log(_floor_mix(pj, d.K[j])).sum()
            if tot > bv:
                best, bv = nm, tot
        final = [_normalize(test[best][j]) for j in range(len(d.K))]
        report, names = {t: (float("nan"), {best: 1.0}) for t in d.targets}, [best]
    _log("stacking took %.1fs" % (time.time() - ts))
    if CONFIG.get("verbose"):
        tot = {}
        for nm in oof:
            lls = []
            for j in range(len(d.K)):
                y = Ytr[:, j]
                ok = y >= 0
                p = _floor_mix(_normalize(oof[nm][j])[ok][np.arange(ok.sum()), y[ok]], d.K[j])
                lls.append(np.log(p).sum())
            tot[nm] = sum(lls) / max(1, (Ytr >= 0).sum())
        _log("OOF mean log score by model:", {k: round(v, 5) for k, v in tot.items()})
        stk = np.sum([report[t][0] * (Ytr[:, j] >= 0).sum() for j, t in enumerate(d.targets)]) / max(1, (Ytr >= 0).sum())
        _log("stacked OOF log score %.5f" % stk)
        try:
            avg = {}
            for t, (ll_, wd) in report.items():
                for k_, v_ in wd.items():
                    avg[k_] = avg.get(k_, 0.0) + float(v_) / max(1, len(report))
            _log("mean linear-pool weights:", {k_: round(v_, 3) for k_, v_ in avg.items()})
        except Exception:
            pass
    if masks:
        try:
            final = _enforce_gates(d, final, masks)
        except Exception:
            pass
    if CONFIG.get("gate_consistency", 0) > 0:
        try:
            final = _gate_consistency(d, final, float(CONFIG["gate_consistency"]))
        except Exception:
            pass
    _log("instrument done in %.1fs" % (time.time() - t0))
    return _assemble(frame, schema, d, final, cells)


def predict(frame, schema):
    t0 = time.time()
    if CONFIG.get("threads") and _HAS_TORCH:
        try:
            torch.set_num_threads(int(CONFIG["threads"]))
        except Exception:
            pass
    try:
        cells, _ = _blank_cells(frame, schema)
    except Exception:
        # last resort: mimic the reference baseline's cell walk
        items = [n for n, r in schema["items"].items() if r["class"] in ("GIVEN", "PREDICT")]
        vals = frame[items].to_numpy(dtype=object)
        cells = [(r, items[c]) for r in range(vals.shape[0]) for c in range(len(items))
                 if pd.isna(vals[r, c])]
    try:
        vectors = _predict_full(frame, schema, cells)
    except Exception as e:
        _log("full pipeline failed:", type(e).__name__, str(e)[:300])
        try:
            vectors = _fallback_predict(frame, schema, cells)
        except Exception:
            vectors = [None] * len(cells)
    out = _validate(vectors, cells, schema)
    try:
        _STATE["done"].append(_fingerprint(schema))
        _STATE["spent"] += time.time() - t0
    except Exception:
        pass
    return out
