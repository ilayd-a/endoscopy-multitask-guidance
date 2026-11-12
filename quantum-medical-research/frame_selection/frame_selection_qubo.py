"""
frame_selection_qubo.py
Fine-tuning frame selection for the pretrained Kvasir-SEG UNet, as a QUBO.

Objective (cardinality K):
    maximise  alpha * sum_i d_i x_i + beta * sum_i c_i x_i - lam * sum_{i<j} S_ij x_i x_j
    s.t.      sum_i x_i = K
d = difficulty (1 - baseline Dice), c = coverage (mean cosine similarity to all
train embeddings), both min-max normalised within the pool; S = pairwise cosine
similarity of UNet encoder embeddings, z-scored over the pool's off-diagonal
entries, zero diagonal.

Layer B (objective question, N=150, K=40): random, hardest-K, QUBO+greedy and
QUBO+simulated annealing subsets each fine-tune the same checkpoint for the same
number of steps; evaluated on val+test (all frames and baseline-hard frames).

Layer C (weight question): sweeps beta, lambda and the d/c normalisation;
eval frames are split in half (stratified by baseline-hard), the config is picked
on one half and it and hardest-K are re-run with more seeds and reported on the
other half, against the random-subset distribution from subset_variance_test.py.

Layer A (solver question, N=20, K=6): exhaustive optimum, greedy, simulated
annealing and QAOA. QAOA is classically simulated in numpy inside the
Hamming-weight-K subspace (Dicke initial state + complete-graph XY mixer); it is
not a QPU run.

Usage (from repo root):
    S=quantum-medical-research/frame_selection/frame_selection_qubo.py
    ./venv/bin/python $S --layer B --methods all
    ./venv/bin/python $S --layer C
    ./venv/bin/python $S --layer A --pool 20 --k 6
"""

import argparse
import itertools
import json
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityRanged
from scipy import stats
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "dataset" / "data"
CKPT = REPO / "models" / "unet_pretrained.pth"
OUT_DIR = REPO / "quantum-medical-research" / "frame_selection" / "results"
SIZE = 256
HARD_THRESHOLD = 0.80
LAYER_B_METHODS = ["no_finetune", "random_k", "hardest_k", "qubo_greedy", "qubo_sa", "pool_all"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data / model (same loading pipeline as models/train_pretrained.py)
# ---------------------------------------------------------------------------
load_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["mask"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image", "mask"], spatial_size=(SIZE, SIZE)),
])


def load_split(split):
    img_dir = DATA_DIR / split / "images"
    names = sorted(f.name for f in img_dir.iterdir() if f.name != ".DS_Store")
    images = np.zeros((len(names), 3, SIZE, SIZE), dtype=np.float32)
    masks = np.zeros((len(names), 1, SIZE, SIZE), dtype=np.float32)
    for i, name in enumerate(names):
        data = load_transforms({"image": str(img_dir / name), "mask": str(DATA_DIR / split / "masks" / name)})
        mask = np.asarray(data["mask"])
        if mask.shape[0] == 3:
            mask = mask.mean(axis=0, keepdims=True)
        images[i] = np.asarray(data["image"])
        masks[i] = (mask > 0.5).astype(np.float32)
    return names, images, masks


def load_model(device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return model.to(device)


@torch.no_grad()
def predict_dice(model, images, masks, device, batch=16):
    model.eval()
    dices = []
    for s in range(0, len(images), batch):
        x = torch.from_numpy(images[s:s + batch]).to(device)
        y = torch.from_numpy(masks[s:s + batch]).to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dices.append(((2 * inter + 1e-6) / (denom + 1e-6)).cpu().numpy())
    return np.concatenate(dices)


@torch.no_grad()
def encoder_embeddings(model, images, device, batch=16):
    model.eval()
    feats = []
    for s in range(0, len(images), batch):
        x = torch.from_numpy(images[s:s + batch]).to(device)
        feats.append(model.encoder(x)[-1].mean(dim=(2, 3)).cpu().numpy())
    emb = np.concatenate(feats)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def load_features(device, need_eval_data):
    """Train baseline Dice, train embeddings, eval baseline Dice (cached)."""
    cache = OUT_DIR / f"features_{SIZE}.npz"
    data = {}
    if need_eval_data or not cache.exists():
        tr_names, tr_img, tr_mask = load_split("train")
        va_names, va_img, va_mask = load_split("val")
        te_names, te_img, te_mask = load_split("test")
        data = {
            "tr_img": tr_img, "tr_mask": tr_mask,
            "ev_img": np.concatenate([va_img, te_img]), "ev_mask": np.concatenate([va_mask, te_mask]),
            "ev_names": [f"val/{n}" for n in va_names] + [f"test/{n}" for n in te_names],
        }
    if cache.exists():
        z = np.load(cache)
        feats = {k: z[k] for k in ("train_dice", "emb", "ev_base")}
    else:
        base = load_model(device)
        feats = {
            "train_dice": predict_dice(base, data["tr_img"], data["tr_mask"], device),
            "emb": encoder_embeddings(base, data["tr_img"], device),
            "ev_base": predict_dice(base, data["ev_img"], data["ev_mask"], device),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **feats)
    return feats, data


# ---------------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------------
def build_pool(train_dice, n_pool, n_strata, seed, candidates=None):
    """Difficulty-stratified pool so the hardest frames do not fill it."""
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(train_dice)) if candidates is None else np.asarray(candidates)
    order = candidates[np.argsort(train_dice[candidates], kind="stable")]
    per = n_pool // n_strata
    pool = [rng.choice(stratum, per, replace=False) for stratum in np.array_split(order, n_strata)]
    return np.sort(np.concatenate(pool))


def minmax(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def qubo_terms(pool, train_dice, emb, norm="minmax"):
    scale = minmax if norm == "minmax" else zscore
    d = scale(1.0 - train_dice[pool])
    c = scale((emb[pool] @ emb.T).mean(axis=1))
    S = emb[pool] @ emb[pool].T
    off = ~np.eye(len(pool), dtype=bool)
    S = (S - S[off].mean()) / S[off].std()
    np.fill_diagonal(S, 0.0)
    return d, c, S


class Qubo:
    def __init__(self, d, c, S, alpha, beta, lam):
        self.d, self.c, self.S = d, c, S
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.lin = alpha * d + beta * c
        self.Q = lam * S

    def objective(self, x):
        """lin·x - 1/2 x^T Q x == sum lin - sum_{i<j} Q_ij (Q symmetric, zero diagonal)."""
        return float(self.lin @ x - 0.5 * x @ self.Q @ x)

    def terms(self, x):
        return {
            "difficulty_term": float(self.alpha * self.d @ x),
            "coverage_term": float(self.beta * self.c @ x),
            "redundancy_term": float(-self.lam * 0.5 * x @ self.S @ x),
        }

    def scale_check(self, k, n_draws, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_draws):
            x = np.zeros(len(self.lin))
            x[rng.choice(len(x), k, replace=False)] = 1
            rows.append(self.terms(x) | {"objective": self.objective(x)})
        df = pd.DataFrame(rows)
        return {col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))} for col in df}


def greedy_select(qubo, k):
    chosen = np.zeros(len(qubo.lin))
    gain = qubo.lin.copy()
    for _ in range(k):
        i = int(np.argmax(np.where(chosen > 0, -np.inf, gain)))
        chosen[i] = 1
        gain = gain - qubo.Q[i]
    return chosen


def anneal_select(qubo, k, seed, n_evals, t1=1e-3):
    """Swap-move simulated annealing on the exact-cardinality feasible set.

    Each proposed swap counts as one objective evaluation. Returns the best
    subset found and its objective.
    """
    lin, Q = qubo.lin, qubo.Q
    rng = np.random.default_rng(seed)
    n = len(lin)
    x = np.zeros(n)
    x[rng.choice(n, k, replace=False)] = 1
    sel, uns = list(np.flatnonzero(x > 0)), list(np.flatnonzero(x == 0))
    field = lin - Q @ x  # marginal value of each frame given the current subset
    cur = qubo.objective(x)
    best_x, best = x.copy(), cur
    t0 = float(np.std(lin)) + 1e-3
    a_idx = rng.integers(0, k, n_evals)
    b_idx = rng.integers(0, n - k, n_evals)
    u = rng.random(n_evals)
    for step in range(n_evals):
        t = t0 * (t1 / t0) ** (step / max(n_evals - 1, 1))
        i, j = sel[a_idx[step]], uns[b_idx[step]]
        delta = -field[i] + field[j] + Q[i, j]
        if delta >= 0 or u[step] < np.exp(delta / t):
            x[i], x[j] = 0, 1
            sel[a_idx[step]], uns[b_idx[step]] = j, i
            field += Q[:, i] - Q[:, j]
            cur += delta
            if cur > best + 1e-12:
                best, best_x = cur, x.copy()
    return best_x, best


# ---------------------------------------------------------------------------
# Layer B: fine-tuning
# ---------------------------------------------------------------------------
def finetune(subset_imgs, subset_masks, device, seed, steps, lr, batch):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = load_model(device)
    model.train()
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(subset_imgs)
    for _ in range(steps):
        idx = rng.choice(n, min(batch, n), replace=False)
        x = torch.from_numpy(subset_imgs[idx]).to(device)
        y = torch.from_numpy(subset_masks[idx]).to(device)
        if rng.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        if rng.random() < 0.5:
            x, y = x.flip(-2), y.flip(-2)
        out = model(x)
        loss = dice_loss(out, y) + bce(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def layer_b_selections(qubo, k, seeds, sa_evals):
    n = len(qubo.lin)
    hardest = np.zeros(n)
    hardest[np.argsort(-qubo.d, kind="stable")[:k]] = 1
    sel = {"hardest_k": {s: hardest for s in seeds}, "random_k": {}}
    for s in seeds:
        r = np.zeros(n)
        r[np.random.default_rng(100 + s).choice(n, k, replace=False)] = 1
        sel["random_k"][s] = r
    t = time.time()
    g = greedy_select(qubo, k)
    sel["qubo_greedy"] = {s: g for s in seeds}
    t_greedy = time.time() - t
    t = time.time()
    a, _ = anneal_select(qubo, k, seed=0, n_evals=sa_evals)
    sel["qubo_sa"] = {s: a for s in seeds}
    t_sa = time.time() - t
    sel["pool_all"] = {s: np.ones(n) for s in seeds}
    return sel, hardest, {"qubo_greedy": t_greedy, "qubo_sa": t_sa}


def run_layer_b(args, device):
    out = OUT_DIR / "layer_b"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    methods = LAYER_B_METHODS if "all" in args.methods else args.methods
    need_training = any(m != "no_finetune" for m in methods)
    feats, data = load_features(device, need_eval_data=need_training)
    train_dice, emb, ev_base = feats["train_dice"], feats["emb"], feats["ev_base"]
    hard = ev_base < HARD_THRESHOLD
    print(f"train Dice {train_dice.mean():.4f} | eval Dice {ev_base.mean():.4f} | eval hard {hard.sum()}/{len(hard)}")

    pool = build_pool(train_dice, args.pool, args.strata, args.pool_seed)
    d, c, S = qubo_terms(pool, train_dice, emb)
    lam = args.lam if args.lam is not None else 1.0 / (args.k - 1)
    qubo = Qubo(d, c, S, args.alpha, args.beta, lam)
    sel, hardest, solve_sec = layer_b_selections(qubo, args.k, args.seeds, args.sa_evals)

    scale = qubo.scale_check(args.k, 1000, seed=0)
    print("scale check over 1000 random K-subsets (std across subsets):")
    for term, v in scale.items():
        print(f"  {term:16s} mean={v['mean']:+.3f} std={v['std']:.3f}")
    sel_info = {}
    for name, per_seed in sel.items():
        x = per_seed[args.seeds[0]]
        sel_info[name] = qubo.terms(x) | {
            "objective": qubo.objective(x),
            "mean_train_dice": float(train_dice[pool][x > 0].mean()),
            "overlap_with_hardest": int((x * hardest).sum()),
            "solve_sec": solve_sec.get(name),
            "selected_train_idx": {int(s): [int(i) for i in pool[v > 0]] for s, v in per_seed.items()},
        }
        print(f"  {name:12s} obj={sel_info[name]['objective']:+.3f} "
              f"d={sel_info[name]['difficulty_term']:+.3f} c={sel_info[name]['coverage_term']:+.3f} "
              f"r={sel_info[name]['redundancy_term']:+.3f} trainDice={sel_info[name]['mean_train_dice']:.3f} "
              f"overlapHardest={sel_info[name]['overlap_with_hardest']}")
    with open(out / "selection_info.json", "w") as f:
        json.dump({"args": vars(args) | {"lam": lam}, "n_eval": int(len(hard)), "n_eval_hard": int(hard.sum()),
                   "pool_train_idx": [int(i) for i in pool], "scale_check_random_subsets": scale,
                   "selections": sel_info}, f, indent=1)

    np.save(out / "runs" / "no_finetune.npy", ev_base)
    for m in methods:
        if m == "no_finetune":
            continue
        for s in args.seeds:
            idx = pool[sel[m][s] > 0]
            t = time.time()
            model = finetune(data["tr_img"][idx], data["tr_mask"][idx], device, seed=s,
                             steps=args.steps, lr=args.lr, batch=args.batch)
            dice = predict_dice(model, data["ev_img"], data["ev_mask"], device)
            np.save(out / "runs" / f"{m}_s{s}.npy", dice)
            print(f"{m:12s} seed={s} full={dice.mean():.4f} hard={dice[hard].mean():.4f} ({time.time() - t:.0f}s)")
    if data:
        pd.Series(data["ev_names"], name="frame").to_csv(out / "eval_frames.csv", index=False)
    build_layer_b_table(out, args, ev_base, hard, sel_info)


def build_layer_b_table(out, args, ev_base, hard, sel_info):
    """Summarise whatever runs exist in out/runs."""
    per_frame = {"no_finetune": ev_base}
    rows = [{"method": "no_finetune", "seed": -1, "full_dice": ev_base.mean(), "hard_dice": ev_base[hard].mean()}]
    for m in LAYER_B_METHODS[1:]:
        for s in args.seeds:
            f = out / "runs" / f"{m}_s{s}.npy"
            if f.exists():
                per_frame[f"{m}_s{s}"] = np.load(f)
                rows.append({"method": m, "seed": s, "full_dice": per_frame[f"{m}_s{s}"].mean(),
                             "hard_dice": per_frame[f"{m}_s{s}"][hard].mean()})
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pf = pd.DataFrame(per_frame)
    names_csv = out / "eval_frames.csv"
    if names_csv.exists():
        pf.index = pd.read_csv(names_csv)["frame"]
    pf.assign(hard=hard).to_csv(out / "per_frame_dice.csv")

    def seed_mean(m):
        cols = [f"{m}_s{s}" for s in args.seeds if f"{m}_s{s}" in per_frame]
        return np.mean([per_frame[c] for c in cols], axis=0) if cols else None

    ref = seed_mean("hardest_k")
    rng = np.random.default_rng(0)
    subsets = {"hard": np.flatnonzero(hard), "full": np.arange(len(hard))}
    boot_idx = {k: rng.choice(v, (args.n_boot, len(v))) for k, v in subsets.items()}
    table = []
    for m in LAYER_B_METHODS:
        g = runs[runs.method == m]
        if g.empty:
            continue
        row = {
            "method": m,
            "n_train_frames": 0 if m == "no_finetune" else (args.pool if m == "pool_all" else args.k),
            "n_seeds": len(g) if m != "no_finetune" else 0,
            "full_dice": g.full_dice.mean(), "full_dice_sd": g.full_dice.std(ddof=0),
            "hard_dice": g.hard_dice.mean(), "hard_dice_sd": g.hard_dice.std(ddof=0),
            "gain_full_vs_noft": g.full_dice.mean() - ev_base.mean(),
            "gain_hard_vs_noft": g.hard_dice.mean() - ev_base[hard].mean(),
        }
        if m in sel_info:
            row["qubo_objective"] = sel_info[m]["objective"]
            row["overlap_with_hardest"] = sel_info[m]["overlap_with_hardest"]
        cur = seed_mean(m) if m != "no_finetune" else ev_base
        if ref is not None and m != "hardest_k":
            diff = cur - ref
            for name, ids in subsets.items():
                boots = diff[boot_idx[name]].mean(axis=1)
                row[f"{name}_diff_vs_hardest"] = diff[ids].mean()
                row[f"{name}_diff_vs_hardest_ci_lo"] = np.percentile(boots, 2.5)
                row[f"{name}_diff_vs_hardest_ci_hi"] = np.percentile(boots, 97.5)
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(out / "comparison_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# Layer C: weight sweep (beta, lambda, normalisation) with a held-out report half
# ---------------------------------------------------------------------------
def eval_halves(hard, seed):
    """Split eval frames in half, stratified by baseline-hard: select on one, report on the other."""
    rng = np.random.default_rng(seed)
    select = np.zeros(len(hard), dtype=bool)
    for group in (np.flatnonzero(hard), np.flatnonzero(~hard)):
        select[rng.choice(group, len(group) // 2, replace=False)] = True
    return select, ~select


def sweep_configs(args):
    configs = []
    for norm in args.norms:
        for beta in args.betas:
            for m in args.lam_mults:
                configs.append({"norm": norm, "beta": beta, "lam_mult": m, "lam": m / (args.k - 1)})
    return configs


def run_layer_c(args, device):
    out = OUT_DIR / "layer_c"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    feats, data = load_features(device, need_eval_data=True)
    train_dice, emb, ev_base = feats["train_dice"], feats["emb"], feats["ev_base"]
    hard = ev_base < HARD_THRESHOLD
    sel_half, rep_half = eval_halves(hard, args.split_seed)
    pool = build_pool(train_dice, args.pool, args.strata, args.pool_seed)
    print(f"eval halves: select hard={int((sel_half & hard).sum())}/{int(sel_half.sum())} "
          f"report hard={int((rep_half & hard).sum())}/{int(rep_half.sum())}")

    # one subset per config; identical subsets share fine-tunes
    subsets, rows = {}, []
    for cfg in sweep_configs(args):
        d, c, S = qubo_terms(pool, train_dice, emb, cfg["norm"])
        qubo = Qubo(d, c, S, args.alpha, cfg["beta"], cfg["lam"])
        if cfg["beta"] == 0 and cfg["lam"] == 0:
            x = np.zeros(len(pool))
            x[np.argsort(-d, kind="stable")[:args.k]] = 1
        else:
            x, _ = anneal_select(qubo, args.k, seed=0, n_evals=args.sa_evals)
        idx = tuple(int(i) for i in pool[x > 0])
        key = subsets.setdefault(idx, f"sub{len(subsets):02d}")
        rows.append(cfg | {"subset": key, **qubo.terms(x), "mean_train_dice": float(train_dice[list(idx)].mean())})
    print(f"{len(rows)} configs -> {len(subsets)} distinct subsets")

    def finetune_eval(key, idx, seeds):
        for s in seeds:
            f = out / "runs" / f"{key}_s{s}.npy"
            if f.exists():
                continue
            t = time.time()
            model = finetune(data["tr_img"][list(idx)], data["tr_mask"][list(idx)], device, seed=s,
                             steps=args.steps, lr=args.lr, batch=args.batch)
            dice = predict_dice(model, data["ev_img"], data["ev_mask"], device)
            np.save(f, dice)
            print(f"{key} seed={s} select-half hard={dice[sel_half & hard].mean():.4f} ({time.time() - t:.0f}s)",
                  flush=True)

    def half_scores(key):
        runs = [np.load(f) for f in sorted((out / "runs").glob(f"{key}_s*.npy"))]
        return {f"{half}_{m}": float(np.mean([r[mask & sub].mean() for r in runs]))
                for half, mask in (("select", sel_half), ("report", rep_half))
                for m, sub in (("hard", hard), ("full", np.ones_like(hard)))} | {"n_seeds": len(runs)}

    for idx, key in subsets.items():
        finetune_eval(key, idx, args.seeds)
    table = pd.DataFrame([r | half_scores(r["subset"]) for r in rows])
    hardest_key = table[(table.beta == 0) & (table.lam == 0)].subset.iloc[0]
    best_key = table.sort_values("select_hard", ascending=False).subset.iloc[0]
    print(f"best on select half: {best_key} | hardest-K: {hardest_key}")

    # confirm best config and hardest-K with more seeds, scored on the report half only
    key_to_idx = {k: i for i, k in subsets.items()}
    for key in {best_key, hardest_key}:
        finetune_eval(key, key_to_idx[key], args.confirm_seeds)
    table = pd.DataFrame([r | half_scores(r["subset"]) for r in rows])
    table.to_csv(out / "sweep_table.csv", index=False)

    # reference: random K-subsets from subset_variance_test.py on the same pool, if present
    ref_dir = OUT_DIR / "methodology_test" / "runs"
    ref = {}
    for f in sorted(ref_dir.glob("random_*_s*.npy")):
        ref.setdefault(f.stem.rsplit("_s", 1)[0], []).append(np.load(f))
    ref_scores = {m: np.array([np.mean([r[rep_half & sub].mean() for r in v]) for v in ref.values()])
                  for m, sub in (("hard", hard), ("full", np.ones_like(hard)))}
    summary = {"args": vars(args), "best_subset": best_key, "hardest_subset": hardest_key,
               "best_configs": table[table.subset == best_key][["norm", "beta", "lam_mult"]].to_dict("records"),
               "n_random_reference": len(ref)}
    for key in (best_key, hardest_key):
        runs = {int(f.stem.rsplit("_s", 1)[1]): np.load(f) for f in (out / "runs").glob(f"{key}_s*.npy")}
        rec = {"n_seeds": len(runs)}
        for m, sub in (("hard", hard), ("full", np.ones_like(hard))):
            vals = np.array([runs[s][rep_half & sub].mean() for s in sorted(runs)])
            rec[f"report_{m}"] = float(vals.mean())
            rec[f"report_{m}_seed_sd"] = float(vals.std(ddof=1))
            if len(ref):
                rec[f"report_{m}_pct_vs_random"] = float((ref_scores[m] < vals.mean()).mean() * 100)
        summary[key] = rec
    if best_key != hardest_key:
        common = sorted(set(int(f.stem.rsplit("_s", 1)[1]) for f in (out / "runs").glob(f"{best_key}_s*.npy"))
                        & set(int(f.stem.rsplit("_s", 1)[1]) for f in (out / "runs").glob(f"{hardest_key}_s*.npy")))
        for m, sub in (("hard", hard), ("full", np.ones_like(hard))):
            diff = np.array([np.load(out / "runs" / f"{best_key}_s{s}.npy")[rep_half & sub].mean()
                             - np.load(out / "runs" / f"{hardest_key}_s{s}.npy")[rep_half & sub].mean()
                             for s in common])
            p = float(stats.ttest_1samp(diff, 0).pvalue) if len(diff) > 1 else None
            summary[f"best_minus_hardest_report_{m}"] = {"mean": float(diff.mean()), "n_seeds": len(common), "p": p}
    if len(ref):
        summary["random_reference_report"] = {m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                                              for m, v in ref_scores.items()}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.sort_values("select_hard", ascending=False)
          [["norm", "beta", "lam_mult", "subset", "mean_train_dice", "n_seeds", "select_hard", "select_full",
            "report_hard", "report_full"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


# ---------------------------------------------------------------------------
# Layer A: solver comparison
# ---------------------------------------------------------------------------
def enumerate_subsets(qubo, k):
    n = len(qubo.lin)
    combos = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64)
    X = np.zeros((len(combos), n))
    np.put_along_axis(X, combos, 1.0, axis=1)
    values = X @ qubo.lin - 0.5 * np.einsum("ij,jk,ik->i", X, qubo.Q, X)
    masks = (np.int64(1) << combos).sum(axis=1)
    return combos, masks, values


def xy_mixer(combos, masks, n):
    """Complete-graph XY mixer restricted to the Hamming-weight-K subspace.

    sum_{i<j} (X_i X_j + Y_i Y_j) / 2 maps a subset to every subset obtained by
    swapping one selected frame for one unselected frame, with amplitude 1.
    """
    order = np.argsort(masks)
    sorted_masks = masks[order]
    rows, cols = [], []
    for out_pos in range(combos.shape[1]):
        out_bit = np.int64(1) << combos[:, out_pos]
        for j in range(n):
            in_bit = np.int64(1) << j
            ok = (masks & in_bit) == 0
            nb = (masks[ok] ^ out_bit[ok]) | in_bit
            rows.append(np.flatnonzero(ok))
            cols.append(order[np.searchsorted(sorted_masks, nb)])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(masks), len(masks)))


class JohnsonMixer:
    """e^{-i beta A} psi for the complete-graph XY mixer A on J(n, k).

    A is the Johnson-graph adjacency, whose distinct eigenvalues are
    theta_i = (k - i)(n - k - i) - i, i = 0..min(k, n-k). The spectral projectors
    P_i = prod_{j != i} (A - theta_j) / (theta_i - theta_j) give the exact
    propagator sum_i e^{-i beta theta_i} P_i psi using sparse mat-vecs only.
    Checked against scipy's expm_multiply in run_layer_a.
    """

    def __init__(self, mixer, n, k):
        self.A = mixer
        self.theta = np.array([(k - i) * (n - k - i) - i for i in range(min(k, n - k) + 1)], dtype=float)

    def project(self, psi):
        comps = []
        for i, ti in enumerate(self.theta):
            v = psi
            for j, tj in enumerate(self.theta):
                if j != i:
                    v = (self.A @ v - tj * v) / (ti - tj)
            comps.append(v)
        return comps

    def apply(self, beta, psi):
        return sum(np.exp(-1j * beta * t) * v for t, v in zip(self.theta, self.project(psi)))


def qaoa_state(params, values, mixer, p):
    psi = np.full(len(values), 1 / np.sqrt(len(values)), dtype=complex)  # Dicke state
    for layer in range(p):
        gamma, beta = params[layer], params[p + layer]
        psi = psi * np.exp(1j * gamma * values)  # e^{-i gamma H_C}, H_C = -objective
        psi = mixer.apply(beta, psi)
    return psi


def best_of_m(values, probs, m):
    """Expected max objective over m i.i.d. samples from probs."""
    order = np.argsort(values)
    cdf = np.cumsum(probs[order])
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    return float(values[order] @ (np.clip(cdf, 0, 1) ** m - np.clip(cdf_prev, 0, 1) ** m))


def run_layer_a(args, device):
    out = OUT_DIR / "layer_a"
    out.mkdir(parents=True, exist_ok=True)
    feats, _ = load_features(device, need_eval_data=False)
    train_dice, emb = feats["train_dice"], feats["emb"]
    pool_b = build_pool(train_dice, args.pool_b, args.strata, args.pool_seed)
    pool = build_pool(train_dice, args.pool, args.strata, args.pool_seed + 1, candidates=pool_b)
    d, c, S = qubo_terms(pool, train_dice, emb)
    k = args.k
    lam = args.lam if args.lam is not None else 1.0 / (k - 1)
    qubo = Qubo(d, c, S, args.alpha, args.beta, lam)
    n = len(pool)

    t = time.time()
    combos, masks, values = enumerate_subsets(qubo, k)
    t_exh = time.time() - t
    e_opt, e_rand = float(values.max()), float(values.mean())
    opt_idx = int(values.argmax())
    n_opt = int(np.isclose(values, e_opt).sum())
    print(f"Layer A: N={n} K={k} subsets={len(values)} optimum={e_opt:.4f} (x{n_opt}) random mean={e_rand:.4f}")

    def ratio(e):
        return (e - e_rand) / (e_opt - e_rand)

    scale = qubo.scale_check(k, 1000, seed=0)
    rows = [{"solver": "exhaustive", "budget_M": len(values), "p_opt": 1.0, "expected_objective": e_opt,
             "best_of_M": e_opt, "approx_ratio_expected": 1.0, "approx_ratio_best_of_M": 1.0, "wall_sec": t_exh}]

    t = time.time()
    g = greedy_select(qubo, k)
    t_g = time.time() - t
    e_g = qubo.objective(g)
    greedy_evals = sum(n - i for i in range(k))
    rows.append({"solver": "greedy", "budget_M": greedy_evals, "p_opt": float(np.isclose(e_g, e_opt)),
                 "expected_objective": e_g, "best_of_M": e_g, "approx_ratio_expected": ratio(e_g),
                 "approx_ratio_best_of_M": ratio(e_g), "wall_sec": t_g})

    for m in args.budgets:
        p_rand = 1 - (1 - n_opt / len(values)) ** m
        rows.append({"solver": "random_sampling", "budget_M": m, "p_opt": p_rand, "expected_objective": e_rand,
                     "best_of_M": best_of_m(values, np.full(len(values), 1 / len(values)), m),
                     "approx_ratio_expected": 0.0, "wall_sec": 0.0})
        rows[-1]["approx_ratio_best_of_M"] = ratio(rows[-1]["best_of_M"])
        t = time.time()
        res = [anneal_select(qubo, k, seed=r, n_evals=m)[1] for r in range(args.sa_runs)]
        t_sa = (time.time() - t) / args.sa_runs
        res = np.array(res)
        rows.append({"solver": "simulated_annealing", "budget_M": m, "p_opt": float(np.isclose(res, e_opt).mean()),
                     "expected_objective": float(res.mean()), "best_of_M": float(res.mean()),
                     "approx_ratio_expected": ratio(res.mean()), "approx_ratio_best_of_M": ratio(res.mean()),
                     "wall_sec": t_sa, "note": f"best-found per run, mean over {args.sa_runs} runs"})

    t = time.time()
    A = xy_mixer(combos, masks, n)
    print(f"XY mixer: {A.shape[0]} states, {A.nnz} nonzeros, built in {time.time() - t:.1f}s")
    mixer = JohnsonMixer(A, n, k)
    check_rng = np.random.default_rng(1)
    psi0 = check_rng.normal(size=len(values)) + 1j * check_rng.normal(size=len(values))
    psi0 /= np.linalg.norm(psi0)
    mixer_check = {}
    for beta in (0.05, 0.3, 1.0):
        ref = expm_multiply(-1j * beta * A, psi0)
        mixer_check[beta] = float(np.abs(mixer.apply(beta, psi0) - ref).max())
    print(f"spectral mixer vs expm_multiply, max |diff|: {mixer_check}")
    opt_mask = np.isclose(values, e_opt)
    qaoa_details = {"note": "QAOA classically simulated in numpy (statevector in the Hamming-weight-K "
                            "subspace); not a QPU run.",
                    "n_states": int(len(values)), "mixer_nnz": int(A.nnz),
                    "mixer_check_max_abs_diff_vs_expm_multiply": mixer_check, "per_depth": {}}
    rng = np.random.default_rng(args.qaoa_seed)
    for p in args.depths:
        n_calls = 0

        def neg_expectation(params):
            nonlocal n_calls
            n_calls += 1
            probs = np.abs(qaoa_state(params, values, mixer, p)) ** 2
            return -float(probs @ values)

        t = time.time()
        best = None
        for r in range(args.qaoa_restarts):
            x0 = np.concatenate([rng.uniform(0, 0.5, p), rng.uniform(0, 0.5, p)])
            res = minimize(neg_expectation, x0, method="COBYLA", options={"maxiter": args.qaoa_maxiter})
            if best is None or res.fun < best.fun:
                best = res
        t_opt = time.time() - t
        probs = np.abs(qaoa_state(best.x, values, mixer, p)) ** 2
        p_opt_shot = float(probs[opt_mask].sum())
        e_q = float(probs @ values)
        qaoa_details["per_depth"][p] = {
            "gammas": best.x[:p].tolist(), "betas": best.x[p:].tolist(), "expected_objective": e_q,
            "p_opt_single_shot": p_opt_shot, "p_opt_uniform": float(opt_mask.mean()),
            "circuit_evaluations_for_training": n_calls, "training_wall_sec": t_opt,
        }
        print(f"QAOA p={p}: E={e_q:.4f} ratio={ratio(e_q):.3f} P(opt)/shot={p_opt_shot:.5f} "
              f"(uniform {opt_mask.mean():.5f}) evals={n_calls} {t_opt:.0f}s")
        for m in args.budgets:
            bom = best_of_m(values, probs, m)
            rows.append({"solver": f"qaoa_p{p}", "budget_M": m, "p_opt": 1 - (1 - p_opt_shot) ** m,
                         "expected_objective": e_q, "best_of_M": bom, "approx_ratio_expected": ratio(e_q),
                         "approx_ratio_best_of_M": ratio(bom), "wall_sec": t_opt,
                         "note": f"M measurement shots after angle training ({n_calls} statevector evals)"})

    table = pd.DataFrame(rows)
    table.to_csv(out / "solver_table.csv", index=False)
    qaoa_details |= {"args": vars(args) | {"lam": lam}, "pool_train_idx": [int(i) for i in pool],
                     "optimum": e_opt, "n_optimal_subsets": n_opt, "random_mean": e_rand,
                     "optimal_subset_train_idx": [int(pool[i]) for i in combos[opt_idx]],
                     "greedy_subset_train_idx": [int(i) for i in pool[g > 0]],
                     "optimal_terms": qubo.terms(np.isin(np.arange(n), combos[opt_idx]).astype(float)),
                     "scale_check_random_subsets": scale}
    with open(out / "qaoa_details.json", "w") as f:
        json.dump(qaoa_details, f, indent=1)
    print("scale check over 1000 random K-subsets (std across subsets):")
    for term, v in scale.items():
        print(f"  {term:16s} mean={v['mean']:+.3f} std={v['std']:.3f}")
    print("optimal subset terms:", {k_: round(v, 3) for k_, v in qaoa_details["optimal_terms"].items()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.drop(columns=["note"], errors="ignore").round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["A", "B", "C"], required=True)
    ap.add_argument("--methods", nargs="+", default=["all"], choices=["all"] + LAYER_B_METHODS)
    ap.add_argument("--pool", type=int, default=None, help="pool size (default 150 for B, 20 for A)")
    ap.add_argument("--pool-b", type=int, default=150, help="Layer A draws its pool from the Layer B pool")
    ap.add_argument("--strata", type=int, default=5)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="budget (default 40 for B, 6 for A)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=None, help="default 1/(K-1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sa-evals", type=int, default=200000, help="Layer B SA swap proposals")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 1000], help="Layer A budgets M")
    ap.add_argument("--sa-runs", type=int, default=200, help="Layer A SA repetitions per budget")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--qaoa-restarts", type=int, default=3)
    ap.add_argument("--qaoa-maxiter", type=int, default=80)
    ap.add_argument("--qaoa-seed", type=int, default=0)
    ap.add_argument("--norms", nargs="+", default=["minmax", "zscore"], choices=["minmax", "zscore"],
                    help="Layer C: normalisation of d and c")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5], help="Layer C coverage weights")
    ap.add_argument("--lam-mults", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                    help="Layer C redundancy weights, in units of 1/(K-1)")
    ap.add_argument("--split-seed", type=int, default=0, help="Layer C select/report eval split")
    ap.add_argument("--confirm-seeds", type=int, nargs="+", default=[2, 3, 4],
                    help="Layer C extra seeds for the selected config and hardest-K")
    args = ap.parse_args()
    if args.layer == "C" and args.seeds == [0, 1, 2]:
        args.seeds = [0, 1]
    if args.pool is None:
        args.pool = 20 if args.layer == "A" else 150
    if args.k is None:
        args.k = 6 if args.layer == "A" else 40

    device = get_device()
    print(f"device={device}")
    {"A": run_layer_a, "B": run_layer_b, "C": run_layer_c}[args.layer](args, device)


if __name__ == "__main__":
    main()
