"""
gradient_qubo_selection.py
Failure-targeted gradient-matching QUBO for choosing UNet fine-tuning frames.

Pool: all 800 Kvasir-SEG train frames; budget K frames. Each frame i has the
loss gradient g_i of the pretrained UNet's last two decoder blocks + head
(gradient_features.py). "Failures" are frames with baseline Dice < 0.80 in the
*select* halves of two eval sets:
    Kvasir val+test (200 frames, split in half stratified by hard, as in Layer C)
    CVC-ClinicDB (612 frames, split by video sequence)
t = mean failure gradient (Kvasir and CVC halves weighted equally).
The QUBO is gradient matching of the subset's mean gradient to t:

    min_x || (1/K) sum_i x_i g_i - t ||^2,  sum_i x_i = K
 == max_x sum_i x_i (g_i.t - |g_i|^2 / 2K) - (1/K) sum_{i<j} x_i x_j g_i.g_j

The pairwise term penalises frames whose gradients point the same way
(redundant) and rewards frames whose gradients cancel each other's off-target
components; a sort on g_i.t cannot express either. Every method is reported on
the *report* halves only (Kvasir report half, held-out CVC sequences).

Methods: no_finetune, all_800, random_k, hardest_k (train loss), kcenter
(encoder embeddings), craig (facility location on gradients), align_topk (sort by
g_i.t, first-order GLISTER), gradmatch_omp (OMP on t, GRAD-MATCH), and the QUBO
solved by greedy and by simulated annealing; qubo_div is an untargeted
alternative (train loss + gradient-cosine redundancy penalty, no eval gradients).

Usage (from repo root):
    S=quantum-medical-research/frame_selection/gradient_qubo_selection.py
    ./venv/bin/python $S --features          # CVC arrays + CVC gradients (once)
    ./venv/bin/python $S --methods all --seeds 0 1 2
    ./venv/bin/python $S --summarize
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_selection_qubo import (DATA_DIR, HARD_THRESHOLD, OUT_DIR, Qubo, anneal_select, eval_halves,  # noqa: E402
                                  finetune, get_device, greedy_select, load_features, load_model,
                                  load_transforms, predict_dice)
from gradient_features import per_frame_gradients  # noqa: E402

CVC_DIR = DATA_DIR.parent / "CVC-ClinicDB"
RUN_DIR = OUT_DIR / "gradient_qubo"
METHODS = ["no_finetune", "all_800", "random_k", "hardest_k", "kcenter", "craig", "align_topk",
           "gradmatch_omp", "qubo_greedy", "qubo_sa", "qubo_div"]
MW_METHODS = [f"mw{lam}_{tgt}_{solver}" for lam in (4, 16, 64) for tgt in ("fail", "th")
              for solver in ("greedy", "sa")]
MW_METHODS += ["mw4_th_ls", "mw4_th_sar4", "mw4_th_sar5"]  # solver-quality ladder
METHODS += MW_METHODS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cvc(device):
    cache = OUT_DIR / "cvc_256.npz"
    if not cache.exists():
        md = pd.read_csv(CVC_DIR / "metadata.csv")
        img = np.zeros((len(md), 3, 256, 256), np.float32)
        msk = np.zeros((len(md), 1, 256, 256), np.float32)
        for i, r in md.iterrows():
            d = load_transforms({"image": str(CVC_DIR.parent / r.image_path), "mask": str(CVC_DIR.parent / r.mask_path)})
            m = np.asarray(d["mask"])
            img[i], msk[i] = np.asarray(d["image"])[:3], (m.mean(0, keepdims=True) > 0.5)
        base = predict_dice(load_model(device), img, msk, device)
        np.savez(cache, img=img, mask=msk, base=base, seq=md.sequence_id.values)
    return dict(np.load(cache))


def build_features(device):
    cvc = load_cvc(device)
    model = load_model(device)
    head_params = {id(p) for mod in (model.decoder.blocks[3], model.decoder.blocks[4], model.segmentation_head)
                   for p in mod.parameters()}
    n_par = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(0)
    sketch = (torch.randint(0, 8192, (n_par,), generator=g).to(device),
              (torch.randint(0, 2, (n_par,), generator=g) * 2 - 1).float().to(device))
    head, _, loss = per_frame_gradients(model, cvc["img"], cvc["mask"], device, head_params, sketch)
    np.savez(OUT_DIR / "grad_features_cvc_256.npz", head=head, loss=loss)
    print(f"CVC: {len(head)} frames, baseline Dice {cvc['base'].mean():.4f}, hard {(cvc['base'] < HARD_THRESHOLD).sum()}")


def cvc_halves(seq, seed):
    rng = np.random.default_rng(seed)
    seqs = np.unique(seq)
    sel_seqs = rng.choice(seqs, len(seqs) // 2, replace=False)
    sel = np.isin(seq, sel_seqs)
    return sel, ~sel


def setup(args):
    z = np.load(OUT_DIR / "features_256.npz")
    G = np.load(OUT_DIR / "grad_features_256.npz")
    Gc = np.load(OUT_DIR / "grad_features_cvc_256.npz")
    cvc_base, cvc_seq = np.load(OUT_DIR / "cvc_256.npz")["base"], np.load(OUT_DIR / "cvc_256.npz")["seq"]
    kv_hard = z["ev_base"] < HARD_THRESHOLD
    cvc_hard = cvc_base < HARD_THRESHOLD
    kv_sel, kv_rep = eval_halves(kv_hard, args.split_seed)
    cvc_sel, cvc_rep = cvc_halves(cvc_seq, args.split_seed)
    t = 0.5 * G["head_eval"][kv_sel & kv_hard].mean(0) + 0.5 * Gc["head"][cvc_sel & cvc_hard].mean(0)
    return {"train_dice": z["train_dice"], "emb": z["emb"], "g": G["head_train"].astype(np.float64),
            "loss": G["loss_train"], "t": t.astype(np.float64),
            "masks": {"kv_hard": kv_hard, "kv_sel": kv_sel, "kv_rep": kv_rep,
                      "cvc_hard": cvc_hard, "cvc_sel": cvc_sel, "cvc_rep": cvc_rep}}


# ---------------------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------------------
def gradient_qubo(g, t, k, scale=None):
    """Qubo (lin, Q) for min ||mean_K g - t||^2, rescaled so the linear term has unit std."""
    a = g @ t - (g * g).sum(1) / (2 * k)
    gram = g @ g.T
    np.fill_diagonal(gram, 0.0)
    s = scale or a.std()
    return Qubo(a / s, np.zeros(len(a)), gram / (k * s), 1.0, 0.0, 1.0), s


def local_search(qubo, x):
    """Steepest-ascent 1-swap local search from x (classical refinement of greedy)."""
    x = x.copy()
    field = qubo.lin - qubo.Q @ x
    while True:
        sel, uns = np.flatnonzero(x > 0), np.flatnonzero(x == 0)
        gain = -field[sel][:, None] + field[uns][None, :] + qubo.Q[np.ix_(sel, uns)]
        i, j = np.unravel_index(gain.argmax(), gain.shape)
        if gain[i, j] <= 1e-9:
            return x
        a, b = sel[i], uns[j]
        x[a], x[b] = 0, 1
        field += qubo.Q[:, a] - qubo.Q[:, b]


def match_qubo(g, t, k, lam):
    """max a.x - (lam / 2K) ||sum_i x_i g_i||^2 (lam=1 is gradient_qubo): a large lam penalises the
    norm of the net update and rewards frames whose gradients cancel, which makes greedy fail."""
    a = g @ t
    gram = g @ g.T
    dg = np.diag(gram).copy()
    np.fill_diagonal(gram, 0.0)
    s = a.std()
    return Qubo((a - lam * dg / (2 * k)) / s, np.zeros(len(a)), lam * gram / (k * s), 1.0, 0.0, 1.0)


def diversity_qubo(g, loss, k, lam_mult):
    """Untargeted: max z(loss).x - lam sum_{i<j} S_ij x_i x_j, S = z-scored gradient cosine."""
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    S = u @ u.T
    off = ~np.eye(len(S), dtype=bool)
    S = (S - S[off].mean()) / S[off].std()
    np.fill_diagonal(S, 0.0)
    d = (loss - loss.mean()) / loss.std()
    return Qubo(d, np.zeros(len(d)), S, 1.0, 0.0, lam_mult / (k - 1))


def matching_error(g, t, x):
    return float(np.sum((g[x > 0].mean(0) - t) ** 2))


def kcenter(emb, k, seed):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(emb)))]
    dist = 1 - emb @ emb[chosen[0]]
    for _ in range(k - 1):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, 1 - emb @ emb[i])
    return chosen


def craig(g, k):
    """Facility location on gradient similarity (CRAIG): max sum_j max_{i in S} sim_ij, lazy-free greedy."""
    gn = g / np.linalg.norm(g, axis=1, keepdims=True)
    sim = gn @ gn.T
    sim = sim - sim.min()
    best = np.zeros(len(g))
    chosen = []
    for _ in range(k):
        gains = np.maximum(sim, best[None, :]).sum(1) - best.sum()
        gains[chosen] = -np.inf
        i = int(np.argmax(gains))
        chosen.append(i)
        best = np.maximum(best, sim[i])
    return chosen


def omp_select(g, t, k):
    """GRAD-MATCH style OMP with non-negative weights; returns the support (size k)."""
    A = g.T  # (P, N)
    chosen, r = [], t.copy()
    for _ in range(k):
        corr = A.T @ r
        corr[chosen] = -np.inf
        chosen.append(int(np.argmax(corr)))
        w, *_ = np.linalg.lstsq(A[:, chosen], t, rcond=None)
        w = np.clip(w, 0, None)
        r = t - A[:, chosen] @ w
    return chosen


def select_all(S, k, sa_evals, div_lam_mult=4.0):
    g, t, n = S["g"], S["t"], len(S["g"])
    sel, info = {}, {}

    def vec(idx):
        x = np.zeros(n)
        x[list(idx)] = 1
        return x

    q, scale = gradient_qubo(g, t, k)
    qd = diversity_qubo(g, S["loss"], k, div_lam_mult)
    timing = {}
    for name, fn in [
        ("hardest_k", lambda: vec(np.argsort(-S["loss"], kind="stable")[:k])),
        ("kcenter", lambda: vec(kcenter(S["emb"], k, 0))),
        ("craig", lambda: vec(craig(g, k))),
        ("align_topk", lambda: vec(np.argsort(-(g @ t), kind="stable")[:k])),
        ("gradmatch_omp", lambda: vec(omp_select(g, t, k))),
        ("qubo_greedy", lambda: greedy_select(q, k)),
        ("qubo_sa", lambda: max((anneal_select(q, k, seed=r, n_evals=sa_evals) for r in range(4)),
                                key=lambda v: v[1])[0]),
        ("qubo_div", lambda: max((anneal_select(qd, k, seed=r, n_evals=sa_evals) for r in range(4)),
                                 key=lambda v: v[1])[0]),
        ("all_800", lambda: np.ones(n)),
    ]:
        t0 = time.time()
        sel[name] = fn()
        timing[name] = time.time() - t0
    t_th = g[np.argsort(-S["loss"], kind="stable")[:k]].mean(0)  # train-only target: hardest-K mean gradient
    for m in MW_METHODS:
        lam, tgt, solver = m.split("_")
        qm = match_qubo(g, t if tgt == "fail" else t_th, k, float(lam[2:]))
        t0 = time.time()
        if solver == "greedy":
            sel[m] = greedy_select(qm, k)
        elif solver == "ls":
            sel[m] = local_search(qm, greedy_select(qm, k))
        elif solver.startswith("sar"):
            sel[m] = anneal_select(qm, k, seed=int(solver[3:]), n_evals=sa_evals)[0]
        else:
            sel[m] = max((anneal_select(qm, k, seed=r, n_evals=sa_evals) for r in range(4)), key=lambda v: v[1])[0]
        timing[m] = time.time() - t0
        info.setdefault("mw_objective", {})[m] = qm.objective(sel[m])
    for name, x in sel.items():
        info[name] = {"qubo_objective": q.objective(x), "matching_error": matching_error(g, t, x),
                      "mean_train_dice": float(S["train_dice"][x > 0].mean()),
                      "overlap_hardest": int((x * sel["hardest_k"]).sum()),
                      "overlap_qubo_sa": int((x * sel["qubo_sa"]).sum()), "solve_sec": timing[name]}
    rng = np.random.default_rng(0)
    rand = [vec(rng.choice(n, k, replace=False)) for _ in range(200)]
    info["random_k"] = {"qubo_objective": float(np.mean([q.objective(x) for x in rand])),
                        "matching_error": float(np.mean([matching_error(g, t, x) for x in rand]))}
    return sel, info


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def score(dice_kv, dice_cvc, m):
    return {"kv_rep_hard": dice_kv[m["kv_rep"] & m["kv_hard"]].mean(), "kv_rep_full": dice_kv[m["kv_rep"]].mean(),
            "cvc_rep_hard": dice_cvc[m["cvc_rep"] & m["cvc_hard"]].mean(),
            "cvc_rep_full": dice_cvc[m["cvc_rep"]].mean(),
            "kv_sel_hard": dice_kv[m["kv_sel"] & m["kv_hard"]].mean(),
            "cvc_sel_hard": dice_cvc[m["cvc_sel"] & m["cvc_hard"]].mean()}


def run(args, device):
    out = RUN_DIR / f"k{args.k}"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    S = setup(args)
    sel, info = select_all(S, args.k, args.sa_evals)
    with open(out / "selection_info.json", "w") as f:
        json.dump({"args": vars(args), "info": info,
                   "selected": {m: [int(i) for i in np.flatnonzero(x)] for m, x in sel.items() if m != "all_800"}},
                  f, indent=1)
    for m, v in info.items():
        print(f"{m:14s} " + " ".join(f"{a}={b:.4g}" for a, b in v.items()))
    methods = METHODS if "all" in args.methods else args.methods
    _, data = load_features(device, need_eval_data=True)
    cvc = load_cvc(device)
    base = load_model(device)
    np.savez(out / "runs" / "no_finetune.npz", kv=predict_dice(base, data["ev_img"], data["ev_mask"], device),
             cvc=cvc["base"])
    for s in args.seeds:
        for m in methods:
            if m == "no_finetune":
                continue
            f = out / "runs" / f"{m}_s{s}.npz"
            if f.exists():
                continue
            if m == "random_k":
                idx = np.random.default_rng(100 + s).choice(len(S["g"]), args.k, replace=False)
            else:
                idx = np.flatnonzero(sel[m])
            t0 = time.time()
            model = finetune(data["tr_img"][idx], data["tr_mask"][idx], device, seed=s,
                             steps=args.steps, lr=args.lr, batch=args.batch)
            kv = predict_dice(model, data["ev_img"], data["ev_mask"], device)
            cv = predict_dice(model, cvc["img"], cvc["mask"], device)
            np.savez(f, kv=kv, cvc=cv)
            sc = score(kv, cv, S["masks"])
            print(f"{m:14s} s={s} " + " ".join(f"{a}={b:.4f}" for a, b in sc.items()) + f" ({time.time() - t0:.0f}s)",
                  flush=True)
    summarize(args)


def summarize(args):
    out = RUN_DIR / f"k{args.k}"
    S = setup(args)
    m = S["masks"]
    nf = np.load(out / "runs" / "no_finetune.npz")
    base = score(nf["kv"], nf["cvc"], m)
    rows = []
    for f in sorted((out / "runs").glob("*_s*.npz")):
        name, seed = f.stem.rsplit("_s", 1)
        z = np.load(f)
        rows.append({"method": name, "seed": int(seed), **score(z["kv"], z["cvc"], m)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    metrics = ["cvc_rep_hard", "cvc_rep_full", "kv_rep_hard", "kv_rep_full"]
    tab = df.groupby("method")[metrics].agg(["mean", "std", "count"])
    gains = pd.DataFrame({k: tab[(k, "mean")] - base[k] for k in metrics})
    gains["n"] = tab[(metrics[0], "count")]
    # paired-by-seed comparison of qubo_sa with every other method
    ref = df[df.method == "qubo_sa"].set_index("seed")
    for k in metrics:
        p, d = {}, {}
        for meth, g in df.groupby("method"):
            g = g.set_index("seed")
            common = ref.index.intersection(g.index)
            if meth == "qubo_sa" or len(common) < 2:
                continue
            diff = ref.loc[common, k] - g.loc[common, k]
            d[meth], p[meth] = diff.mean(), stats.ttest_1samp(diff, 0).pvalue
        gains[f"sa_minus_{k}"] = pd.Series(d)
        gains[f"p_{k}"] = pd.Series(p)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("baseline", {k: round(v, 4) for k, v in base.items()})
    print(gains.sort_values("cvc_rep_hard", ascending=False).round(4).to_string())
    gains.to_csv(out / "summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["all"], choices=["all"] + METHODS)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sa-evals", type=int, default=300000)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()
    device = get_device()
    if args.features:
        build_features(device)
    elif args.summarize:
        summarize(args)
    elif args.select_only:
        S = setup(args)
        _, info = select_all(S, args.k, args.sa_evals)
        for m, v in info.items():
            print(f"{m:14s} " + " ".join(f"{a}={b:.4g}" for a, b in v.items()))
    else:
        run(args, device)


if __name__ == "__main__":
    main()
