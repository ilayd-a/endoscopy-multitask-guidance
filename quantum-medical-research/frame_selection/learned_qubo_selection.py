"""
learned_qubo_selection.py
Datamodel-style learned QUBO for choosing UNet fine-tuning frames.

The hand-designed gradient QUBOs (gradient_qubo_selection.py) are not calibrated
to fine-tuning outcome: near-optimal subsets differ by +-0.01 Dice. Here the QUBO
is *learned* from fine-tuning runs instead.

Each train frame i gets a small feature vector phi_i (r dims: train loss, top
principal components of its head gradient and of its encoder embedding). A subset
x (|x| = K) is summarised by its mean feature m(x) = Phi^T x / K, and the outcome
(Dice gain on the *select* halves of Kvasir and CVC) is modelled as

    y(x) = b + theta.m + m^T W m - rho * (m - mu)^T P (m - mu)

The first three terms are a quadratic datamodel fit by ridge on sampled subsets;
the last is a trust region (P = inverse covariance of m over the sampled subsets)
so the solver cannot run to corners where the fit has no data. Expanding m in x
gives a QUBO with linear terms and dense rank-r pairwise couplings
A_ij = phi_i^T (W - rho P) phi_j, which can be frustrated when W is indefinite.

Stages:
    --sample 0 160   fine-tune sampled subsets (Gumbel top-K on random scores mixing
                     hardness and feature directions), 1 seed each
    --fit            CV of the datamodel + out-of-sample check on the method subsets
                     of gradient_qubo/k100
    --solve          greedy / local search / SA on the learned QUBO
    --run            fine-tune the solver subsets (--seeds), report on report halves

Usage (from repo root):
    S=quantum-medical-research/frame_selection/learned_qubo_selection.py
    ./venv/bin/python $S --sample 0 160
    ./venv/bin/python $S --fit
    ./venv/bin/python $S --run --seeds 0 1 2
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_selection_qubo import (OUT_DIR, Qubo, anneal_select, finetune, get_device, greedy_select,  # noqa: E402
                                  load_features, load_model, predict_dice)
from gradient_qubo_selection import RUN_DIR, load_cvc, local_search, score, setup  # noqa: E402

LQ_DIR = OUT_DIR / "learned_qubo"
METRICS = ["kv_hard", "kv_full", "cvc_hard", "cvc_full"]
SOLVERS = ["greedy", "ls", "sa"] + [f"sar{r}" for r in range(4)]


# ---------------------------------------------------------------------------
# Frame features and subset sampler
# ---------------------------------------------------------------------------
def frame_features(S, n_grad=5, n_emb=2):
    """phi (N, r): z(train loss), top PCs of head gradients and of embeddings, each column z-scored."""
    def pcs(a, n):
        a = a - a.mean(0)
        u, s, _ = np.linalg.svd(a @ a.T)
        return u[:, :n] * np.sqrt(s[:n])
    phi = np.column_stack([S["loss"], pcs(S["g"], n_grad), pcs(S["emb"].astype(np.float64), n_emb)])
    return (phi - phi.mean(0)) / phi.std(0)


def sample_subset(phi, k, n):
    """Subset n: Gumbel top-K on score = beta * z(loss) + gamma * u.phi (every 4th subset uniform)."""
    rng = np.random.default_rng(10_000 + n)
    score_ = np.zeros(len(phi))
    if n % 4:
        beta, gamma = rng.uniform(-1.0, 3.0), rng.uniform(0.0, 2.0)
        u = rng.normal(size=phi.shape[1] - 1)
        score_ = beta * phi[:, 0] + gamma * phi[:, 1:] @ (u / np.linalg.norm(u))
    return np.sort(np.argsort(-(score_ + rng.gumbel(size=len(phi))))[:k])


# ---------------------------------------------------------------------------
# Machine guard (thermal / battery / rest), checked before every fine-tune
# ---------------------------------------------------------------------------
class Guard:
    def __init__(self, rest_after=2100, rest=240):
        self.t0, self.rest_after, self.rest = time.time(), rest_after, rest

    def wait(self):
        while True:
            therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
            lim = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
            if (lim and int(lim.group(1)) < 100) or re.search(r"warning level\D*[1-9]", therm):
                print(time.strftime("%T"), "thermal pause", flush=True)
                time.sleep(300)
                continue
            break
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        pct = re.search(r"(\d+)%", batt)
        if "Battery Power" in batt and pct and int(pct.group(1)) <= 5:
            print("battery stop", flush=True)
            sys.exit(0)
        if time.time() - self.t0 > self.rest_after:
            print(time.strftime("%T"), "rest", flush=True)
            time.sleep(self.rest)
            self.t0 = time.time()


def finetune_eval(idx, seed, args, data, cvc, device, guard):
    guard.wait()
    t0 = time.time()
    model = finetune(data["tr_img"][idx], data["tr_mask"][idx], device, seed=seed,
                     steps=args.steps, lr=args.lr, batch=args.batch)
    kv = predict_dice(model, data["ev_img"], data["ev_mask"], device)
    cv = predict_dice(model, cvc["img"], cvc["mask"], device)
    dt = time.time() - t0
    if dt > 90:
        print(f"slow run ({dt:.0f}s), pausing", flush=True)
        time.sleep(300)
    return kv, cv, dt


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def gains(kv, cv, m, base, half):
    """Dice gain over no fine-tuning on the given half ('sel' or 'rep') for the four metrics."""
    out = {}
    for key in METRICS:
        ds, part = key.split("_")
        mask = m[f"{ds}_{half}"] & (m[f"{ds}_hard"] if part == "hard" else True)
        d = kv if ds == "kv" else cv
        b = base["kv"] if ds == "kv" else base["cvc"]
        out[key] = float(d[mask].mean() - b[mask].mean())
    return out


def load_samples(S, base):
    rows = []
    for f in sorted((LQ_DIR / "samples").glob("sub*.npz")):
        z = np.load(f)
        rows.append({"name": f.stem, "idx": z["idx"], **gains(z["kv"], z["cvc"], S["masks"], base, "sel"),
                     **{f"rep_{a}": b for a, b in gains(z["kv"], z["cvc"], S["masks"], base, "rep").items()}})
    return rows


def load_method_subsets(S, base, k):
    """Subsets chosen by the methods of gradient_qubo/k100, outcome averaged over their seeds."""
    info = json.load(open(RUN_DIR / f"k{k}" / "selection_info.json"))["selected"]
    rows = []
    for meth, idx in info.items():
        files = sorted((RUN_DIR / f"k{k}" / "runs").glob(f"{meth}_s*.npz"))
        files = [f for f in files if f.stem.rsplit("_s", 1)[0] == meth]
        if not files:
            continue
        g = [gains(np.load(f)["kv"], np.load(f)["cvc"], S["masks"], base, "sel") for f in files]
        rows.append({"name": meth, "idx": np.array(idx), "n_seeds": len(files),
                     **{key: float(np.mean([r[key] for r in g])) for key in METRICS}})
    return rows


def composite(rows, scale):
    return np.array([np.mean([r[key] / scale[key] for key in METRICS]) for r in rows])


# ---------------------------------------------------------------------------
# Datamodel and learned QUBO
# ---------------------------------------------------------------------------
def design(M):
    """[m, upper-triangular m_a m_b] for a quadratic model in the mean feature m."""
    iu = np.triu_indices(M.shape[1])
    return np.column_stack([M, M[:, iu[0]] * M[:, iu[1]]])


class Datamodel:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, M, y):
        X = design(M)
        self.xm, self.xs, self.ym = X.mean(0), X.std(0) + 1e-12, y.mean()
        Xs = (X - self.xm) / self.xs
        self.coef = np.linalg.solve(Xs.T @ Xs + self.alpha * np.eye(Xs.shape[1]), Xs.T @ (y - self.ym))
        r = M.shape[1]
        c = self.coef / self.xs
        self.theta = c[:r]
        iu = np.triu_indices(r)
        W = np.zeros((r, r))
        W[iu] = c[r:]
        self.W = (W + W.T) / 2  # m^T W m reproduces sum_{a<=b} c_ab m_a m_b
        self.b = self.ym - c @ self.xm
        return self

    def predict(self, M):
        return self.b + M @ self.theta + np.einsum("na,ab,nb->n", M, self.W, M)


def cv_scores(M, y, alpha, folds=5, seed=0):
    pred = np.zeros_like(y)
    for f in np.array_split(np.random.default_rng(seed).permutation(len(y)), folds):
        tr = np.setdiff1d(np.arange(len(y)), f)
        pred[f] = Datamodel(alpha).fit(M[tr], y[tr]).predict(M[f])
    return 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2), stats.spearmanr(pred, y)[0], pred


def learned_qubo(dm, phi, k, rho, mu, P):
    """QUBO for max y(x) = b + theta.m + m^T (W - rho P) m + linear trust-region terms, m = phi^T x / K."""
    Wt = dm.W - rho * P
    A = phi @ Wt @ phi.T
    lin = phi @ (dm.theta + 2 * rho * P @ mu) / k + np.diag(A) / k ** 2
    Q = -2 * A / k ** 2
    np.fill_diagonal(Q, 0.0)
    s = lin.std()
    q = Qubo(lin / s, np.zeros(len(lin)), Q / s, 1.0, 0.0, 1.0)
    q.const, q.scale = dm.b - rho * mu @ P @ mu, s
    return q


def predicted(q, x):
    return q.const + q.scale * q.objective(x)


def solve(q, k, sa_evals):
    sel = {"greedy": greedy_select(q, k)}
    sel["ls"] = local_search(q, sel["greedy"])
    sa = [anneal_select(q, k, seed=r, n_evals=sa_evals) for r in range(4)]
    for r, (x, _) in enumerate(sa):
        sel[f"sar{r}"] = x
    sel["sa"] = max(sa, key=lambda v: v[1])[0]
    return sel


def build(args, S=None, verbose=True):
    """Fit the datamodel on all sampled subsets and return (S, phi, dm, qubo, mahalanobis bounds)."""
    S = S or setup(args)
    phi = frame_features(S)
    base = np.load(RUN_DIR / f"k{args.k}" / "runs" / "no_finetune.npz")
    rows = load_samples(S, base)
    scale = {key: np.std([r[key] for r in rows]) for key in METRICS}
    y = composite(rows, scale) if args.target == "composite" else np.array([r[args.target] for r in rows])
    M = np.array([phi[r["idx"]].mean(0) for r in rows])
    mu, P = M.mean(0), np.linalg.inv(np.cov(M.T))
    dm = Datamodel(args.alpha).fit(M, y)
    maha = np.einsum("na,ab,nb->n", M - mu, P, M - mu)
    q = learned_qubo(dm, phi, args.k, args.rho, mu, P)
    return {"S": S, "phi": phi, "base": base, "rows": rows, "scale": scale, "y": y, "M": M, "mu": mu, "P": P,
            "dm": dm, "maha": maha, "q": q}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def sample_stage(args, device):
    S = setup(args)
    phi = frame_features(S)
    _, data = load_features(device, need_eval_data=True)
    cvc = load_cvc(device)
    load_model(device)
    (LQ_DIR / "samples").mkdir(parents=True, exist_ok=True)
    guard = Guard()
    for n in range(*args.sample):
        f = LQ_DIR / "samples" / f"sub{n:03d}.npz"
        if f.exists():
            continue
        idx = sample_subset(phi, args.k, n)
        kv, cv, dt = finetune_eval(idx, n, args, data, cvc, device, guard)
        np.savez(f, idx=idx, kv=kv, cvc=cv)
        sc = score(kv, cv, S["masks"])
        print(f"sub{n:03d} loss_z={phi[idx, 0].mean():+.2f} " + " ".join(f"{a}={b:.4f}" for a, b in sc.items())
              + f" ({dt:.0f}s)", flush=True)
    print("DONE", flush=True)


def fit_stage(args):
    B = build(args)
    rows, phi, M, y, S = B["rows"], B["phi"], B["M"], B["y"], B["S"]
    print(f"{len(rows)} sampled subsets; composite scale {B['scale']}")
    for key in METRICS:
        v = np.array([r[key] for r in rows])
        rv = np.array([r[f"rep_{key}"] for r in rows])
        print(f"  {key}: sel mean {v.mean():+.4f} sd {v.std():.4f}; sel~rep spearman {stats.spearmanr(v, rv)[0]:.2f}")
    print("CV (5-fold) of the composite:")
    for alpha in (0.1, 1, 10, 100, 1000):
        r2, rho, _ = cv_scores(M, y, alpha)
        print(f"  alpha={alpha:<6} R2={r2:+.3f} spearman={rho:+.3f}")
    Xl = M  # linear-only reference
    for alpha in (1, 10, 100):
        pred = np.zeros_like(y)
        for f in np.array_split(np.random.default_rng(0).permutation(len(y)), 5):
            tr = np.setdiff1d(np.arange(len(y)), f)
            Xs = np.column_stack([np.ones(len(tr)), Xl[tr]])
            w = np.linalg.solve(Xs.T @ Xs + alpha * np.diag([0] + [1] * Xl.shape[1]), Xs.T @ y[tr])
            pred[f] = np.column_stack([np.ones(len(f)), Xl[f]]) @ w
        print(f"  linear-only alpha={alpha:<4} R2={1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2):+.3f} "
              f"spearman={stats.spearmanr(pred, y)[0]:+.3f}")
    # out-of-sample: method subsets (never used in the fit)
    meth = load_method_subsets(S, B["base"], args.k)
    ym = composite(meth, B["scale"])
    Mm = np.array([phi[r["idx"]].mean(0) for r in meth])
    pm = B["dm"].predict(Mm)
    mm = np.einsum("na,ab,nb->n", Mm - B["mu"], B["P"], Mm - B["mu"])
    print(f"method subsets: spearman(pred, actual composite) = {stats.spearmanr(pm, ym)[0]:+.3f} (n={len(meth)})")
    for r, a, p, d in sorted(zip(meth, ym, pm, mm), key=lambda v: -v[1]):
        print(f"  {r['name']:16s} actual={a:+.3f} pred={p:+.3f} maha={d:6.1f} " +
              " ".join(f"{key}={r[key]:+.4f}" for key in METRICS))
    print(f"sampled maha: median {np.median(B['maha']):.1f}, 95% {np.percentile(B['maha'], 95):.1f}, "
          f"max {B['maha'].max():.1f}")
    print(f"theta {np.round(B['dm'].theta, 3)}")
    print(f"W eigenvalues {np.round(np.linalg.eigvalsh(B['dm'].W), 3)}")


def solve_stage(args):
    B = build(args)
    q, phi = B["q"], B["phi"]
    sel = solve(q, args.k, args.sa_evals)
    hard = np.argsort(-B["S"]["loss"], kind="stable")[:args.k]
    for name, x in sel.items():
        m = phi[x > 0].mean(0)
        d = (m - B["mu"]) @ B["P"] @ (m - B["mu"])
        print(f"{name:7s} pred={predicted(q, x):+.4f} maha={d:5.1f} overlap_sa={int((x * sel['sa']).sum()):3d} "
              f"overlap_greedy={int((x * sel['greedy']).sum()):3d} overlap_hardest={int(x[hard].sum()):3d} "
              f"loss_z={m[0]:+.2f}")
    print(f"best sampled subset: actual {B['y'].max():+.4f}, predicted {B['dm'].predict(B['M']).max():+.4f}")
    return B, sel


def run_stage(args, device):
    B, sel = solve_stage(args)
    split = f"_split{args.split_seed}" if args.split_seed else ""
    out = LQ_DIR / f"{args.target}_rho{args.rho:g}_a{args.alpha:g}{split}"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "selected": {m: [int(i) for i in np.flatnonzero(x)] for m, x in sel.items()},
               "predicted": {m: predicted(B["q"], x) for m, x in sel.items()}}, open(out / "selection.json", "w"),
              indent=1)
    _, data = load_features(device, need_eval_data=True)
    cvc = load_cvc(device)
    guard = Guard()
    for s in args.seeds:
        for m in args.solvers:
            f = out / "runs" / f"lq_{m}_s{s}.npz"
            if f.exists():
                continue
            kv, cv, dt = finetune_eval(np.flatnonzero(sel[m]), s, args, data, cvc, device, guard)
            np.savez(f, kv=kv, cvc=cv)
            g = gains(kv, cv, B["S"]["masks"], B["base"], "rep")
            print(f"lq_{m:6s} s={s} " + " ".join(f"rep_{a}={b:+.4f}" for a, b in g.items()) + f" ({dt:.0f}s)",
                  flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, nargs=2)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--solvers", nargs="+", default=["greedy", "sa"], choices=SOLVERS)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", default="composite", choices=["composite"] + METRICS)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sa-evals", type=int, default=300000)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()
    if args.fit:
        fit_stage(args)
    elif args.solve:
        solve_stage(args)
    elif args.sample or args.run:
        device = get_device()
        sample_stage(args, device) if args.sample else run_stage(args, device)


if __name__ == "__main__":
    main()
