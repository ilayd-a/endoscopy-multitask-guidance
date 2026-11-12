"""
qaoa_block_selection.py
Frame selection for UNet fine-tuning with a 10-qubit QAOA circuit (gate model, Qiskit).

The learned QUBO of learned_qubo_selection.py has 800 binary variables. Here it is
coarse-grained to 10 variables so that it fits a 10-qubit circuit:

    pool    the 200 train frames with the highest loss (hardest_k is its top half)
    blocks  10 blocks of 20 frames, balanced k-means on the frame features phi
    choose  5 of the 10 blocks = 100 frames, one qubit per block

The objective is the same learned surrogate (predicted CVC select-half hard Dice
gain) with the same trust region, applied to block mean features; rho is set by
the Mahalanobis rule (smallest rho whose exact optimum stays inside the 95th
percentile of the sampled subsets). The cardinality constraint enters the circuit
as a penalty A (sum x - 5)^2, with A the smallest value for which the feasible
optimum is the global optimum. QAOA: |+>^10, p layers of RZ/RZZ cost and RX mixer,
angles optimised on the exact statevector of the circuit, then sampled with shots.
The selection is the best feasible sampled bitstring.

With 10 qubits there are 252 feasible subsets, so brute force gives the exact
optimum instantly; no quantum advantage is possible or claimed. The circuit runs
on the Aer simulator; --hardware submits the same circuit to IBM Quantum (the
account must already be saved with QiskitRuntimeService.save_account).

Stages:
    --design         pool, blocks, rho, exhaustive landscape, greedy (no fine-tuning)
    --circuit        optimise and sample the QAOA circuit, save the selection
    --hardware       run the saved circuit on an IBM backend, save its selection
    --run            fine-tune the selected frames (--seeds), report halves untouched
    --report         gains on report halves vs the classical baselines

Usage (from repo root):
    S=quantum-medical-research/frame_selection/qaoa_block_selection.py
    ./venv/bin/python $S --circuit --split-seed 1
    ./venv/bin/python $S --run --split-seed 1 --seeds 20 21 22
    ./venv/bin/python $S --report --split-seed 1
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learned_qubo_selection as L  # noqa: E402

QA_DIR = L.OUT_DIR / "qaoa_blocks"
N_BLOCKS, BLOCK_SIZE, K_BLOCKS = 10, 20, 5
BASELINES = ["hardest_k", "all_800", "random_k", "kcenter", "craig", "align_topk", "gradmatch_omp", "qubo_div"]


def base_args(split_seed):
    return SimpleNamespace(split_seed=split_seed, k=100, target="cvc_hard", alpha=10.0, rho=0.0, steps=150,
                           lr=3e-5, batch=8)


# ---------------------------------------------------------------------------
# 10-variable instance
# ---------------------------------------------------------------------------
def balanced_blocks(X, n_blocks, size, seed=0, iters=100):
    """Equal-size k-means: assignment step is a linear assignment to `size` slots per centre."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), n_blocks, replace=False)]
    for _ in range(iters):
        D = ((X[:, None] - C[None]) ** 2).sum(-1)
        r, c = linear_sum_assignment(np.repeat(D, size, axis=1))
        lab = np.empty(len(X), int)
        lab[r] = c // size
        Cn = np.array([X[lab == b].mean(0) for b in range(n_blocks)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return lab


def instance(split_seed):
    """Pool, blocks, block QUBO and its exhaustive landscape."""
    B = L.build(base_args(split_seed))
    phi, S = B["phi"], B["S"]
    pool = np.argsort(-S["loss"], kind="stable")[:N_BLOCKS * BLOCK_SIZE]
    lab = balanced_blocks(phi[pool], N_BLOCKS, BLOCK_SIZE)
    order = np.argsort([-phi[pool[lab == b], 0].mean() for b in range(N_BLOCKS)])  # block 0 = highest loss
    blocks = [pool[lab == b] for b in order]
    phi_b = np.array([phi[b].mean(0) for b in blocks])
    combos = np.array(list(itertools.combinations(range(N_BLOCKS), K_BLOCKS)))
    X = np.zeros((len(combos), N_BLOCKS))
    X[np.arange(len(combos))[:, None], combos] = 1
    Mc = X @ phi_b / K_BLOCKS
    maha = np.einsum("na,ab,nb->n", Mc - B["mu"], B["P"], Mc - B["mu"])
    lim = np.percentile(B["maha"], 95)
    for rho in [0.0] + list(np.geomspace(1e-5, 1.0, 51)):
        q = L.learned_qubo(B["dm"], phi_b, K_BLOCKS, rho, B["mu"], B["P"])
        val = np.array([L.predicted(q, x) for x in X])
        if maha[val.argmax()] <= lim:
            break
    return {"B": B, "pool": pool, "blocks": blocks, "phi_b": phi_b, "X": X, "val": val, "maha": maha, "lim": lim,
            "rho": rho, "q": q}


def frames(I, x):
    return np.sort(np.concatenate([I["blocks"][b] for b in np.flatnonzero(x)]))


def out_dir(args):
    return QA_DIR / (f"split{args.split_seed}" + ("_blk" if args.objective == "block" else ""))


# ---------------------------------------------------------------------------
# Block-level objective measured directly: fine-tune random 5-block subsets, fit a 10-variable QUBO
# ---------------------------------------------------------------------------
def sample_blocks_stage(args):
    """Fine-tune random 5-of-10 block subsets (1 seed each); outcomes are stored per frame, so they serve every split."""
    blocks = instance(0)["blocks"]
    (QA_DIR / "block_samples").mkdir(parents=True, exist_ok=True)
    device = L.get_device()
    _, data = L.load_features(device, need_eval_data=True)
    cvc = L.load_cvc(device)
    guard = L.Guard()
    for n in range(*args.sample_blocks):
        f = QA_DIR / "block_samples" / f"b{n:03d}.npz"
        if f.exists():
            continue
        b = np.sort(np.random.default_rng(7000 + n).choice(N_BLOCKS, K_BLOCKS, replace=False))
        kv, cv, dt = L.finetune_eval(np.sort(np.concatenate([blocks[i] for i in b])), 3000 + n, base_args(0), data,
                                     cvc, device, guard)
        np.savez(f, blocks=b, kv=kv, cvc=cv)
        print(f"b{n:03d} ({dt:.0f}s)", flush=True)
    print("DONE", flush=True)


def block_design(X, pairwise):
    iu = np.triu_indices(X.shape[1], 1)
    return np.column_stack([X, X[:, iu[0]] * X[:, iu[1]]]) if pairwise else X


def ridge(D, y, alpha):
    mu, sd = D.mean(0), D.std(0) + 1e-12
    Z = (D - mu) / sd
    c = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean())) / sd
    return c, y.mean() - c @ mu


def block_instance(split_seed):
    """10-variable QUBO fit on the select-half CVC hard gain of the sampled block subsets (report halves unused)."""
    I = instance(split_seed)
    m = I["B"]["S"]["masks"]
    mk = m["cvc_sel"] & m["cvc_hard"]
    base = I["B"]["base"]["cvc"][mk].mean()
    rows = [(np.load(f)["blocks"], np.load(f)["cvc"]) for f in sorted((QA_DIR / "block_samples").glob("b*.npz"))]
    for f in sorted((QA_DIR / "shared").glob("pool_random_s*.npz")):  # same pool, random blocks, seed in the name
        s = int(f.stem.rsplit("_s", 1)[1])
        rows.append((np.random.default_rng(500 + s).choice(N_BLOCKS, K_BLOCKS, replace=False), np.load(f)["cvc"]))
    Xs = np.zeros((len(rows), N_BLOCKS))
    for i, (b, _) in enumerate(rows):
        Xs[i, b] = 1
    y = np.array([cv[mk].mean() - base for _, cv in rows])
    best = None
    for pairwise in (False, True):
        D = block_design(Xs, pairwise)
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            pred = np.zeros(len(y))
            for i in range(len(y)):
                tr = np.arange(len(y)) != i
                c, b0 = ridge(D[tr], y[tr], alpha)
                pred[i] = D[i] @ c + b0
            r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"  block model pairwise={pairwise} alpha={alpha:g}: LOO R2={r2:+.3f} "
                  f"spearman={stats.spearmanr(pred, y)[0]:+.3f}")
            if best is None or r2 > best[0]:
                best = (r2, pairwise, alpha)
    r2, pairwise, alpha = best
    c, b0 = ridge(block_design(Xs, pairwise), y, alpha)
    lin, Q = c[:N_BLOCKS], np.zeros((N_BLOCKS, N_BLOCKS))
    if pairwise:
        Q[np.triu_indices(N_BLOCKS, 1)] = -c[N_BLOCKS:]
        Q = Q + Q.T
    s = lin.std()
    q = L.Qubo(lin / s, np.zeros(N_BLOCKS), Q / s, 1.0, 0.0, 1.0)
    q.const, q.scale = b0, s
    val = np.array([L.predicted(q, x) for x in I["X"]])
    g = L.greedy_select(q, K_BLOCKS)
    print(f"split {split_seed}: {len(y)} sampled block subsets, y mean {y.mean():+.4f} sd {y.std():.4f}; chosen "
          f"pairwise={pairwise} alpha={alpha:g} (LOO R2 {r2:+.3f}); optimum blocks "
          f"{np.flatnonzero(I['X'][val.argmax()]).tolist()} pred {val.max():+.4f}; greedy rank "
          f"{int((val > L.predicted(q, g) + 1e-12).sum()) + 1}; best observed {y.max():+.4f}")
    return {**I, "q": q, "val": val, "rho": 0.0, "pairwise": pairwise, "loo_r2": r2}


def design_stage(args):
    I = instance(args.split_seed)
    q, X, val = I["q"], I["X"], I["val"]
    S = I["B"]["S"]
    hard100 = set(np.argsort(-S["loss"], kind="stable")[:100])
    print(f"split {args.split_seed}: pool {len(I['pool'])} frames, rho={I['rho']:.3g}, maha limit {I['lim']:.1f}; "
          f"feasible subsets inside limit: {(I['maha'] <= I['lim']).sum()}/{len(X)}")
    print("block  loss_z  n_in_hardest100  lin")
    for b, fr in enumerate(I["blocks"]):
        print(f"  {b}    {I['phi_b'][b, 0]:+.2f}   {len(hard100 & set(fr)):3d}            {q.lin[b]:+.2f}")
    order = np.argsort(-val)
    g = L.greedy_select(q, K_BLOCKS)
    ls = L.local_search(q, g)
    rank = {n: int(np.flatnonzero((X[order] == x).all(1))[0]) + 1 for n, x in (("greedy", g), ("greedy+LS", ls))}
    print(f"predicted gain over the 252 subsets: best {val.max():+.4f}, median {np.median(val):+.4f}, "
          f"worst {val.min():+.4f}; second best {val[order[1]]:+.4f}")
    print(f"optimum blocks {np.flatnonzero(X[order[0]]).tolist()} maha {I['maha'][order[0]]:.1f}; "
          f"greedy blocks {np.flatnonzero(g).tolist()} rank {rank['greedy']}; greedy+LS rank {rank['greedy+LS']}")
    opt = frames(I, X[order[0]])
    print(f"optimum: {len(hard100 & set(opt))}/100 frames shared with hardest_k")
    return I


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------
def states(n):
    """All 2^n bitstrings; qubit i is bit i of the index (Qiskit little-endian)."""
    return (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1


def min_penalty(q, k):
    """Smallest A (grid 0.1) for which the best feasible subset maximises f(x) - A (sum x - k)^2 over all x."""
    xs = states(len(q.lin))
    f = xs @ q.lin - 0.5 * np.einsum("ni,ij,nj->n", xs, q.Q, xs)
    feas = xs.sum(1) == k
    opt = int(np.argmax(np.where(feas, f, -np.inf)))
    for a in np.arange(0.1, 100, 0.1):
        if np.argmax(f - a * (xs.sum(1) - k) ** 2) == opt:
            return float(a), xs, f, feas, opt
    raise ValueError("no penalty found")


def ising(q, k, penalty):
    """Cost C(x) = -(lin.x - 1/2 x^T Q x) + A (sum x - k)^2 as h, J with x = (1 - z) / 2 (constant dropped)."""
    a = -q.lin + penalty * (1 - 2 * k)
    b = q.Q + 2 * penalty
    np.fill_diagonal(b, 0.0)
    return -a / 2 - b.sum(1) / 4, b / 4


def qaoa_circuit(h, J, p):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n = len(h)
    gam, bet = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            qc.rz(2 * gam[layer] * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * gam[layer] * J[i, j], i, j)
        for i in range(n):
            qc.rx(2 * bet[layer], i)
    return qc, gam, bet


def sample_summary(counts, shots, f, feas, opt, xs):
    idx = np.array([int(b, 2) for b in counts])
    cnt = np.array(list(counts.values()))
    ok = feas[idx]
    pick = int(idx[ok][np.argmax(f[idx[ok]])])
    mode = int(idx[ok][np.argmax(cnt[ok])])
    return {"p_feasible": float(cnt[ok].sum() / shots), "p_opt": float(cnt[idx == opt].sum() / shots),
            "n_opt": int(cnt[idx == opt].sum()), "blocks": np.flatnonzero(xs[pick]).tolist(),
            "rank": int((f[feas] > f[pick]).sum()) + 1, "mode_blocks": np.flatnonzero(xs[mode]).tolist(),
            "mode_rank": int((f[feas] > f[mode]).sum()) + 1}


def circuit_stage(args):
    from qiskit import qasm3, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    I = block_instance(args.split_seed) if args.objective == "block" else design_stage(args)
    q, n, k = I["q"], N_BLOCKS, K_BLOCKS
    penalty, xs, f, feas, opt = min_penalty(q, k)
    C = -f + penalty * (xs.sum(1) - k) ** 2
    s = C.std()
    h, J = ising(q, k, penalty)
    qc, gam, bet = qaoa_circuit(h / s, J / s, args.p)

    def bind(theta):
        return qc.assign_parameters({**dict(zip(gam, theta[:args.p])), **dict(zip(bet, theta[args.p:]))})

    def probs(theta):
        return Statevector(bind(theta)).probabilities()

    rng = np.random.default_rng(0)
    best = None
    ramp = np.concatenate([np.linspace(0.1, 0.6, args.p), np.linspace(0.6, 0.1, args.p)])
    for start in [ramp] + [rng.uniform(0, 0.8, 2 * args.p) for _ in range(args.starts - 1)]:
        res = minimize(lambda t: probs(t) @ C / s, start, method="COBYLA", options={"maxiter": 600, "rhobeg": 0.2})
        if best is None or res.fun < best.fun:
            best = res
    pr = probs(best.x)
    print(f"\nQAOA p={args.p}: {n} qubits, {qc.count_ops().get('rzz', 0)} RZZ gates; penalty A={penalty:.1f}")
    print(f"exact statevector: P(optimum)={pr[opt]:.4f} (uniform over all states {1 / 2 ** n:.4f}, over feasible "
          f"{1 / feas.sum():.4f}), P(feasible)={pr[feas].sum():.3f} (uniform {feas.mean():.3f}), "
          f"P(optimum | feasible)={pr[opt] / pr[feas].sum():.4f}")

    bound = bind(best.x)
    bound.measure_all()
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(3e-4, 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(8e-3, 2), ["cx"])
    noise.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.02, 0.98]]))
    tq = transpile(bound, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=0)
    out = {"split_seed": args.split_seed, "rho": float(I["rho"]), "p": args.p, "theta": best.x.tolist(),
           "penalty": penalty, "scale": float(s), "p_opt_exact": float(pr[opt]),
           "p_feasible_exact": float(pr[feas].sum()), "optimum_blocks": np.flatnonzero(xs[opt]).tolist(),
           "depth": tq.depth(), "cx": int(tq.count_ops().get("cx", 0)), "shots": args.shots}
    sel = {}
    for name, sim in (("ideal", AerSimulator()), ("noisy", AerSimulator(noise_model=noise))):
        counts = sim.run(tq, shots=args.shots, seed_simulator=args.split_seed).result().get_counts()
        out[name] = sample_summary(counts, args.shots, f, feas, opt, xs)
        sel[name] = xs[int(sum(2 ** b for b in out[name]["blocks"]))].astype(float)
        print(f"{name:5s} sampler, {args.shots} shots (depth {tq.depth()}, {out['cx']} CX): {out[name]}")
    g = L.greedy_select(q, k)
    out["selected"] = {"qaoa": [int(i) for i in frames(I, sel["ideal"])],
                       "qaoa_noisy": [int(i) for i in frames(I, sel["noisy"])],
                       "block_greedy": [int(i) for i in frames(I, g)]}
    out["predicted"] = {"qaoa": L.predicted(q, sel["ideal"]), "qaoa_noisy": L.predicted(q, sel["noisy"]),
                        "block_greedy": L.predicted(q, g)}
    out["blocks"] = [[int(i) for i in b] for b in I["blocks"]]
    d = out_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{args.out_name}.json", "w"), indent=1)
    (d / f"{args.out_name}_circuit.qasm").write_text(qasm3.dumps(bound))
    return out


def hardware_stage(args):
    """Run the circuit saved by --circuit on IBM Quantum; needs qiskit-ibm-runtime and a saved account."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    d = out_dir(args)
    out = json.load(open(d / f"{args.out_name}.json"))
    I = instance(args.split_seed)
    _, xs, f, feas, opt = min_penalty(I["q"], K_BLOCKS)
    h, J = ising(I["q"], K_BLOCKS, out["penalty"])
    qc, gam, bet = qaoa_circuit(h / out["scale"], J / out["scale"], out["p"])
    bound = qc.assign_parameters({**dict(zip(gam, out["theta"][:out["p"]])), **dict(zip(bet, out["theta"][out["p"]:]))})
    bound.measure_all()
    if args.dry:  # transpile for a generic heavy-hex-like line and stop; no account needed
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        t = transpile(bound, basis_gates=["rz", "sx", "x", "cz"], coupling_map=CouplingMap.from_line(N_BLOCKS),
                      optimization_level=3, seed_transpiler=0)
        print(f"dry run: p={out['p']}, line connectivity: depth {t.depth()}, ops {dict(t.count_ops())}")
        return
    service = QiskitRuntimeService()
    job_file = d / f"{args.out_name}_job.json"
    if job_file.exists():
        rec = json.load(open(job_file))
        job = service.job(rec["job_id"])
    else:
        backend = service.backend(args.backend) if args.backend else service.least_busy(
            operational=True, simulator=False, min_num_qubits=N_BLOCKS)
        isa = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=0).run(bound)
        rec = {"backend": backend.name, "shots": args.shots, "depth": isa.depth(),
               "ops": {k: int(v) for k, v in isa.count_ops().items()}}
        job = SamplerV2(mode=backend).run([isa], shots=args.shots)
        rec["job_id"] = job.job_id()
        json.dump(rec, open(job_file, "w"), indent=1)
        print("submitted", rec, flush=True)
    counts = job.result()[0].data.meas.get_counts()
    out["hardware"] = {**rec, **sample_summary(counts, sum(counts.values()), f, feas, opt, xs)}
    x = xs[int(sum(2 ** b for b in out["hardware"]["blocks"]))].astype(float)
    out["selected"]["qaoa_hw"] = [int(i) for i in frames(I, x)]
    out["predicted"]["qaoa_hw"] = L.predicted(I["q"], x)
    json.dump(out, open(d / f"{args.out_name}.json", "w"), indent=1)
    json.dump(counts, open(d / f"{args.out_name}_hardware_counts.json", "w"))
    print("hardware", out["hardware"])


# ---------------------------------------------------------------------------
# Fine-tuning and report
# ---------------------------------------------------------------------------
def run_stage(args):
    d = out_dir(args)
    out = json.load(open(d / "selection.json"))
    (d / "runs").mkdir(exist_ok=True)
    (QA_DIR / "shared").mkdir(exist_ok=True)
    device = L.get_device()
    _, data = L.load_features(device, need_eval_data=True)
    cvc = L.load_cvc(device)
    guard = L.Guard()
    a = base_args(args.split_seed)
    todo = {m: np.array(out["selected"][m]) for m in args.methods if m in out["selected"]}
    for m in ("block_greedy", "qaoa_noisy", "qaoa_hw"):  # identical subsets are not fine-tuned twice
        if m in out["selected"] and sorted(out["selected"][m]) == sorted(out["selected"]["qaoa"]):
            todo.pop(m, None)
    for s in args.seeds:
        runs = [(m, idx, d / "runs" / f"{m}_s{s}.npz") for m, idx in todo.items()]
        if "pool_random" in args.methods:  # 5 random blocks of the same pool; does not depend on the split
            rb = np.random.default_rng(500 + s).choice(N_BLOCKS, K_BLOCKS, replace=False)
            runs.append(("pool_random", np.sort(np.concatenate([out["blocks"][b] for b in rb])),
                         QA_DIR / "shared" / f"pool_random_s{s}.npz"))
        for m, idx, f in runs:
            if f.exists():
                continue
            kv, cv, dt = L.finetune_eval(idx, s, a, data, cvc, device, guard)
            np.savez(f, kv=kv, cvc=cv)
            print(f"{m:12s} s={s} ({dt:.0f}s)", flush=True)
    print("DONE", flush=True)


def report_stage(args):
    d = out_dir(args)
    m = L.setup(base_args(args.split_seed))["masks"]
    K = L.RUN_DIR / "k100" / "runs"
    nf = np.load(K / "no_finetune.npz")
    test = np.arange(200) >= 100
    sets = {"cvc_rep_hard": ("cvc", m["cvc_rep"] & m["cvc_hard"]), "cvc_rep_full": ("cvc", m["cvc_rep"]),
            "kv_rep_hard": ("kv", m["kv_rep"] & m["kv_hard"]), "kv_test_hard": ("kv", test & m["kv_hard"]),
            "kv_rep_full": ("kv", m["kv_rep"])}
    src = {"qaoa": d / "runs", "qaoa_noisy": d / "runs", "qaoa_hw": d / "runs", "block_greedy": d / "runs",
           "pool_random": QA_DIR / "shared", **{b: K for b in BASELINES}}
    seeds = [s for s in args.seeds if (src["qaoa"] / f"qaoa_s{s}.npz").exists()]
    ld = {n: [np.load(p / f"{n}_s{s}.npz") for s in seeds] for n, p in src.items()
          if all((p / f"{n}_s{s}.npz").exists() for s in seeds)}
    print(f"split {args.split_seed}, seeds {seeds}; gain over no fine-tuning (report halves)")
    for key, (ds, mk) in sets.items():
        print(f"{key:13s} n={mk.sum():3d} " + " ".join(
            f"{n}={np.mean([r[ds][mk].mean() for r in ld[n]]) - nf[ds][mk].mean():+.4f}" for n in ld))
    rng = np.random.default_rng(0)
    for key, (ds, mk) in sets.items():
        res = []
        for b in [n for n in ld if n != "qaoa"]:
            diff = np.array([x[ds][mk].mean() - y[ds][mk].mean() for x, y in zip(ld["qaoa"], ld[b])])
            fr = np.mean([x[ds][mk] for x in ld["qaoa"]], 0) - np.mean([y[ds][mk] for y in ld[b]], 0)
            bs = [fr[rng.integers(0, len(fr), len(fr))].mean() for _ in range(5000)]
            res.append((b, diff.mean(), stats.ttest_1samp(diff, 0).pvalue, (diff > 0).sum(),
                        np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
        ps = sorted(r[2] for r in res if r[0] in BASELINES)
        holm = max(p * (len(ps) - i) for i, p in enumerate(ps)) if ps else float("nan")
        print(f"\nqaoa - other, {key} (seed-paired p, wins; frame-bootstrap 95% CI); worst Holm p over the "
              f"{len(ps)} baselines = {holm:.3f}")
        for b, dm, p, w, lo, hi in res:
            print(f"  {b:14s} {dm:+.4f} p={p:.3f} {w}/{len(seeds)}  CI [{lo:+.4f}, {hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--circuit", action="store_true")
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out-name", default="selection")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--objective", default="surrogate", choices=["surrogate", "block"])
    ap.add_argument("--sample-blocks", type=int, nargs=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20, 32)))
    ap.add_argument("--methods", nargs="+", default=["qaoa", "qaoa_noisy", "qaoa_hw", "block_greedy", "pool_random"])
    args = ap.parse_args()
    if args.sample_blocks:
        sample_blocks_stage(args)
    elif args.design:
        design_stage(args)
    elif args.circuit:
        circuit_stage(args)
    elif args.hardware:
        hardware_stage(args)
    elif args.run:
        run_stage(args)
    elif args.report:
        report_stage(args)


if __name__ == "__main__":
    main()
