#!/usr/bin/env python3
"""
hardware_kernel_check.py
========================
Secondary experiment: how well do noisy simulation and IBM quantum hardware
reproduce the exact fidelity kernel, for the entangled ZZ circuit and its
no-entanglement ablation?

This is a kernel-fidelity check on a small fixed subset, not a classification
benchmark. With about 20 frames, AUROC is meaningless, so we report how far each
estimated kernel deviates from the exact one and how much class structure
(kernel-target alignment) survives the noise.

Circuits are identical to failure_quantum_benchmark.py: H, Rz(2 x_i) and, in
the entangled map only, CX - Rz(2 (pi - x_i)(pi - x_j)) - CX on neighbours,
repeated `reps` times. Kernel entry k(x, x') = P(0...0) of U(x') U(x)^dagger
(compute-uncompute). A barrier between the two halves keeps the transpiler
from cancelling them. Both maps use the same physical qubits, shots,
transpiler seed and optimisation level.

Modes (run them in this order)
  exact      numpy only: build the subset and exact kernels
  verify     Qiskit statevector kernels must equal the numpy kernels, and a
             noiseless Aer run must agree within shot noise. Stop here if it fails.
  noisy-sim  Aer with the noise model of an IBM fake backend (fallback that
             always works)
  hardware   IBM Runtime SamplerV2 on a real backend. The job id is saved
             right away, so `fetch` can collect results later
  fetch      collect a submitted hardware job by --job-id
  report     rebuild report/figure from whatever results exist

Typical use
  python hardware_kernel_check.py --features failure_detection/features_cvc.csv \
      --results-dir failure_detection/results_hm --out failure_detection/hardware --mode verify
  ... --mode noisy-sim
  ... --mode hardware --pilot          # 10 circuits per map, to measure QPU usage
  ... --mode hardware                  # full run
  ... --mode fetch --job-id <id>

IBM credentials: save once on your own machine with
  QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...")
and never paste the token into chats or code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_quantum_benchmark import (entangling_pairs, feature_map_states,  # noqa: E402
                                       fidelity_kernel)

ALPHA_GRID = (0.1, 0.25, 0.5, 1.0)  # same order as model_grid in failure_quantum_benchmark.py
C_GRID = (1.0, 10.0)
MAPS = ("entangled", "product")


# --------------------------------------------------------------------------- #
# Subset, angles, exact kernels (numpy only)
# --------------------------------------------------------------------------- #
def pick_alpha(results_dir: Path | None, threshold: float, default: float) -> tuple[float, str]:
    """Bandwidth most often selected for q_entangled in the main run (fixed rule, decided in advance)."""
    if results_dir and (results_dir / "diagnostics.csv").exists():
        d = pd.read_csv(results_dir / "diagnostics.csv")
        d = d[(d.model == "q_entangled") & (np.isclose(d.threshold, threshold))]
        if len(d):
            k = int(d.chosen_config.mode().iloc[0])
            return ALPHA_GRID[k // len(C_GRID)], f"most-selected alpha for q_entangled in {results_dir}"
    return default, "default (no main-run diagnostics found)"


def build_subset(args):
    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    X = np.nan_to_num(df[cols].to_numpy(np.float64))
    # Preprocessing fit on all frames: no classifier is trained here, so no leakage issue.
    Z = PCA(n_components=args.n_qubits, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    lo, span = Z.min(0), np.maximum(Z.max(0) - Z.min(0), 1e-12)
    U = (Z - lo) / span
    fail = (df.dice.to_numpy() < args.threshold)
    rng = np.random.default_rng(args.subset_seed)
    k = args.subset_size // 2
    idx = np.concatenate([rng.choice(np.flatnonzero(fail), k, replace=False),
                          rng.choice(np.flatnonzero(~fail), args.subset_size - k, replace=False)])
    alpha, why = pick_alpha(args.results_dir, args.threshold, args.alpha)
    angles = alpha * np.pi * U[idx]
    sub = pd.DataFrame({"sample_id": df.sample_id.to_numpy()[idx], "sequence_id": df.sequence_id.to_numpy()[idx],
                        "dice": df.dice.to_numpy()[idx], "failure": fail[idx].astype(int)})
    return sub, angles, alpha, why


def exact_kernels(angles, reps, topology):
    pairs = entangling_pairs(angles.shape[1], topology)
    return {m: fidelity_kernel(*(2 * [feature_map_states(angles, reps, pairs, m == "entangled")]))
            for m in MAPS}


def pair_list(m: int, pilot: bool):
    pl = [(i, j) for i in range(m) for j in range(i, m)]
    return pl[:10] if pilot else pl


# --------------------------------------------------------------------------- #
# Qiskit parts (imported lazily)
# --------------------------------------------------------------------------- #
def feature_map_circuit(x, reps, pairs, entangle):
    from qiskit import QuantumCircuit
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for q in range(n):
            qc.rz(2 * x[q], q)
        if entangle:
            for i, j in pairs:
                qc.cx(i, j)
                qc.rz(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def kernel_circuits(angles, plist, reps, topology, entangle):
    pairs = entangling_pairs(angles.shape[1], topology)
    circs = []
    for i, j in plist:
        qc = feature_map_circuit(angles[i], reps, pairs, entangle)
        qc.barrier()
        qc.compose(feature_map_circuit(angles[j], reps, pairs, entangle).inverse(), inplace=True)
        qc.measure_all()
        circs.append(qc)
    return circs


def p_zero(counts: dict, n: int, shots: int) -> float:
    return counts.get("0" * n, 0) / shots


def to_matrix(m, plist, vals):
    K = np.full((m, m), np.nan)
    for (i, j), v in zip(plist, vals):
        K[i, j] = K[j, i] = v
    return K


def fixed_layout_pm(backend, circ_entangled, seed):
    """Choose physical qubits once (from an entangled circuit) and reuse them for both maps."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm0 = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=seed)
    layout = pm0.run(circ_entangled).layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=layout,
                                      seed_transpiler=seed)
    return pm, list(layout)


def two_qubit_count(circ) -> int:
    ops = circ.count_ops()
    return int(sum(ops.get(g, 0) for g in ("cx", "ecr", "cz", "rzz")))


def run_verify(angles, Kex, args, out):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    m, n = angles.shape
    pairs = entangling_pairs(n, args.topology)
    ok = True
    sim = AerSimulator(seed_simulator=1)
    res = {}
    for mp in MAPS:
        ent = mp == "entangled"
        states = [Statevector(feature_map_circuit(x, args.reps, pairs, ent)).data for x in angles]
        Ksv = np.abs(np.conj(np.array(states)) @ np.array(states).T) ** 2
        err_sv = float(np.abs(Ksv - Kex[mp]).max())
        plist = pair_list(m, pilot=False)
        circs = transpile(kernel_circuits(angles, plist, args.reps, args.topology, ent), sim, seed_transpiler=7)
        counts = sim.run(circs, shots=args.shots).result().get_counts()
        vals = [p_zero(c, n, args.shots) for c in counts]
        K = to_matrix(m, plist, vals)
        z = np.abs(K - Kex[mp]) / np.sqrt(np.clip(Kex[mp] * (1 - Kex[mp]), 1e-4, None) / args.shots)
        frac_ok = float((z < 4).mean())
        res[mp] = dict(max_abs_err_statevector=err_sv, noiseless_mae=float(np.nanmean(np.abs(K - Kex[mp]))),
                       frac_within_4sigma=frac_ok)
        sv_ok, shot_ok = err_sv < 1e-9, frac_ok > 0.99
        ok &= sv_ok and shot_ok
        print(f"[verify] {mp}: Qiskit statevector vs numpy max error {err_sv:.2e} ({'PASS' if sv_ok else 'FAIL'}); "
              f"noiseless Aer within shot noise for {frac_ok:.1%} of entries ({'PASS' if shot_ok else 'FAIL'})")
        np.save(out / f"K_noiseless_{mp}.npy", K)
    res["passed"] = bool(ok)
    (out / "verify.json").write_text(json.dumps(res, indent=2))
    print("VERIFY", "PASSED" if ok else "FAILED")
    return ok


def get_fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider
    for nm in ([name] if name else []) + ["FakeTorino", "FakeSherbrooke", "FakeBrisbane"]:
        if hasattr(fake_provider, nm):
            return getattr(fake_provider, nm)()
    sys.exit("No IBM fake backend available in qiskit_ibm_runtime.fake_provider")


def run_noisy_sim(angles, args, out):
    from qiskit_aer import AerSimulator
    fake = get_fake_backend(args.fake_backend)
    sim = AerSimulator.from_backend(fake, seed_simulator=1)
    m, n = angles.shape
    plist = pair_list(m, args.pilot)
    meta = {"backend": f"noisy simulator ({fake.name})", "shots": args.shots, "pilot": args.pilot}
    circs = {mp: kernel_circuits(angles, plist, args.reps, args.topology, mp == "entangled") for mp in MAPS}
    pm, layout = fixed_layout_pm(sim, circs["entangled"][0], args.transpile_seed)
    meta["physical_qubits"] = layout
    for mp in MAPS:
        isa = pm.run(circs[mp])
        meta[f"two_qubit_gates_{mp}"] = int(np.median([two_qubit_count(c) for c in isa]))
        meta[f"depth_{mp}"] = int(np.median([c.depth() for c in isa]))
        counts = sim.run(isa, shots=args.shots).result().get_counts()
        counts = counts if isinstance(counts, list) else [counts]
        np.save(out / f"K_noisysim_{mp}.npy", to_matrix(m, plist, [p_zero(c, n, args.shots) for c in counts]))
    (out / "noisysim_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[noisy-sim] done on {fake.name}: 2q gates entangled={meta['two_qubit_gates_entangled']}, "
          f"product={meta['two_qubit_gates_product']}")


def runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService()
    except Exception as e:
        sys.exit(f"Could not load a saved IBM Quantum account ({e}). Save it once with "
                 "QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=..., instance=...)")


def make_sampler(backend):
    from qiskit_ibm_runtime import SamplerV2
    try:
        return SamplerV2(mode=backend)
    except TypeError:  # older qiskit-ibm-runtime
        return SamplerV2(backend=backend)


def run_hardware(angles, args, out):
    service = runtime_service()
    m, n = angles.shape
    backend = (service.backend(args.backend) if args.backend else
               service.least_busy(operational=True, simulator=False, min_num_qubits=n))
    plist = pair_list(m, args.pilot)
    circs = {mp: kernel_circuits(angles, plist, args.reps, args.topology, mp == "entangled") for mp in MAPS}
    pm, layout = fixed_layout_pm(backend, circs["entangled"][0], args.transpile_seed)
    isa = {mp: pm.run(circs[mp]) for mp in MAPS}
    all_isa = isa["entangled"] + isa["product"]
    job = make_sampler(backend).run(all_isa, shots=args.shots)
    meta = {"backend": backend.name, "job_id": job.job_id(), "shots": args.shots, "pilot": args.pilot,
            "n_pairs_per_map": len(plist), "physical_qubits": layout,
            "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for mp in MAPS:
        meta[f"two_qubit_gates_{mp}"] = int(np.median([two_qubit_count(c) for c in isa[mp]]))
        meta[f"depth_{mp}"] = int(np.median([c.depth() for c in isa[mp]]))
    (out / "hardware_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[hardware] submitted job {job.job_id()} to {backend.name} "
          f"({len(all_isa)} circuits x {args.shots} shots). Saved to hardware_meta.json.")
    print("[hardware] waiting for results (Ctrl-C is safe; later run --mode fetch --job-id ...)")
    collect_hardware(job, meta, m, n, plist, out)


def collect_hardware(job, meta, m, n, plist, out):
    result = job.result()
    vals = []
    for pub in result:
        reg = getattr(pub.data, "meas", None) or next(iter(pub.data.values()))
        vals.append(p_zero(reg.get_counts(), n, meta["shots"]))
    k = len(plist)
    np.save(out / "K_hardware_entangled.npy", to_matrix(m, plist, vals[:k]))
    np.save(out / "K_hardware_product.npy", to_matrix(m, plist, vals[k:2 * k]))
    try:
        meta["quantum_seconds"] = job.metrics()["usage"]["quantum_seconds"]
    except Exception:
        pass
    meta["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "hardware_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[hardware] results saved. QPU usage: {meta.get('quantum_seconds', 'see IBM dashboard')} s")


def run_fetch(args, out, m, n):
    meta = json.loads((out / "hardware_meta.json").read_text())
    job_id = args.job_id or meta["job_id"]
    job = runtime_service().job(job_id)
    collect_hardware(job, meta, m, n, pair_list(m, meta.get("pilot", False)), out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def kta(K, y):
    """Centered kernel-target alignment (Cortes et al. 2012): how much class structure the kernel carries."""
    m = len(y)
    H = np.eye(m) - 1.0 / m
    Kc = H @ K @ H
    t = np.where(y == 1, 1.0, -1.0)
    Y = np.outer(t, t)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-12))


def report(out, sub, Kex, meta_note):
    y = sub.failure.to_numpy()
    rows = []
    for mp in MAPS:
        rows.append(dict(map=mp, source="exact", mae_vs_exact=0.0, max_err=0.0, pearson_r=1.0,
                         mean_diag=float(np.mean(np.diag(Kex[mp]))),
                         mean_offdiag=float(Kex[mp][~np.eye(len(y), dtype=bool)].mean()),
                         kta=kta(Kex[mp], y), n_entries=int(len(y) * (len(y) + 1) / 2)))
        for src in ("noiseless", "noisysim", "hardware"):
            f = out / f"K_{src}_{mp}.npy"
            if not f.exists():
                continue
            K = np.load(f)
            iu = np.triu_indices(len(y))
            have = ~np.isnan(K[iu])
            a, b = K[iu][have], Kex[mp][iu][have]
            full = not np.isnan(K).any()
            off = ~np.eye(len(y), dtype=bool)
            rows.append(dict(map=mp, source=src, mae_vs_exact=float(np.abs(a - b).mean()),
                             max_err=float(np.abs(a - b).max()),
                             pearson_r=float(np.corrcoef(a, b)[0, 1]) if have.sum() > 2 else np.nan,
                             mean_diag=float(np.nanmean(np.diag(K))),
                             mean_offdiag=float(np.nanmean(K[off])),
                             kta=kta(K, y) if full else np.nan, n_entries=int(have.sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "kernel_fidelity_summary.csv", index=False)
    sub.to_csv(out / "subset.csv", index=False)
    metas = {p.stem: json.loads(p.read_text()) for p in out.glob("*_meta.json")}
    try:
        table_txt = tab.to_markdown(index=False, floatfmt=".3f")
    except Exception:  # tabulate not installed
        table_txt = "```\n" + tab.to_string(index=False) + "\n```"
    L = ["# Hardware / noisy-simulation kernel check", "",
         f"Subset: {len(sub)} CVC frames ({int(sub.failure.sum())} failures, {int((1 - sub.failure).sum())} non-failures), "
         f"{meta_note}", "",
         "mean_diag should be 1 (k(x,x)=1); its drop measures noise directly. KTA = how much failure/non-failure "
         "structure the kernel carries (exact = reference).", "",
         table_txt, ""]
    for k, v in metas.items():
        L += [f"**{k}**: " + ", ".join(f"{a}={b}" for a, b in v.items()), ""]
    L += ["Interpretation: compare the entangled and product rows for the same source. The entangled circuit has "
          "many more two-qubit gates after compute-uncompute, so a larger error is expected and is a finding about "
          "the noise cost of entanglement, not a bug. This subset is too small for classification claims."]
    (out / "HARDWARE_REPORT.md").write_text("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        srcs = [s for s in ("noisysim", "hardware") if (out / f"K_{s}_entangled.npy").exists()]
        if srcs:
            fig, axes = plt.subplots(1, len(srcs), figsize=(4.2 * len(srcs), 4), squeeze=False)
            for ax, s in zip(axes[0], srcs):
                for mp, col in (("entangled", "#d62728"), ("product", "#ff7f0e")):
                    K = np.load(out / f"K_{s}_{mp}.npy")
                    iu = np.triu_indices(len(y))
                    have = ~np.isnan(K[iu])
                    ax.scatter(Kex[mp][iu][have], K[iu][have], s=8, alpha=0.6, color=col, label=mp)
                ax.plot([0, 1], [0, 1], "k:", lw=0.8)
                ax.set(xlabel="exact kernel value", ylabel=f"{s} estimate", xlim=(0, 1.02), ylim=(0, 1.02),
                       title={"noisysim": "Noisy simulation", "hardware": "IBM hardware"}[s])
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "kernel_scatter.png", dpi=200)
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"(figure skipped: {e})")
    print(tab.to_string(index=False))
    print(f"Report: {out / 'HARDWARE_REPORT.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=None, help="Main-run output (to read the chosen alpha)")
    ap.add_argument("--out", type=Path, default=Path("failure_detection/hardware"))
    ap.add_argument("--mode", choices=["exact", "verify", "noisy-sim", "hardware", "fetch", "report"], required=True)
    ap.add_argument("--feature-prefix", default="f_hm_")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--subset-size", type=int, default=20)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--alpha", type=float, default=0.5, help="Used only if no main-run diagnostics are found")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--transpile-seed", type=int, default=7)
    ap.add_argument("--fake-backend", default="FakeTorino")
    ap.add_argument("--backend", default=None, help="IBM backend name; default = least busy")
    ap.add_argument("--pilot", action="store_true", help="Only 10 circuits per map, to measure QPU usage")
    ap.add_argument("--job-id", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sub, angles, alpha, why = build_subset(args)
    Kex = exact_kernels(angles, args.reps, args.topology)
    for mp in MAPS:
        np.save(args.out / f"K_exact_{mp}.npy", Kex[mp])
    note = (f"alpha={alpha} ({why}), {args.n_qubits} qubits, reps={args.reps}, {args.topology} entanglement, "
            f"shots={args.shots}, subset seed {args.subset_seed}.")
    print(note)
    m, n = angles.shape
    if args.mode == "verify" and not run_verify(angles, Kex, args, args.out):
        sys.exit("Verification failed: do not use hardware time until this passes.")
    if args.mode == "noisy-sim":
        run_noisy_sim(angles, args, args.out)
    if args.mode == "hardware":
        vf = args.out / "verify.json"
        if not vf.exists() or not json.loads(vf.read_text()).get("passed"):
            sys.exit("Run --mode verify first; it must PASS before spending QPU time.")
        run_hardware(angles, args, args.out)
    if args.mode == "fetch":
        run_fetch(args, args.out, m, n)
    report(args.out, sub, Kex, note)


if __name__ == "__main__":
    main()
