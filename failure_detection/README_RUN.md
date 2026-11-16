# CVPR failure-detection experiment: run instructions

**Question:** does a quantum circuit (and its entanglement in particular) help detect frames where the U-Net segmentation failed?

**Setup:** Kvasir-trained U-Net (`models/unet_pretrained.pth`) applied to all 612 CVC-ClinicDB frames. Failure means Dice < 0.5 (primary) or Dice < 0.2 (secondary). Every model sees the same 8 PCA features (8 qubits). The comparison is classical models vs. a quantum fidelity kernel (ZZ feature map, exact simulation) vs. **the same circuit with the entangling gates removed**.

## Protocol (frozen before the real data is seen; do not change after looking at results)

| Item | Setting |
|---|---|
| Datasets | Kvasir-SEG is the U-Net training source only. Failure detection is evaluated on CVC only: Kvasir val+test has just 8 frames with Dice < 0.5, too few to evaluate, and the train split is in-sample (decision 2026-11-14) |
| Primary endpoint | AUROC, `q_entangled` vs `q_product`, Dice < 0.5, heatmap features (`f_hm_`) |
| Training-free baselines | U-Net uncertainty used directly, with no training and a fixed direction: low max confidence, low confidence inside the mask, mean entropy, entropy inside the mask, fragmentation (number of components; an empty prediction counts as maximal) |
| Secondary | Dice < 0.2; AUPRC; all features (`f_`); quantum vs best classical; selective Dice at 10/20/30% rejection; qubit scaling (4 and 12 qubits, seeds 0–2) |
| CV | 5-fold StratifiedGroupKFold grouped by CVC sequence, seeds 0–4 |
| Leakage control | Scaler, PCA and min-max fit on the outer training fold only. Hyperparameters chosen by inner grouped 3-fold CV. GT masks used only for labels |
| Tuning budget | 8 configurations for every model |
| Statistics | mean ± SD over seeds; paired cluster bootstrap over sequences (2000 resamples) |
| Quantum | ZZ feature map, reps = 2, linear entanglement, bandwidth α ∈ {0.1, 0.25, 0.5, 1}, exact statevector simulation (not hardware) |

## Steps (run from the repository root)

```bash
# 0. Put both scripts in quantum-medical-research/endoscopy_guidance/
# 1. Environment (GPU PyTorch matching your CUDA, plus:)
pip install segmentation-models-pytorch opencv-python pandas scikit-learn scipy matplotlib

# 2. Export features (needs CVC data in dataset/dataset_b: metadata.csv, PNG/Original, PNG/Ground Truth)
python quantum-medical-research/endoscopy_guidance/export_cvc_failure_features.py \
    --cvc-root dataset/dataset_b --checkpoint models/unet_pretrained.pth \
    --official-csv eval/results/cvc_test_kvasir_to_cvc_per_sample.csv \
    --out failure_detection/features_cvc.csv

# 3. Check that the circuit simulator is correct
python quantum-medical-research/endoscopy_guidance/failure_quantum_benchmark.py --self-test

# 4. Primary run (about 10–40 min depending on CPU cores; set --jobs to your core count)
python quantum-medical-research/endoscopy_guidance/failure_quantum_benchmark.py \
    --features failure_detection/features_cvc.csv --feature-prefix f_hm_ \
    --out failure_detection/results_hm --jobs 8

# 5. Secondary run, all features
python quantum-medical-research/endoscopy_guidance/failure_quantum_benchmark.py \
    --features failure_detection/features_cvc.csv --feature-prefix f_ \
    --out failure_detection/results_all --jobs 8
```

```bash
# 6. Secondary qubit-scaling analysis (pre-specified; 8 qubits stays the primary result)
for q in 4 12; do
  python quantum-medical-research/endoscopy_guidance/failure_quantum_benchmark.py \
      --features failure_detection/features_cvc.csv --feature-prefix f_hm_ \
      --n-qubits $q --seeds 0 1 2 --no-reference \
      --out failure_detection/results_hm_q$q --jobs 8
done
```

Send back all `results_*` folders. The key files are `REPORT.md`, `summary.csv`, `comparisons.csv`, `diagnostics.csv` and the three `.png` figures.

## Secondary: IBM hardware / noisy-simulation kernel check (run after step 4)

This is a kernel-fidelity check, not a classification benchmark. It uses a fixed balanced subset of 20 CVC frames (10 with Dice < 0.5, 10 without; seed 0) and the bandwidth α most often chosen for `q_entangled` in the main run. For each frame pair it compares the exact kernel with the noisy-simulator estimate and the hardware estimate, for the entangled circuit and the product circuit. Both circuits use the same physical qubits, shots (1000) and transpiler settings. With 20 frames this gives 210 circuits per map and 420 in total.

```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime
H="python quantum-medical-research/endoscopy_guidance/hardware_kernel_check.py --features failure_detection/features_cvc.csv --results-dir failure_detection/results_hm --out failure_detection/hardware"
$H --mode verify               # MUST print VERIFY PASSED (Qiskit circuits == numpy simulator); hardware refuses to run otherwise
$H --mode noisy-sim            # fallback that always works: Aer with an IBM backend noise model (FakeTorino)
# Hardware: the account holder saves credentials once on their own machine; never share the token:
#   python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='...', instance='...')"
$H --mode hardware --pilot     # 20 circuits; check 'QPU usage' (Open Plan = 10 min per 28 days)
$H --mode hardware             # full 420 circuits; the job id is saved right away
$H --mode fetch                # if the wait was interrupted: collect the saved job later
```

The pilot and full runs write to the same folder, so copy the pilot outputs elsewhere before the full run. Deadline rule: if hardware results are not back by Sunday 8 pm ET, the paper reports only the noisy simulation, with hardware as future work. Outputs: `HARDWARE_REPORT.md`, `kernel_fidelity_summary.csv` and `kernel_scatter.png`. The Qiskit parts were written without being executed in the author's sandbox (PyPI was unreachable there), so `--mode verify` is the gate: if it errors because of a Qiskit API change, fix the API call only, never the circuit.

## Checks before anyone writes a number in the paper

1. Step 3 prints `SELF-TEST PASSED`.
2. Step 2 prints `matched 612/612`, a mean official Dice of about 0.789, a correlation of at least 0.95 and no WARNING line. Expected failure counts with official labels: **78** (Dice < 0.5) and **36** (Dice < 0.2).
3. `REPORT.md` reports 612 frames and the expected number of sequences.
4. The `u_*` rows (training-free uncertainty) appear in the table. Every learned model, quantum or classical, should be judged against the best `u_*` row as well. At least one classical model has an AUROC clearly above 0.5. If every model sits near 0.5, the features are broken.
5. Circuit diagnostics: `q_product` purity is 1.000 and `q_entangled` purity is below 1. If the off-diagonal kernel SD is close to 0, the kernel has concentrated.
6. Report every row, including unfavourable ones. Do not rerun with new settings to get a better quantum result. Any change after seeing results must be reported as post hoc.

## How results map to claims

| Outcome | What we can say |
|---|---|
| `q_entangled` > `q_product` with CI excluding 0, and ≥ best classical | Entanglement contributes on this task (simulation only; one dataset) |
| `q_entangled` ≈ `q_product` (CI includes 0) | No evidence that entanglement helps; any quantum-kernel effect is reproducible with a product-state (classically easy) circuit |
| Quantum < classical | Classical models are sufficient; the honest negative result is the contribution |
| No learned model beats the best `u_*` score | Simple U-Net confidence already detects failures as well; learned detectors, quantum or classical, add nothing |

## Claude Code prompt (paste into a fresh Claude Code window opened at the repo root)

> Run the CVPR failure-detection experiment exactly as written in `failure_detection/README_RUN.md` (or wherever this README is). Steps: set up a venv with CUDA-matched PyTorch plus the listed packages; copy `export_cvc_failure_features.py` and `failure_quantum_benchmark.py` into `quantum-medical-research/endoscopy_guidance/`; run steps 2–6. Do not change any hyperparameters, thresholds, seeds, folds or feature sets, and do not edit the benchmark logic. If the CVC data is missing, stop and tell me where `dataset/dataset_b` should come from. After each step, verify the matching item in "Checks before anyone writes a number in the paper"; if a check fails, stop and report rather than fixing it yourself. At the end, show me the full `REPORT.md` from both runs.
