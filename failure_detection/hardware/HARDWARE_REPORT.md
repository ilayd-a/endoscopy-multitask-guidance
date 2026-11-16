# Hardware / noisy-simulation kernel check

Subset: 20 CVC frames (10 failures, 10 non-failures), alpha=0.1 (most-selected alpha for q_entangled in failure_detection\results_hm), 8 qubits, reps=2, linear entanglement, shots=1000, subset seed 0.

mean_diag should be 1 (k(x,x)=1); its drop measures noise directly. KTA = how much failure/non-failure structure the kernel carries (exact = reference).

| map       | source    |   mae_vs_exact |   max_err |   pearson_r |   mean_diag |   mean_offdiag |   kta |   n_entries |
|:----------|:----------|---------------:|----------:|------------:|------------:|---------------:|------:|------------:|
| entangled | exact     |          0.000 |     0.000 |       1.000 |       1.000 |          0.445 | 0.299 |         210 |
| entangled | noiseless |          0.010 |     0.042 |       0.999 |       1.000 |          0.445 | 0.300 |         210 |
| entangled | noisysim  |          0.239 |     0.539 |       0.996 |       0.512 |          0.232 | 0.291 |         210 |
| product   | exact     |          0.000 |     0.000 |       1.000 |       1.000 |          0.953 | 0.346 |         210 |
| product   | noiseless |          0.004 |     0.038 |       0.984 |       1.000 |          0.954 | 0.351 |         210 |
| product   | noisysim  |          0.303 |     0.344 |       0.817 |       0.680 |          0.652 | 0.245 |         210 |

**noisysim_meta**: backend=noisy simulator (fake_torino), shots=1000, pilot=False, physical_qubits=[0, 1, 2, 3, 4, 5, 6, 7], two_qubit_gates_entangled=56, depth_entangled=195, two_qubit_gates_product=0, depth_product=11

Interpretation: compare the entangled and product rows for the same source. The entangled circuit has many more two-qubit gates after compute-uncompute, so a larger error is expected and is a finding about the noise cost of entanglement, not a bug. This subset is too small for classification claims.