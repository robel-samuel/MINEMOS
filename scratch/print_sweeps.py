import json
data = json.load(open('scratch/phase20_results.json'))

print('=== EXP 1: WEIGHT SWEEP ===')
for w, res in data['exp1_weight_sweep'].items():
    m = res['metrics']
    d = res['diagnostics']
    p = res['phase20_diagnostics']
    print(f'  w={w:4s}: TrueCons={m["true_consolidation_rate"]:.4f}  ContraRet={m["contradiction_retention"]:.4f}  FMR={p["contradiction_false_merge_rate"]:.4f}  E2E={d["end_to_end_accuracy"]:.4f}  Pen={p["avg_applied_penalty"]:.4f}')

print('\n=== EXP 2: CAPACITY SWEEP ===')
for k, res in data['exp2_capacity_sweep'].items():
    m = res['metrics']
    d = res['diagnostics']
    p = res['phase20_diagnostics']
    print(f'  k={k:9s}: TrueCons={m["true_consolidation_rate"]:.4f}  ContraRet={m["contradiction_retention"]:.4f}  FMR={p["contradiction_false_merge_rate"]:.4f}  E2E={d["end_to_end_accuracy"]:.4f}  Anchors={p["total_anchors_retained"]}')

print('\n=== EXP 3: DECAY SWEEP ===')
for lmb, res in data['exp3_decay_sweep'].items():
    m = res['metrics']
    d = res['diagnostics']
    p = res['phase20_diagnostics']
    print(f'  lambda={lmb:7s}: TrueCons={m["true_consolidation_rate"]:.4f}  ContraRet={m["contradiction_retention"]:.4f}  FMR={p["contradiction_false_merge_rate"]:.4f}  E2E={d["end_to_end_accuracy"]:.4f}  Age={p["avg_anchor_age"]:.2f}')
