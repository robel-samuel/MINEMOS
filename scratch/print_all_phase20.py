import json

data = json.load(open('scratch/phase20_results.json'))

print("==================================================")
print("PHASE 20 EXPERIMENT SUMMARY")
print("==================================================")

# 1. Exp 1
print("\n--- EXP 1: ANCHOR WEIGHT SWEEP ---")
for w, r in data['exp1_weight_sweep'].items():
    m = r['metrics']
    d = r['diagnostics']
    p = r['phase20_diagnostics']
    print(f"w={w:4s} | TrueCons={m['true_consolidation_rate']:.4f} | ContraRet={m['contradiction_retention']:.4f} | FMR={p['contradiction_false_merge_rate']:.4f} | E2E={d['end_to_end_accuracy']:.4f} | AvgPen={p['avg_applied_penalty']:.4f}")

# 2. Exp 2
print("\n--- EXP 2: BOUNDED ANCHOR CAPACITY SWEEP ---")
for k, r in data['exp2_capacity_sweep'].items():
    m = r['metrics']
    d = r['diagnostics']
    p = r['phase20_diagnostics']
    print(f"k={k:9s} | TrueCons={m['true_consolidation_rate']:.4f} | ContraRet={m['contradiction_retention']:.4f} | FMR={p['contradiction_false_merge_rate']:.4f} | E2E={d['end_to_end_accuracy']:.4f} | AnchorsRetained={p['total_anchors_retained']}")

# 3. Exp 3
print("\n--- EXP 3: TEMPORAL DECAY SWEEP ---")
for lmb, r in data['exp3_decay_sweep'].items():
    m = r['metrics']
    d = r['diagnostics']
    p = r['phase20_diagnostics']
    print(f"lambda={lmb:7s} | TrueCons={m['true_consolidation_rate']:.4f} | ContraRet={m['contradiction_retention']:.4f} | FMR={p['contradiction_false_merge_rate']:.4f} | E2E={d['end_to_end_accuracy']:.4f} | AvgAge={p['avg_anchor_age']:.2f}")

# 4. Exp 4
print("\n--- EXP 4: 5,000-OBSERVATION CONTINUAL STREAM ---")
for sys_id, r in data['exp4_continual_5000'].items():
    m = r['metrics']
    d = r['diagnostics']
    p = r['phase20_diagnostics']
    print(f"System {sys_id:2s} | TrueCons={m['true_consolidation_rate']:.4f} | ContraRet={m['contradiction_retention']:.4f} | FMR={p['contradiction_false_merge_rate']:.4f} | E2E={d['end_to_end_accuracy']:.4f} | Top1={d['top1_retrieval_recall']:.4f} | TopK={d['topk_retrieval_recall']:.4f} | CondNLI={d['conditional_nli_accuracy']:.4f} | NLI_calls={r['nli_calls']} | Wall_s={r['wall_seconds']:.1f}")

# 5. Exp 5
print("\n--- EXP 5: LONG-GAP PARAPHRASE RECALL ---")
for gap in ['10', '50', '100', '500']:
    print(f"Gap {gap}:")
    for sys_id, r in data['exp5_long_gap'][gap].items():
        m = r['metrics']
        d = r['diagnostics']
        pr = r.get('probe_recall_rate', 'N/A')
        print(f"  System {sys_id:2s} | TrueCons={m['true_consolidation_rate']:.4f} | ProbeRecall={pr} | E2E={d['end_to_end_accuracy']:.4f} | Wall_s={r['wall_seconds']:.1f}")

# 6. Exp 6
print("\n--- EXP 6: CONTROLLED SEMANTIC DRIFT (SCENARIO 7) ---")
for sys_id, r in data['exp6_semantic_drift'].items():
    m = r['metrics']
    d = r['diagnostics']
    p = r['phase20_diagnostics']
    print(f"System {sys_id:2s} | TrueCons={m['true_consolidation_rate']:.4f} | ContraRet={m['contradiction_retention']:.4f} | FMR={p['contradiction_false_merge_rate']:.4f} | E2E={d['end_to_end_accuracy']:.4f} | TempUpdateAccept={p['temporal_update_acceptance_rate']:.4f} | NLI_calls={r['nli_calls']}")
