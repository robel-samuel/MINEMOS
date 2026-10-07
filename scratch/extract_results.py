import json

data = json.load(open('scratch/phase19_results.json'))

# ─── EXP 1 ───
print("=== EXP 1: Contradiction-Aware Utility (D vs D1) ===")
for cap in ['200','100']:
    for sys_id in ['D','D1']:
        m = data['exp1_contra_utility'][cap][sys_id]['metrics']
        p = data['exp1_contra_utility'][cap][sys_id]['phase19_diagnostics']
        d = data['exp1_contra_utility'][cap][sys_id]['diagnostics']
        print(f"  C={cap} {sys_id}: true_cons={m['true_consolidation_rate']}, "
              f"contra_ret={m['contradiction_retention']}, "
              f"false_merge={p['contradiction_false_merge_rate']}, "
              f"e2e={d['end_to_end_accuracy']}")

# ─── EXP 2 ───
print("\n=== EXP 2: Adaptive Beam Policies ===")
for pol in ['fixed_k3','tau_min_030','tau_min_040_std','tau_min_050','margin_005','margin_015']:
    m = data['exp2_adaptive_beam'][pol]['metrics']
    p = data['exp2_adaptive_beam'][pol]['phase19_diagnostics']
    d = data['exp2_adaptive_beam'][pol]['diagnostics']
    print(f"  {pol}: true_cons={m['true_consolidation_rate']}, "
          f"false_merge={p['contradiction_false_merge_rate']}, "
          f"e2e={d['end_to_end_accuracy']}, "
          f"avg_cands={p['avg_candidates_examined']:.2f}, "
          f"wall_s={m['wall_seconds']:.1f}")

# ─── EXP 3 ───
print("\n=== EXP 3: Safety Stress Test (D..D4) ===")
for sys_id in ['D','D1','D2','D3','D4']:
    m = data['exp3_safety_stress'][sys_id]['metrics']
    p = data['exp3_safety_stress'][sys_id]['phase19_diagnostics']
    d = data['exp3_safety_stress'][sys_id]['diagnostics']
    print(f"  {sys_id}: true_cons={m['true_consolidation_rate']}, "
          f"contra_ret={m['contradiction_retention']}, "
          f"false_merge={p['contradiction_false_merge_rate']}, "
          f"e2e={d['end_to_end_accuracy']}")

# ─── EXP 4 ───
print("\n=== EXP 4: 5k Stream (C=200, D..D4) ===")
for sys_id in ['D','D1','D2','D3','D4']:
    m = data['exp4_continual_5000'][sys_id]['metrics']
    p = data['exp4_continual_5000'][sys_id]['phase19_diagnostics']
    d = data['exp4_continual_5000'][sys_id]['diagnostics']
    print(f"  {sys_id}: true_cons={m['true_consolidation_rate']}, "
          f"contra_ret={m['contradiction_retention']}, "
          f"false_merge={p['contradiction_false_merge_rate']}, "
          f"e2e={d['end_to_end_accuracy']}, "
          f"nli_calls={m['nli_calls']}, "
          f"wall_s={m['wall_seconds']:.1f}")

# ─── EXP 5 ───
print("\n=== EXP 5: Capacity Sweep (D vs D3) ===")
for cap in ['1500','200','100','50']:
    for sys_id in ['D','D3']:
        m = data['exp5_capacity_sweep'][cap][sys_id]['metrics']
        p = data['exp5_capacity_sweep'][cap][sys_id]['phase19_diagnostics']
        d = data['exp5_capacity_sweep'][cap][sys_id]['diagnostics']
        print(f"  C={cap} {sys_id}: true_cons={m['true_consolidation_rate']}, "
              f"contra_ret={m['contradiction_retention']}, "
              f"false_merge={p['contradiction_false_merge_rate']}, "
              f"e2e={d['end_to_end_accuracy']}")

# ─── EXP 6 ───
print("\n=== EXP 6: Long-Gap Paraphrase Recall (D vs D3) ===")
for gap in ['10','50','100','500']:
    for sys_id in ['D','D3']:
        r = data['exp6_long_gap'][gap][sys_id]
        m = r['metrics']
        p = r['phase19_diagnostics']
        probe_recall = r.get('probe_recall_rate', 'N/A')
        d = r['diagnostics']
        print(f"  Gap={gap} {sys_id}: true_cons={m['true_consolidation_rate']}, "
              f"probe_recall={probe_recall}, "
              f"e2e={d['end_to_end_accuracy']}, "
              f"false_merge={p['contradiction_false_merge_rate']}, "
              f"wall_s={m['wall_seconds']:.1f}")
