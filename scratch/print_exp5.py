import json
data = json.load(open('scratch/phase20_results.json'))

print('=== EXP 5: LONG GAP PARAPHRASE RECALL ===')
for gap, syss in data['exp5_long_gap'].items():
    print(f'Gap {gap}:')
    for sys_id, res in syss.items():
        m = res['metrics']
        d = res['diagnostics']
        pr = res.get('probe_recall_rate', 'N/A')
        print(f"  System {sys_id:2s}: TrueCons={m['true_consolidation_rate']:.4f}  "
              f"ProbeRecall={pr}  "
              f"E2E={d['end_to_end_accuracy']:.4f}  "
              f"Wall_s={res['wall_seconds']:.1f}")
