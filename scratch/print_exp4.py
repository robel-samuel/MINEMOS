import json
data = json.load(open('scratch/phase20_results.json'))

print('=== EXP 4: 5,000-OBSERVATION CONTINUAL STREAM (C=200) ===')
for sys_id, res in data['exp4_continual_5000'].items():
    m = res['metrics']
    d = res['diagnostics']
    p = res['phase20_diagnostics']
    print(f"System {sys_id:2s}: TrueCons={m['true_consolidation_rate']:.4f}  "
          f"ContraRet={m['contradiction_retention']:.4f}  "
          f"FMR={p['contradiction_false_merge_rate']:.4f}  "
          f"E2E={d['end_to_end_accuracy']:.4f}  "
          f"Top1={d['top1_retrieval_recall']:.4f}  "
          f"TopK={d['topk_retrieval_recall']:.4f}  "
          f"NLI_acc={d['conditional_nli_accuracy']:.4f}  "
          f"NLI_calls={res['nli_calls']}  "
          f"Wall_s={res['wall_seconds']:.1f}")
