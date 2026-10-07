import json

p19 = json.load(open('scratch/phase19_results.json'))

d4_5k = p19['exp4_continual_5000']['D4']
m = d4_5k['metrics']
d = d4_5k['diagnostics']
p = d4_5k['phase19_diagnostics']

print('--- Phase 19 D4 Recorded Checkpoint Verification ---')
print(f'True Consolidation Rate: {m["true_consolidation_rate"]}')
print(f'Contradiction Retention: {m["contradiction_retention"]}')
print(f'False Merge Rate:        {p["contradiction_false_merge_rate"]}')
print(f'End-to-End Accuracy:     {d["end_to_end_accuracy"]}')
print(f'NLI Calls:               {d4_5k["nli_calls"]}')
print(f'Wall Seconds:            {d4_5k["wall_seconds"]}')

d_5k = p19['exp4_continual_5000']['D']
md = d_5k['metrics']
dd = d_5k['diagnostics']
pd = d_5k['phase19_diagnostics']
print('--- Phase 19 D Recorded Checkpoint Verification ---')
print(f'True Consolidation Rate: {md["true_consolidation_rate"]}')
print(f'Contradiction Retention: {md["contradiction_retention"]}')
print(f'False Merge Rate:        {pd["contradiction_false_merge_rate"]}')
print(f'End-to-End Accuracy:     {dd["end_to_end_accuracy"]}')
print(f'NLI Calls:               {d_5k["nli_calls"]}')
