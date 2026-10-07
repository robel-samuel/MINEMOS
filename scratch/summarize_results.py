import json

with open('scratch/phase17_results.json') as f:
    res = json.load(f)

print('=== SCENARIO 1: RECALL VS GAP ===')
print(json.dumps(res['scenario_1']['recall_vs_gap'], indent=2))

print('\n=== SCENARIO 1: GAP METRICS SUMMARY ===')
for gap in ['10', '50', '100', '500']:
    print(f'Gap {gap}:')
    for s in ['A_nli', 'B_oracle', 'C_lexical']:
        d = res['scenario_1'][gap][s]
        m = d['metrics']
        diag = d['diagnostics']
        rec = d.get('probe_recall_rate')
        print(f"  {s}: recall={rec}, fcr={m['false_consolidation_rate']}, true_cons={m['true_consolidation_rate']}, diag_recall={diag.get('candidate_retrieval_recall')}, diag_nli_acc={diag.get('nli_accuracy_given_retrieval')}, end2end={diag.get('end_to_end_accuracy')}")

print('\n=== SCENARIO 2: REPEATED CONTRADICTIONS ===')
for s in ['A_nli', 'B_oracle', 'C_lexical']:
    d = res['scenario_2'][s]
    m = d['metrics']
    crt = d['contradiction_retention_over_time']
    diag = d['diagnostics']
    print(f"  {s}: contra_retention={m['contradiction_retention']}, fcr={m['false_consolidation_rate']}, crt={crt}, diag={diag}")

print('\n=== SCENARIO 3: CONTRADICTION -> RECOVERY ===')
for s in ['A_nli', 'B_oracle', 'C_lexical']:
    d = res['scenario_3'][s]
    m = d['metrics']
    bha = d['bounded_head_amplification']
    diag = d['diagnostics']
    print(f"  {s}: contra_retention={m['contradiction_retention']}, initial_merges={bha.get('initial_contradiction_merges')}, initial_err_rate={bha.get('initial_error_rate')}, step2_merges={bha.get('bounded_head_step2_merges')}, amp_ratio={bha.get('bounded_amplification_ratio')}, rec_key_sim={bha.get('mean_recovery_key_sim')}, rec_val_sim={bha.get('mean_recovery_val_sim')}")

print('\n=== SCENARIO 4: SEMANTIC DRIFT ===')
for s in ['A_nli', 'B_oracle', 'C_lexical']:
    d = res['scenario_4'][s]
    m = d['metrics']
    diag = d['diagnostics']
    print(f"  {s}: contra_retention={m['contradiction_retention']}, para_cons={m['paraphrase_consolidation']}, true_cons={m['true_consolidation_rate']}, fcr={m['false_consolidation_rate']}, diag={diag}")

print('\n=== SCENARIO 5: DISTRACTOR STRESS ===')
for n in ['50', '100', '200', '500']:
    print(f'Distractors {n}:')
    for s in ['A_nli', 'B_oracle', 'C_lexical']:
        d = res['scenario_5'][n][s]
        m = d['metrics']
        diag = d['diagnostics']
        print(f"  {s}: unrel_sep={m['unrelated_separation']}, slots={m['final_slots']}, fcr={m['false_consolidation_rate']}, diag={diag}")

print('\n=== SCENARIO 6: CAPACITY PRESSURE ===')
for cap in ['1500', '100', '50']:
    print(f'Capacity {cap}:')
    for s in ['A_nli', 'B_oracle', 'C_lexical']:
        d = res['scenario_6'][cap][s]
        m = d['metrics']
        diag = d['diagnostics']
        print(f"  {s}: slots={m['final_slots']}, updates={m['n_updates']}, inserts={m['n_inserts']}, evicts={m['n_evicts']}, true_cons={m['true_consolidation_rate']}, fcr={m['false_consolidation_rate']}, contra_ret={m['contradiction_retention']}, diag={diag}")

print('\n=== SCENARIO 7: VERY LONG CONTINUAL (5000 OBS) ===')
for s in ['A_nli', 'B_oracle', 'C_lexical']:
    d = res['scenario_7'][s]
    m = d['metrics']
    diag = d['diagnostics']
    print(f"  {s}: slots={m['final_slots']}, updates={m['n_updates']}, inserts={m['n_inserts']}, evicts={m['n_evicts']}, true_cons={m['true_consolidation_rate']}, fcr={m['false_consolidation_rate']}, contra_ret={m['contradiction_retention']}, para_cons={m['paraphrase_consolidation']}, diag={diag}, time={d['wall_seconds']}s")
