import json

d = json.load(open('scratch/phase22_results.json'))

r = d['exp1_vocab_shift']['D']
print('metrics keys:', list(r['metrics'].keys()))
print('diagnostics keys:', list(r['phase22_diagnostics'].keys()))
print()

# Detailed diagnostics for exp1 and exp3
for exp_key in ['exp1_vocab_shift', 'exp3_adversarial_contradictions']:
    print('--- ' + exp_key + ' ---')
    for sys in ['D8', 'H1']:
        diag = d[exp_key][sys]['phase22_diagnostics']
        dense_rsc = diag.get('dense_rescue_count', 0)
        bm25_rsc = diag.get('bm25_rescue_count', 0)
        avg_beam = diag.get('avg_candidate_beam_size', 0)
        cat = diag.get('category_accuracy', {})
        print('  ' + sys + ': dense_rescue=' + str(dense_rsc) + ' bm25_rescue=' + str(bm25_rsc) + ' avg_beam=' + str(round(avg_beam, 2)))
        if cat:
            print('       categories: ' + str(cat))

# Exp 6 full metrics
print()
print('--- EXP 6 full metrics ---')
for sys in ['D', 'D4', 'D8', 'H1']:
    m = d['exp6_held_out'][sys]['metrics']
    print('  ' + sys + ': ' + str(m))

# NLI call comparison
print()
print('--- NLI call summary ---')
for exp_key in ['exp1_vocab_shift', 'exp2_dense_distractors', 'exp3_adversarial_contradictions']:
    print(exp_key + ':')
    for sys in ['D', 'D4', 'D8', 'H1']:
        nli = d[exp_key][sys]['nli_calls']
        wall = d[exp_key][sys]['wall_seconds']
        print('  ' + sys + ': nli=' + str(nli) + ' wall=' + str(round(wall, 1)) + 's')

# Exp5 ablation detail
print()
print('--- EXP 5 ablation full ---')
for variant in ['A_dense_only', 'B_bm25_only', 'C_dense_bm25_no_anchor', 'D_full_hybrid']:
    r = d['exp5_ablation'][variant]
    m = r['metrics']
    diag = r['phase22_diagnostics']
    print(variant + ':')
    print('  metrics: ' + str(m))
    print('  diagnostics: ' + str(diag))

# Phase21 baseline comparison (for reference)
print()
print('--- EXP 6 NLI and timing detail ---')
for sys in ['D', 'D4', 'D8', 'H1']:
    r = d['exp6_held_out'][sys]
    nli = r['nli_calls']
    wall = r['wall_seconds']
    t_dense = r.get('time_retrieval_dense', 0)
    t_bm25 = r.get('time_retrieval_bm25', 0)
    anc = r.get('anchor_count', 0)
    print('  ' + sys + ': nli=' + str(nli) + ' wall=' + str(round(wall, 1)) + 's t_dense=' + str(round(t_dense, 2)) + ' t_bm25=' + str(round(t_bm25, 2)) + ' anchors=' + str(anc))
