import json

d = json.load(open("scratch/phase23_results.json"))
print("Top Keys:", list(d.keys()))

print("\n=== EXP 1: THRESHOLD SWEEP ===")
for th in sorted(d["exp1_threshold_sweep"].keys()):
    for s in ["D8", "H1"]:
        r = d["exp1_threshold_sweep"][th][s]
        e2e = r.get("e2e_accuracy")
        fmr = r.get("false_merge_rate")
        para = r.get("paraphrase_accuracy")
        contra = r.get("contradiction_accuracy")
        dist = r.get("failure_category_distribution")
        nli = r.get("nli_calls")
        wall = r.get("wall_seconds")
        print(f"th={th} {s}: e2e={e2e} fmr={fmr} para={para} contra={contra} nli={nli} wall={wall}s dist={dist}")

print("\n=== EXP 2: ORACLE RETRIEVAL ===")
for k, r in d["exp2_oracle_retrieval"].items():
    e2e = r.get("e2e_accuracy")
    para = r.get("paraphrase_accuracy")
    ret_rec = r.get("retrieval_recall")
    nli_acc = r.get("nli_acceptance_rate")
    dist = r.get("failure_category_distribution")
    print(f"{k}: e2e={e2e} para={para} ret_rec={ret_rec} nli_acc={nli_acc} dist={dist}")

print("\n=== EXP 3: RETENTION ORACLE ===")
for k, r in d["exp3_retention_oracle"].items():
    e2e = r.get("e2e_accuracy")
    surv = r.get("slot_survival_rate")
    evicts = r.get("eviction_count")
    dist = r.get("failure_category_distribution")
    print(f"{k}: e2e={e2e} surv={surv} evicts={evicts} dist={dist}")

print("\n=== EXP 4: MATRIX 2x2 ===")
for k, r in d["exp4_matrix"].items():
    e2e = r.get("e2e_accuracy")
    surv = r.get("slot_survival_rate")
    ret_rec = r.get("retrieval_recall")
    dist = r.get("failure_category_distribution")
    print(f"{k}: e2e={e2e} surv={surv} ret_rec={ret_rec} dist={dist}")

print("\n=== EXP 5: EMBEDDING DRIFT ===")
for s, r in d["exp5_embedding_drift"].items():
    m_drift = r.get("mean_embedding_drift")
    min_drift = r.get("min_embedding_drift")
    e2e = r.get("e2e_accuracy")
    dist = r.get("failure_category_distribution")
    print(f"{s}: mean_drift={m_drift} min_drift={min_drift} e2e={e2e} dist={dist}")
    drift_details = r.get("drift_details", {})
    for tid, tinfo in list(drift_details.items())[:3]:
        hist = [h["cosine_to_initial"] for h in tinfo["history"]]
        print(f"   {tid}: updates={tinfo['update_count']} cosine_traj={hist}")

print("\n=== EXP 6: TRACES ===")
for s, r in d["exp6_traces"].items():
    print(f"{s}: e2e={r.get('e2e_accuracy')} dist={r.get('failure_category_distribution')}")
    for t in r.get("case_traces", []):
        print(f"   target={t['target_id']} probe={t['probe_id']} cat={t['category']} fail={t['failure_category']} correct={t['final_correct']} nli_score={t['nli_score']}")

print("\n=== EXP 7: HELD-OUT (seed=2323) ===")
for s, r in d["exp7_held_out"].items():
    e2e = r.get("e2e_accuracy")
    fmr = r.get("false_merge_rate")
    para = r.get("paraphrase_accuracy")
    contra = r.get("contradiction_accuracy")
    ret_rec = r.get("retrieval_recall")
    surv = r.get("slot_survival_rate")
    dist = r.get("failure_category_distribution")
    nli = r.get("nli_calls")
    wall = r.get("wall_seconds")
    print(f"{s}: e2e={e2e} fmr={fmr} para={para} contra={contra} ret_rec={ret_rec} surv={surv} nli={nli} wall={wall}s dist={dist}")
