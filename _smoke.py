import sys, json
sys.path.insert(0, "consensus_lab")
import app as a

with a.app.test_client() as c:
    print("=== ALL PAGES ===", flush=True)
    for p in ["/", "/datasets", "/results", "/analytics", "/test"]:
        r = c.get(p)
        print(p.ljust(15), r.status_code, flush=True)
    r = c.get("/generate")
    print("/generate".ljust(15), r.status_code, flush=True)

    print("\n=== PREVIEW ===", flush=True)
    for sc in ["gaussian","overlap","elongated","density","highdim","imbalance"]:
        r = c.post("/api/generate-preview",
            data=json.dumps({"scenario":sc,"n_samples":300,"n_clusters":3,"dim":2,"difficulty":"medium","seed":19}),
            content_type="application/json")
        d = json.loads(r.data)
        print("  " + sc.ljust(10), "OK" if "x" in d else "ERR", flush=True)

    print("\n=== EXPERIMENT/RUN ===", flush=True)
    for algo in ["hierarchical_baseline","hierarchical_weighted","sdgca","sdgca_modified"]:
        r = c.post("/api/experiment/run",
            data=json.dumps({"dataset":"analysis_densired_compact","algorithm":algo,"method":"average","seed":19,"m":5,"runs":1}),
            content_type="application/json")
        d = json.loads(r.data)
        msg = ("ERR:" + str(d["error"])[:60]) if d.get("error") else f"NMI={d['nmi_mean']:.3f} t={d['elapsed_sec']}s"
        print("  " + algo.ljust(25), msg, flush=True)

    print("\n=== AI ENDPOINTS ===", flush=True)
    r = c.get("/api/dataset-analysis/analysis_densired_compact")
    d = json.loads(r.data)
    print("  dataset-analysis :", r.status_code, "rec.m =", d.get("recommendations",{}).get("m"), flush=True)
    r = c.get("/api/ai-agent/dataset/analysis_densired_compact")
    d = json.loads(r.data)
    print("  ai-agent/dataset :", r.status_code, "summary OK:", bool(d.get("agent",{}).get("summary")), flush=True)
    r = c.post("/api/ai-agent/chat",
        data=json.dumps({"message":"Объясни NMI","dataset":"analysis_densired_compact","history":[]}),
        content_type="application/json")
    d = json.loads(r.data)
    print("  ai-agent/chat    :", r.status_code, "reply len:", len(d.get("response","")), flush=True)
