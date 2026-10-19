# -*- coding: utf-8 -*-
"""P0-1 analysis v2: exact WHT spectra with CORRECT additive basis (popcount<=1),
order-wise energy decomposition, typed signature energies, additive-null MC,
per-unit citation-vector spectra, reference-answer robustness, gate verdict.

Key v2 fixes: additive projection uses subset indices {0} u {1,2,4,8,16,32};
E_ge2 sums popcount>=2; variance shares exclude the constant term.
Gate logic (deterministic T=0 protocol, rerun flips=0 => measurement noise ~0):
  C1 (materiality): nonadditive variance share E_ge2/E_ge1 >= 0.05
  C1 (statistical, formal): E_ge2 > Q95 of null(sigma=s_rerun)  [formality when sigma~0]
  Robustness tier: E_ge2 vs Q95(null, sigma=s_ref) — "beyond reference-choice ambiguity?"
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from p0_data import TASKS, states  # noqa: E402

ENGINES = ["qwen3b", "internvl3-2b"]
B_NULL = 20000
MATERIALITY = 0.05
RNG = np.random.default_rng(42)

def load_records(engine):
    path = os.path.join(ROOT, "results", "p0_1", f"{engine}.jsonl")
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                recs[(r["task"], tuple(r["state"]), r.get("rerun", 0))] = r
    return recs

def design(n=6):
    X = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
    M = np.ones((2 ** n, 2 ** n))
    for s in range(1, 2 ** n):
        bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
        col = np.ones(2 ** n)
        for i in range(n):
            if bits[i]:
                col = col * X[:, i]
        M[:, s] = col
    return X, M

X, M = design()
N6 = 64
ADD_IDX = [0] + [2 ** i for i in range(6)]            # const + 6 first-order
GE2_IDX = [s for s in range(64) if s != 0 and s not in ADD_IDX]
GE1_IDX = [s for s in range(64) if s != 0]
POP = [bin(s).count("1") for s in range(64)]

def wht(F):
    return M.T @ F / N6

def sig_of(task_key, s):
    types = [u["type"] if not u.get("caption") else "T" for u in TASKS[task_key]["units"]]
    sig = sorted(types[i] for i in range(6) if (s >> (5 - i)) & 1)
    return ",".join(sig) if sig else "const"

def spectrum_report(F, task_key, sigmas):
    """sigmas: dict with measurement (rerun) and estimand (ref) noise scales."""
    F = np.asarray(F, dtype=float)
    fh = wht(F)
    E = fh ** 2
    var_total = float(np.var(F))          # = E_ge1 (orthogonality)
    E_ge1 = float(E[GE1_IDX].sum())
    E_ge2 = float(E[GE2_IDX].sum())
    E_1 = float(E[[2 ** i for i in range(6)]].sum())
    nonadd_share = E_ge2 / E_ge1 if E_ge1 > 0 else 0.0
    add_r2 = 1.0 - nonadd_share
    # signature energies (non-const)
    sig_energy, sig_ge2 = {}, {}
    for s in range(1, 64):
        k = sig_of(task_key, s)
        sig_energy[k] = sig_energy.get(k, 0.0) + float(E[s])
        if POP[s] >= 2:
            sig_ge2[k] = sig_ge2.get(k, 0.0) + float(E[s])
    order_energy = {}
    for s in range(1, 64):
        order_energy[POP[s]] = order_energy.get(POP[s], 0.0) + float(E[s])
    # top coefficients (excluding const)
    order_idx = sorted(GE1_IDX, key=lambda s: -abs(fh[s]))
    top = [{"S": [i + 1 for i in range(6) if (s >> (5 - i)) & 1],
            "sig": sig_of(task_key, s), "val": float(fh[s])} for s in order_idx[:8]]
    # null MC for E_ge2 under additive + noise
    F_add = M[:, ADD_IDX] @ fh[ADD_IDX]
    nulls = {}
    for tag, sg in sigmas.items():
        if sg <= 0:
            nulls[tag] = {"q95": 0.0, "p": (0.0 if E_ge2 > 0 else 1.0)}
            continue
        draws = np.zeros(B_NULL)
        base = fh[ADD_IDX]
        for b in range(B_NULL):
            Fn = F_add + RNG.normal(0, sg, N6)
            Eh = (M.T @ Fn / N6) ** 2
            draws[b] = Eh[GE2_IDX].sum()
        nulls[tag] = {"q95": float(np.quantile(draws, 0.95)),
                      "p": float((draws >= E_ge2).mean())}
    # top-m variance explained (non-additive residual): generic vs signature-structured
    def var_resid(idx):
        Fr = M[:, idx] @ fh[idx]
        return float(1 - np.var(F - Fr) / var_total) if var_total > 0 else 1.0
    generic_curve = []
    for m in (3, 5, 8, 12, 20):
        generic_curve.append({"m": m, "var_explained": var_resid(order_idx[:m])})
    top_sigs = [k for k, v in sorted(sig_ge2.items(), key=lambda kv: -kv[1])[:4]]
    struct_idx = [s for s in GE2_IDX if sig_of(task_key, s) in top_sigs]
    struct = {"top_signatures_ge2": top_sigs, "n_coefs": len(struct_idx),
              "var_explained": var_resid(struct_idx)}
    return {"E_ge1": E_ge1, "E_ge2": E_ge2, "E_1": E_1, "var_total": var_total,
            "nonadd_share_of_var": nonadd_share, "additive_R2": add_r2,
            "share_ge2_incl_const": (E_ge2 / float(E.sum())) if float(E.sum()) > 0 else 0.0,
            "order_energy": {str(k): round(v, 6) for k, v in sorted(order_energy.items())},
            "sig_energy_ge1": {k: round(v, 6) for k, v in sorted(sig_energy.items(), key=lambda kv: -kv[1])[:8]},
            "sig_energy_ge2": {k: round(v, 6) for k, v in sorted(sig_ge2.items(), key=lambda kv: -kv[1])[:8]},
            "top_coefs": top, "null_tests": nulls,
            "generic_curve": generic_curve, "struct_curve": struct}

def main():
    out = {"per_task": {}, "citation_units": {}, "ref_robustness": {}, "cross_engine": {}, "verdict": {}}
    spectra_store = {}
    for eng in ENGINES:
        recs = load_records(eng)
        if not recs:
            continue
        out["per_task"][eng] = {}
        out["citation_units"][eng] = {}
        for tk in TASKS:
            sl = [tuple(s) for s in states()]
            if not all((tk, st, 0) in recs for st in sl):
                print(f"[{eng}] {tk}: incomplete, skipped")
                continue
            # assemble utilities
            U = {k: np.zeros(64) for k in ("U_ans_mean", "U_ans_ref1", "U_ans_ref2", "U_R", "U_P")}
            C = np.zeros((64, 6))
            refs, insuff = [], 0
            for i, st in enumerate(sl):
                r = recs[(tk, st, 0)]
                U["U_ans_ref1"][i], U["U_ans_ref2"][i] = r["U_ans"]
                U["U_ans_mean"][i] = float(np.mean(r["U_ans"]))
                U["U_R"][i], U["U_P"][i] = r["U_R"], r["U_P"]
                C[i] = r["c"]
                refs.append(r["gen"])
                insuff += int(r["insufficient"])
            # noise scales
            d_ref = np.abs(U["U_ans_ref1"] - U["U_ans_ref2"]) / np.sqrt(2)
            s_ref = float(np.std(d_ref, ddof=1))
            rr = [k for k in recs if k[0] == tk and k[2] > 0]
            diffs, flips, pairs = [], 0, 0
            seen = {}
            for (t2, st, k) in rr:
                seen.setdefault(st, {})[k] = recs[(t2, st, k)]
            for st, runs in seen.items():
                runs[0] = recs.get((tk, st, 0))
                vals = [runs[k] for k in sorted(runs) if runs.get(k)]
                for a, b in itertools.combinations(vals, 2):
                    pairs += 1
                    diffs.append(abs(np.mean(a["U_ans"]) - np.mean(b["U_ans"])) / np.sqrt(2))
                    if a["tags"] != b["tags"]:
                        flips += 1
            s_rerun = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            sigmas = {"measurement": max(s_rerun, 1e-4), "estimand": s_ref}
            out["per_task"][eng][tk] = spectrum_report(U["U_ans_mean"], tk, sigmas)
            out["per_task"][eng][tk]["noise"] = {"s_ref": s_ref, "s_rerun": s_rerun,
                                                 "rerun_pairs": pairs, "citation_flips": flips,
                                                 "insufficient_rate": insuff / 64}
            # per-unit citation spectra (binary, deterministic at T=0)
            out["citation_units"][eng][tk] = {}
            for j in range(6):
                rep = spectrum_report(C[:, j], tk, {"measurement": 0.0})
                out["citation_units"][eng][tk][f"c{j+1}"] = {
                    "cite_rate": float(C[:, j].mean()),
                    "nonadd_share": rep["nonadd_share_of_var"],
                    "E_ge2": rep["E_ge2"],
                    "top_ge2_sigs": rep["sig_energy_ge2"]}
            fh = wht(U["U_ans_mean"])
            spectra_store[(eng, tk)] = fh
            # ref robustness: spectral cosine between ref1/ref2 versions (non-const)
            f1, f2 = wht(U["U_ans_ref1"]), wht(U["U_ans_ref2"])
            cos = float(np.dot(f1[GE1_IDX], f2[GE1_IDX]) /
                        (np.linalg.norm(f1[GE1_IDX]) * np.linalg.norm(f2[GE1_IDX]) + 1e-12))
            out["ref_robustness"][f"{eng}|{tk}"] = {
                "cosine_nonconst_ref1_ref2": cos,
                "nonadd_share_ref1": spectrum_report(U["U_ans_ref1"], tk, sigmas)["nonadd_share_of_var"],
                "nonadd_share_ref2": spectrum_report(U["U_ans_ref2"], tk, sigmas)["nonadd_share_of_var"]}
            r = out["per_task"][eng][tk]
            print(f"[{eng}] {tk}: addR2={r['additive_R2']:.3f} nonadd={r['nonadd_share_of_var']:.3f} "
                  f"E_ge2={r['E_ge2']:.4f} | meas_q95={r['null_tests']['measurement']['q95']:.4f} "
                  f"estimand_q95={r['null_tests']['estimand']['q95']:.4f} | s_ref={s_ref:.3f} s_rerun={s_rerun:.4f} "
                  f"flips={flips}/{pairs} insuff={insuff}/64")
    # cross-engine (structure only: non-const coefficients)
    ce = {}
    for tk in TASKS:
        pair = {}
        if all((e, tk) in spectra_store for e in ENGINES):
            fa, fb = spectra_store[("qwen3b", tk)], spectra_store[("internvl3-2b", tk)]
            fan, fbn = fa[GE1_IDX], fb[GE1_IDX]
            pair["cosine_nonconst"] = float(np.dot(fan, fbn) / (np.linalg.norm(fan) * np.linalg.norm(fbn) + 1e-12))
            ra = np.argsort(np.argsort(-np.abs(fan))); rb = np.argsort(np.argsort(-np.abs(fbn)))
            pair["spearman_abs_nonconst"] = float(np.corrcoef(ra, rb)[0, 1])
            ta = set(np.argsort(-np.abs(fan))[:10]); tb = set(np.argsort(-np.abs(fbn))[:10])
            pair["jaccard_top10_nonconst"] = len(ta & tb) / len(ta | tb)
        ce[tk] = pair
    out["cross_engine"] = ce
    # verdict
    verdict = {}
    for eng, pt in out["per_task"].items():
        c1_mat = [tk for tk, r in pt.items() if r["nonadd_share_of_var"] >= MATERIALITY]
        c1_stat = [tk for tk, r in pt.items() if r["null_tests"]["measurement"]["p"] < 0.05]
        c1_robust = [tk for tk, r in pt.items() if r["null_tests"]["estimand"]["p"] < 0.05]
        c2_sigs = {}
        for tk, r in pt.items():
            for k in r["struct_curve"]["top_signatures_ge2"]:
                c2_sigs.setdefault(k, []).append(tk)
        verdict[eng] = {
            "additive_R2_by_task": {tk: round(r["additive_R2"], 3) for tk, r in pt.items()},
            "C1_materiality": {"pass": len(c1_mat) >= 1, "tasks": c1_mat},
            "C1_beyond_estimand_noise": {"tasks": c1_robust},
            "C2_top_ge2_signatures_by_task": c2_sigs}
    out["verdict"] = verdict
    with open(os.path.join(ROOT, "results", "p0_1", "analysis_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("\n== VERDICT ==")
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str))
    print("cross-engine:", json.dumps(ce, indent=1))
    print("written: results/p0_1/analysis_v2.json")

if __name__ == "__main__":
    main()
