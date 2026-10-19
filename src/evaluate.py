import json
import re
import sys
import time
from pathlib import Path

from retrieve import retrieve, retrieve_hybrid

QUESTIONS = Path("eval/questions.json")
RESULTS = Path("eval/results.json")
KS = (1, 3, 5)


def norm(s):
    return re.sub(r"\s+", "", s.lower())


def hit(retrieved, paper, keywords):
    keys = [norm(k) for k in keywords]
    return any(r["paper"] == paper and any(k in norm(r["text"]) for k in keys)
               for r in retrieved)


def run_retrieval(retriever=retrieve):
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    answerable = [q for q in items if q["answerable"]]
    totals = {k: 0 for k in KS}
    for q in items:
        retrieved = retriever(q["question"], max(KS))
        top = retrieved[0]["score"]
        if q["answerable"]:
            flags = {k: hit(retrieved[:k], q["paper"], q["keywords"]) for k in KS}
            for k in KS:
                totals[k] += flags[k]
            print(f"#{q['id']} answerable  top={top:.3f}  hit@5={flags[5]}")
            if not flags[5]:
                print("   MISS:", q["question"])
                for r in retrieved:
                    print(f"     {r['paper']} p.{r['page']} [{r['score']:.3f}]")
        else:
            print(f"#{q['id']} unanswerable top={top:.3f}")
    print()
    for k in KS:
        print(f"recall@{k}: {totals[k]}/{len(answerable)} = {totals[k]/len(answerable):.2f}")


def run_generation():
    from generate import ask  # needs your API key
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = []
    for q in items:
        out = ask(q["question"], k=4)
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "reference": q.get("reference", ""),
            "found": out["found"],
            "answer": out["answer"],
            "cited": [f"{c['paper']} p.{c['page']}" for c in out["citations"]],
            "grade": "",  # fill in by hand: correct / partial / unsupported / wrong
        })
        print(f"done #{q['id']}")
        time.sleep(6)  # stay under free-tier rate limits
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run_summary():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    ans = [r for r in res if r["answerable"]]
    unans = [r for r in res if not r["answerable"]]
    counts = {}
    for r in ans:
        if r["grade"]:
            counts[r["grade"]] = counts.get(r["grade"], 0) + 1
    print("Graded answerable:", counts, f"(of {len(ans)})")
    print(f"Correct abstentions: {sum(not r['found'] for r in unans)}/{len(unans)}")
    print(f"Wrongly abstained: {sum(not r['found'] for r in ans)}/{len(ans)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    MODES = {
        "retrieval": lambda: run_retrieval(retrieve),
        "hybrid": lambda: run_retrieval(retrieve_hybrid),
        "generate": run_generation,
        "summary": run_summary,
    }
    MODES[mode]()