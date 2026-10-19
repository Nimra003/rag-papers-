import json
import re
from pathlib import Path


def norm(s):
    return re.sub(r"\s+", "", s.lower())


chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
by_paper = {}
for c in chunks:
    by_paper.setdefault(c["paper"], []).append(norm(c["text"]))

questions = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
ids = [q["id"] for q in questions]
if len(ids) != len(set(ids)):
    print("WARNING: duplicate ids")
for q in questions:
    if not q["answerable"]:
        continue
    texts = by_paper.get(q["paper"])
    if texts is None:
        print(f"#{q['id']}: unknown paper '{q['paper']}'")
        continue
    if not any(norm(k) in t for k in q["keywords"] for t in texts):
        print(f"#{q['id']}: no keyword found in {q['paper']}: {q['keywords']}")
print("Check finished")
