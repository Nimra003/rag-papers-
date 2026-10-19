import json
from pathlib import Path

GRADES = {1: "correct", 2: "correct", 3: "correct", 4: "correct",
          5: "abstained", 6: "abstained", 7: "correct", 8: "correct",
          9: "partial", 10: "correct", 11: "correct"}

path = Path("eval/results.json")
results = json.loads(path.read_text(encoding="utf-8"))
for r in results:
    if r["id"] in GRADES:
        r["grade"] = GRADES[r["id"]]
path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved grades")