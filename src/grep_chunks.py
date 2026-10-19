import json
import re
import sys

term = re.sub(r"\s+", "", " ".join(sys.argv[1:]).lower())
chunks = json.load(open("data/chunks.json", encoding="utf-8"))
found = 0
for c in chunks:
    text = re.sub(r"\s+", "", c["text"].lower())
    if term in text:
        found += 1
        print(f"{c['paper']} p.{c['page']}")
print(f"{found} chunk(s) contain '{term}'")