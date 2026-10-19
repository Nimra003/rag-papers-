import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from retrieve import retrieve_hybrid as retrieve

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=os.environ["LLM_API_KEY"])

CACHE_FILE = Path("data/llm_cache.json")
_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


class Answer(BaseModel):
    found: bool = Field(description="True only if the passages contain relevant information")
    answer: str
    used_passages: list[int] = Field(description="Numbers of the passages actually used")


def build_prompt(question, passages):
    blocks = [f"[{i}] ({p['paper']}, p.{p['page']})\n{p['text']}"
              for i, p in enumerate(passages, start=1)]
    return (
        "Answer the question using ONLY the numbered passages below.\n"
        "If the passages contain relevant information, answer with what they "
        "support, and say briefly what is missing if the answer is incomplete.\n"
        "Set found to false only if the passages contain nothing relevant, and "
        "then set answer to 'Not found in the documents.'\n"
        "Write the answer in your own words in at most 3 sentences.\n"
        "List the numbers of the passages you actually used in used_passages.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {question}"
    )


def call_llm(prompt, retries=3):
    key = hashlib.sha256((MODEL + prompt).encode("utf-8")).hexdigest()
    if key in _cache:  # same model + same prompt: no request needed
        return Answer.model_validate(_cache[key])
    last_error = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=Answer,
                ),
            )
            parsed = Answer.model_validate_json(resp.text)
            _cache[key] = parsed.model_dump()
            CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
            return parsed
        except ValueError as e:  # malformed JSON: retry right away
            last_error = e
        except Exception as e:  # API or rate-limit errors
            last_error = e
            msg = str(e)
            if "NOT_FOUND" in msg or "PERMISSION_DENIED" in msg or "PerDay" in msg:
                break  # retrying won't help: bad model/key or daily quota used up
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {last_error}")


def ask(question, k=4):
    passages = retrieve(question, k)
    parsed = call_llm(build_prompt(question, passages))
    valid = [n for n in parsed.used_passages if 1 <= n <= len(passages)]
    return {
        "question": question,
        "found": parsed.found,
        "answer": parsed.answer,
        "citations": [passages[n - 1] for n in valid],
        "retrieved": passages,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which features predict heart disease?"
    result = ask(q)
    print("\nANSWER:", result["answer"])
    print("Found:", result["found"])
    for c in result["citations"]:
        print(f"  - {c['paper']} p.{c['page']} (score {c['score']:.3f})")
    if not result["found"]:
        print("\nRetrieved (for debugging):")
        for r in result["retrieved"]:
            print(f"  [{r['score']:.3f}] {r['paper']} p.{r['page']}: {r['text'][:150]}...")