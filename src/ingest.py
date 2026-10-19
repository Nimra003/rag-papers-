import json
from pathlib import Path
from pypdf import PdfReader

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

PDF_DIR = Path("data/pdfs")
OUT_FILE = Path("data/chunks.json")
CHUNK_WORDS = 150
OVERLAP_WORDS = 30


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 30:  # skip tiny leftovers
            chunks.append(" ".join(piece))
    return chunks


def main():
    chunks = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        title = pdf_path.stem
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in chunk_words(text):
                chunks.append({
                    "id": len(chunks),
                    "paper": title,
                    "page": page_num,
                    "text": piece,
                })
        print(f"{title}: done")
    OUT_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()