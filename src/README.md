# rag-papers: Question answering over research papers, with citations and abstention

A small retrieval-augmented generation (RAG) system that answers questions from a collection of heart disease prediction papers. It cites the passages it used, returns schema-validated JSON, and says "Not found in the documents" when the retrieved text does not contain the answer. The repo also contains a hand-graded evaluation set and an honest account of where the system fails.

> **Not medical advice.** Answers come only from the loaded papers and may be wrong or incomplete.

## What it does

- Reads PDFs, splits them into overlapping chunks and keeps the paper name and page number for every chunk.
- Retrieves with **hybrid search**: sentence embeddings (FAISS) merged with BM25 keyword search using reciprocal rank fusion.
- Sends the top passages to an LLM (Gemini) and asks for a JSON answer with the numbers of the passages it used. The output is validated with Pydantic and retried on malformed JSON.
- Abstains when the passages contain nothing relevant.
- Provides a Streamlit app that shows the answer, the cited passages and the retrieval similarity scores.

## How it works

```
PDFs -> pypdf -> chunks (paper, page, text)
chunks -> all-MiniLM-L6-v2 embeddings -> FAISS index
question -> FAISS top-20 + BM25 top-20 -> rank fusion -> top-k passages
top-k passages + question -> Gemini -> JSON {found, answer, used_passages}
JSON -> Pydantic validation (retry on failure) -> answer + citations
```

| Component | Choice |
|---|---|
| PDF text | `pypdf` |
| Chunking | about 150 words with 30 words of overlap, split per page so citations have accurate page numbers |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` |
| Vector index | FAISS (inner product on normalized vectors) |
| Keyword search | `rank-bm25` |
| Fusion | Reciprocal rank fusion (k = 60, candidate pool of 20 per method) |
| LLM | Gemini via `google-genai`, temperature 0, structured JSON output |
| UI | Streamlit |

## Corpus

Eleven PDFs: ten published papers on heart disease prediction and one project report written by the author's team. The PDFs are not included in this repo because of their different licenses.

| File name | Source |
|---|---|
| `algorithms` | Bhatt et al., "Effective Heart Disease Prediction Using Machine Learning Techniques", Algorithms 2023, doi:10.3390/a16020088 |
| `biomedical` | Kehkashan et al., "Sustainable and interpretable heart disease prediction: a clinical decision support approach for biomedical healthcare applications", Scientific Reports 2026, doi:10.1038/s41598-026-37840-0 |
| `Comprehensive-survey` | Mondal, Maity and Nag, "A Comprehensive Survey of Heart Disease Prediction Approaches", Archives of Computational Methods in Engineering 2026, doi:10.1007/s11831-025-10438-x |
| `HDP using ML` | Shah, Patel and Bharti, "Heart Disease Prediction using Machine Learning Techniques", SN Computer Science 2020, doi:10.1007/s42979-020-00365-y |
| `HDPS` | Chen et al., "HDPS: Heart Disease Prediction System", Computing in Cardiology 2011 |
| `heart_report` | Bibi, Hashmi and Yasir, project report on multi-site heart disease prediction (the author's team) |
| `Hybrid_Machine_Learning_Techniques` | Mohan, Thirumalai and Srivastava, "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques", IEEE Access 2019, doi:10.1109/ACCESS.2019.2923707 |
| `Intelligent_heart_disease_prediction` | Palaniappan and Awang, "Intelligent Heart Disease Prediction System Using Data Mining Techniques", IEEE 2008 |
| `ML` | Jindal et al., "Heart disease prediction using machine learning algorithms", IOP Conf. Ser.: Mater. Sci. Eng. 1022, 2021, doi:10.1088/1757-899X/1022/1/012072 |
| `Performance_Evaluation` | Hasan et al., "Performance Evaluation of Hybrid Machine Learning Models for Heart Disease Prediction in U.S. Clinical Decision Support Systems", ICSADL 2026, doi:10.1109/ICSADL67539.2026.11452047 |
| `prediction using AI` | Hasan et al., "Heart Disease Prediction Using Artificial Intelligence Algorithms: A Comparative Study", Vascular and Endovascular Review 9(1), 2026 |

## Evaluation

The evaluation set (`eval/questions.json`) has **48 questions**: 41 answerable and 7 unanswerable. Eleven answerable questions come from the team report and 30 come from the other ten papers. The unanswerable questions are a mix of off-topic and near-miss questions (for example, fine-tuning a model that no paper uses). Answerable questions have a reference answer and keywords that must appear in a retrieved chunk from the right paper. The questions were drafted with AI assistance and checked by hand against the source PDFs.

### Retrieval

Recall@k counts a question as a hit if one of the top k retrieved chunks comes from the right paper and contains an answer keyword.

| Method | Questions | recall@1 | recall@3 | recall@5 |
|---|---|---|---|---|
| Embeddings only | 11 (version 1) | 0.36 | 0.64 | 0.64 |
| Hybrid (embeddings + BM25) | 11 (version 1) | 0.55 | 0.64 | 0.73 |
| Embeddings only | 41 (version 2) | 0.29 | 0.39 | 0.44 |
| Hybrid (embeddings + BM25) | 41 (version 2) | 0.41 | 0.54 | 0.71 |

Hybrid search was chosen after looking at the misses on the first 11 questions, which is mild tuning on the test set. The 30 questions added later give a fairer check: on those alone, recall@5 rose from 0.37 (11 of 30) with embeddings only to 0.70 (21 of 30) with hybrid search. Across all 41 answerable questions, hybrid fixed 12 questions and broke 1.

### Generation, version 1 (11 questions, strict prompt)

Hybrid retrieval, top 4 passages, Gemini. Answers were graded by hand against the reference answer and the cited pages.

| Result | Count |
|---|---|
| Correct | 8 of 11 |
| Partial | 1 of 11 |
| Abstained although the answer existed | 2 of 11 |
| Unanswerable questions correctly refused | 3 of 3 |

The partial answer (question 9) captured that binarizing the disease label loses severity information, and the cited passage supports it, but it missed the borderline-case point in the reference.

### Generation, version 2 (48 questions, softer prompt)

The prompt was relaxed so the model answers with whatever the passages support instead of refusing broad questions.

[TODO: after running `python src/evaluate.py generate` over the full set, add the grade counts, how many of the 7 unanswerable questions were refused, and any hallucinations found.]

### Failure analysis

- **Underspecified questions.** Several evaluation questions refer to "that study" or "these authors" and give no way to tell which paper is meant (for example 26, 28, 32, 38, 40, 41 and 44). Retrieval cannot resolve this, so those misses say as much about the questions as about the system. The app also has no way to restrict a question to one paper.
- **Similar papers.** Every paper discusses logistic regression, decision trees and the UCI datasets, so general questions match many papers about equally. Question 43 asks about the 27 encoded features in one paper, but the team report (28 encoded features) ranked first.
- **Refusals caused by retrieval misses.** The L2 penalty and the decision tree temperature (T* = 2.10) both appear in the team report's chunks, but ranked below text from other papers, so the system refused instead of guessing, which is the safe failure.
- **Vocabulary mismatch.** One question asked about a "yes/no label", while the report says "binary label".
- **Similarity scores do not separate answerable from unanswerable questions.** In the hybrid run, the top similarity for answerable questions ranged from about 0.30 to 0.72, and for the unanswerable questions from 0.12 to 0.56. A fixed similarity threshold would either refuse answerable questions or accept unanswerable ones. This is the motivation for calibrated confidence (see next steps).
- **Broad questions.** Early on, "what features cause heart disease" was refused even though the papers discuss feature importance. The papers describe prediction, not causation, and the first prompt was strict. This motivated version 2 of the prompt.
- **Extraction noise.** Some PDFs produce spacing errors (for example "asubset"), which can leak into answers.

## Limitations

- Small evaluation set (48 questions). Correctness depends on keyword matching, and the answerable questions are fact lookups (numbers, names, short definitions), not broad or multi-paper questions.
- Eleven of the 41 answerable questions come from one document, the team report.
- Several papers contradict themselves (for example 87.5% overall accuracy and 88.52% for KNN in the same paper). The questions were written to avoid this, but it can still confuse answers.
- Grading was done by one person. Faithfulness (does the cited text support the claim) was checked by hand and is not an automated metric.
- The free Gemini quota (a small number of requests per day) limits how often the full evaluation can be run. Answers are cached in `data/llm_cache.json` to avoid repeat requests.

## Run it yourself

Requires Python 3.10-3.12 and a Gemini API key.

```powershell
git clone https://github.com/Nimra003/rag-papers.git
cd rag-papers
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
LLM_API_KEY=your_key_here
LLM_MODEL=gemini-3.8-flash
```

Model names change over time. If you get a "model not found" error, set `LLM_MODEL` to a current model available to your key.

Put your PDFs in `data/pdfs/`, then build the index and start the app:

```powershell
python src/ingest.py
python src/build_index.py
streamlit run src/app.py
```

Command line version:

```powershell
python src/generate.py "your question here"
```

Evaluation:

```powershell
python src/check_questions.py      # checks that question keywords exist in the chunks
python src/evaluate.py retrieval   # embeddings only
python src/evaluate.py hybrid      # embeddings + BM25
python src/evaluate.py generate    # asks the LLM every question, writes eval/results.json
python src/evaluate.py summary     # counts the grades you entered in results.json
```

## Project structure

```
src/ingest.py           PDFs -> data/chunks.json
src/build_index.py      chunks -> FAISS index
src/retrieve.py         embedding search and hybrid search
src/generate.py         prompt, Gemini call, validation, caching
src/app.py              Streamlit interface
src/evaluate.py         retrieval and generation evaluation
src/check_questions.py  sanity checks for the question file
eval/questions.json     evaluation questions
```

## Possible next steps

- Rewrite the underspecified questions so each one identifies its paper, then re-run the evaluation as version 3.
- Calibrated confidence scores with expected calibration error (ECE) and a risk-coverage curve, so low-confidence answers can be refused automatically.
- A cross-encoder reranker.