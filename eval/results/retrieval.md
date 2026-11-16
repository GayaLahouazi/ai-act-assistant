Retrieval benchmark on 58 questions, embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

| | Chunking | Ranker | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|---|---|
| A | fixed 1000-char windows | bm25 | 849 | 0.19 | 0.60 | 0.69 | 0.397 | 1 |
| B | fixed 1000-char windows | dense | 849 | 0.24 | 0.57 | 0.62 | 0.413 | 34 |
| C | fixed 1000-char windows | hybrid | 849 | 0.38 | 0.55 | 0.67 | 0.497 | 24 |
| D | structure-aware, 1600 chars | hybrid | 578 | 0.29 | 0.59 | 0.64 | 0.452 | 19 |
| E | D + contextual header | bm25 | 578 | 0.40 | 0.71 | 0.78 | 0.558 | 1 |
| F | D + contextual header | dense | 578 | 0.28 | 0.48 | 0.57 | 0.400 | 24 |
| G | D + contextual header | hybrid | 578 | 0.38 | 0.62 | 0.69 | 0.532 | 24 |
| H | G with 700-char chunks | dense | 950 | 0.38 | 0.57 | 0.60 | 0.479 | 38 |
| I | G with 700-char chunks | hybrid | 950 | 0.41 | 0.66 | 0.74 | 0.565 | 33 |
| J | I, binding text only (no recitals) | hybrid | 950 | 0.62 | 0.72 | 0.81 | 0.698 | 76 |
| K | J + reference routing | hybrid | 950 | 0.69 | 0.78 | 0.86 | 0.761 | 63 |
