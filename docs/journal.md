# Journal: what was tried, what it gave, what I keep

All figures come from `python -m eval.run_retrieval` (58 questions with a known source); the letters
refer to the table in the README.

| # | Attempt | Result | Lesson |
|---|---|---|---|
| 1 | Baseline: fixed 1000-character windows, hybrid search (C) | recall@5 0.67, MRR 0.497 | A reference point before any idea. |
| 2 | Chunks that follow the articles and paragraphs (D) | recall@5 0.64: slightly **worse** | Clean boundaries are not enough. A paragraph of Article 26 never says which article it belongs to. |
| 3 | Prefix every chunk with "Article N - title (chapter > section)" (G) | recall@5 0.69, MRR 0.532; BM25 alone 0.78 (E) | The title carries the topic. Lexical search benefits most. |
| 4 | Surprise: hybrid (G) is below BM25 alone (E) | dense alone: recall@5 0.57 (F) | Fusion averages a good ranker with a bad one. Find out why dense is bad before blaming the fusion. |
| 5 | Hypothesis: the embedding model does not read the whole chunk. Test: two texts identical for 125 tokens, different after | cosine similarity = 1.0 | `paraphrase-multilingual-MiniLM` truncates at 128 tokens, without warning. With 1600-character chunks it saw about a third of each. |
| 6 | Chunks of 700 characters (H, I) | dense recall@1 0.28 → 0.38; hybrid recall@5 0.69 → 0.74 | Chunk size must follow the encoder's window, not the LLM's context. |
| 7 | Looked at the failures: recitals rank above the articles they explain | e.g. for the chatbot question, recitals 132-134 come before Article 50 | Recitals paraphrase the law in plain words: semantically closest, legally weakest. |
| 8 | Search articles and annexes by default, recitals on request (J) | recall@1 0.41 → 0.62, recall@5 0.81 | Largest single gain, and it came from knowing the document, not from a better model. |
| 9 | "Que dit l'article 14 ?" still failed: nothing in the question resembles the content | — | A reference is a key, not a query. |
| 10 | Reference routing: sources named in the query go first (K) | recall@1 0.69, recall@5 0.86, MRR 0.761 | Deterministic code where the task is deterministic. |

## Still failing

Eight questions miss at 5 (q01, q06, q29, q34, q41, q45, q46, q50). All are worded far from the text:
"mon chatbot" vs "systèmes d'IA destinés à interagir directement avec des personnes physiques",
"trier des CV" vs "recrutement ou sélection de personnes physiques". Retrieval alone will not close
this gap with a small encoder; the agent can, by rewriting the query in the vocabulary of the
regulation before searching again. Measuring how often it does is the job of `eval/run_answers.py`.

## Not tried yet

- a larger multilingual encoder (e5, bge-m3) or a cross-encoder reranker on the top 30;
- query expansion by the LLM before the first search (costs one call per question);
- weighting BM25 and dense in the fusion instead of treating them equally;
- an English corpus alongside the French one.
