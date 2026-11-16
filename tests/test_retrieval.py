from collections import Counter

from aiact.bm25 import BM25, tokenize
from aiact.ingest import MAX_CHARS, find_references, naive_chunks
from aiact.tools import BINDING

# --- parsing and chunking ---------------------------------------------------------------------


def test_every_source_of_the_regulation_is_parsed(chunks):
    sources = {c.source_id for c in chunks}
    kinds = Counter(s.split("-")[0] for s in sources)
    assert kinds == {"art": 113, "rct": 180, "anx": 13}


def test_article_3_gives_one_chunk_per_definition(chunks):
    definitions = {c.term: c for c in chunks if c.kind == "definition"}
    assert len(definitions) == 68
    assert "utilisant sous sa propre autorité" in definitions["déployeur"].text


def test_chunks_carry_their_context(chunks):
    first = next(c for c in chunks if c.source_id == "art-6")
    assert first.title.startswith("Règles relatives à la classification")
    assert "Chapitre III" in first.context and "Section 1" in first.context
    assert first.indexed_text().startswith("Article 6 - ")
    assert not first.indexed_text(with_header=False).startswith("Article 6")


def test_chunks_stay_within_the_size_budget(chunks):
    # A chunk closes before the line that would overflow it; only an unbreakable line can exceed.
    oversized = [c.id for c in chunks if c.kind in ("article", "annex") and len(c.text) > 2 * MAX_CHARS]
    assert not oversized


def test_naive_baseline_covers_the_same_sources(chunks):
    assert {c.source_id for c in naive_chunks(chunks)} == {c.source_id for c in chunks}


def test_find_references_normalises_what_people_write():
    text = "Voir l'article 6, paragraphe 2, l'art. 50, l'annexe III, l'annexe 3 et le considérant 27 [Article 6]."
    assert find_references(text) == ["art-6", "art-50", "anx-III", "rct-27"]
    assert find_references("aucune référence ici") == []


# --- ranking ------------------------------------------------------------------------------------


def test_tokenizer_folds_accents_plurals_and_elisions():
    assert tokenize("Les systèmes d'IA à haut risque") == ["systeme", "ia", "haut", "risque"]


def test_bm25_prefers_the_document_with_the_rare_term():
    bm25 = BM25(["le marquage CE est apposé", "le système est documenté", "le système est surveillé"])
    scores = bm25.scores("marquage du système")
    assert scores.argmax() == 0


def test_reference_routing_puts_the_named_article_first(retriever):
    hits = retriever.search("Que dit l'article 14 ?", k=5)
    assert hits[0].chunk.source_id == "art-14"
    unrouted = retriever.search("Que dit l'article 14 ?", k=1, mode="dense", route_references=False)
    assert unrouted  # still returns something, just without the guarantee


def test_scope_filter_applies_to_both_rankers(retriever):
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.search("pratiques interdites manipulation", k=8, mode=mode, kinds=["recital"])
        assert hits and all(h.chunk.kind == "recital" for h in hits)


def test_hybrid_finds_the_prohibited_practices_article(retriever):
    hits = retriever.search("pratiques d'IA interdites", k=5, kinds=BINDING)
    assert "art-5" in [h.chunk.source_id for h in hits]
    assert all(h.bm25_rank or h.dense_rank for h in hits)
