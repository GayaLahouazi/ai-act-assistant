"""The three tools the agent can call, with their JSON schemas (OpenAI "function calling" format).

Each tool is a plain Python function over the retriever: the same functions are exposed to the LLM
agent (``agent.py``) and to any MCP client (``mcp_server.py``).
"""

from __future__ import annotations

import difflib
import json
import unicodedata

from .ingest import find_references
from .retriever import Retriever

MAX_SOURCE_CHARS = 7000

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulation",
            "description": (
                "Recherche sémantique et lexicale dans le règlement (UE) 2024/1689 (AI Act). "
                "À utiliser pour toute question de fond. Retourne les passages les plus pertinents "
                "avec leur référence (article, considérant ou annexe)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête autonome en français, avec le vocabulaire du règlement si possible.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["binding", "recitals", "all"],
                        "description": "'binding' (défaut) = articles et annexes, le dispositif contraignant ; "
                        "'recitals' = considérants, utiles pour la motivation et l'interprétation ; "
                        "'all' = les deux.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "Lit le texte intégral d'un article, d'un considérant ou d'une annexe à partir de sa "
                "référence exacte. À utiliser quand l'utilisateur cite une référence, ou pour lire en "
                "entier un article repéré par la recherche."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Par exemple 'article 50', 'annexe III' ou 'considérant 27'.",
                    }
                },
                "required": ["reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_definition",
            "description": (
                "Retourne la définition officielle d'un terme donnée à l'article 3 "
                "(par exemple 'déployeur', 'fournisseur', 'système d'IA', 'incident grave')."
            ),
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string", "description": "Terme à définir, en français."}},
                "required": ["term"],
            },
        },
    },
]

# The 180 recitals paraphrase the articles in plainer words, so they tend to outrank them in a
# search, yet only articles and annexes are binding. Searching the binding text by default raised
# recall@1 from 0.41 to 0.62 on the evaluation set (see docs/journal.md).
BINDING = ["article", "definition", "annex"]
SCOPES = {"binding": BINDING, "recitals": ["recital"], "all": None}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip()


class Toolbox:
    """Executes tool calls and remembers which sources were shown to the model during a turn.

    ``seen`` (sources returned by a tool) and ``mentioned`` (references quoted inside those texts)
    are what the citation check in ``guardrails.py`` compares the final answer against.
    """

    def __init__(self, retriever: Retriever, top_k: int = 6, mode: str = "hybrid"):
        self.retriever, self.top_k, self.mode = retriever, top_k, mode
        self.seen: dict[str, None] = {}
        self.mentioned: set[str] = set()
        self._definitions = {
            _fold(c.term): c for c in retriever.chunks if c.kind == "definition"
        }

    # --- tools ----------------------------------------------------------------------------------

    def search_regulation(self, query: str, scope: str = "binding") -> str:
        hits = self.retriever.search(query, self.top_k, self.mode, SCOPES.get(scope, BINDING))
        if not hits:
            return "Aucun passage trouvé."
        blocks = []
        for hit in hits:
            self.record(hit.chunk.source_id, f"{hit.chunk.title}\n{hit.chunk.text}")
            blocks.append(f"[{hit.chunk.label}] {hit.chunk.title}\n{hit.chunk.text}")
        return "\n\n---\n\n".join(blocks)

    def read_source(self, reference: str) -> str:
        ids = find_references(reference)
        if not ids:
            return f"Référence non reconnue : '{reference}'. Formats attendus : 'article 50', 'annexe III', 'considérant 27'."
        source_id = ids[0]
        text, meta = self.retriever.source_text(source_id), self.retriever.source_meta(source_id)
        if text is None:
            return f"'{reference}' n'existe pas : le règlement compte 113 articles, 180 considérants et 13 annexes."
        self.record(source_id, text)
        if len(text) > MAX_SOURCE_CHARS:
            text = text[:MAX_SOURCE_CHARS] + "\n[... texte tronqué : utiliser search_regulation pour cibler un passage]"
        return f"[{meta['label']}] {meta['title']}\n{text}"

    def lookup_definition(self, term: str) -> str:
        key = _fold(term)
        chunk = self._definitions.get(key)
        if chunk is None:
            # Tolerate plurals, typos and partial terms: "fournisseurs", "deployeur", "donnees biometriques".
            close = difflib.get_close_matches(key, self._definitions, n=1, cutoff=0.75)
            partial = [k for k in self._definitions if key and (key in k or k in key)]
            chunk = self._definitions.get(close[0] if close else min(partial, key=len, default=""))
        if chunk is None:
            terms = ", ".join(sorted(c.term for c in self._definitions.values()))
            return f"'{term}' n'est pas défini à l'article 3. Termes définis : {terms}."
        self.record("art-3", chunk.text)
        return f"[Article 3] point {chunk.id.split('#')[1]}\n{chunk.text}"

    # --- dispatch ---------------------------------------------------------------------------------

    def record(self, source_id: str, text: str) -> None:
        self.seen.setdefault(source_id)
        self.mentioned.update(find_references(text))

    def call(self, name: str, arguments: str) -> str:
        """Run a tool call coming from the model. Errors are returned as text so the model can retry."""
        try:
            kwargs = json.loads(arguments or "{}")
            if not isinstance(kwargs, dict):
                raise ValueError("arguments must be a JSON object")
            if name == "search_regulation":
                return self.search_regulation(str(kwargs["query"]), str(kwargs.get("scope", "binding")))
            if name == "read_source":
                return self.read_source(str(kwargs["reference"]))
            if name == "lookup_definition":
                return self.lookup_definition(str(kwargs["term"]))
            return f"Outil inconnu : {name}."
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return f"Appel invalide de {name} ({exc!r}). Corrige les arguments et réessaie."
