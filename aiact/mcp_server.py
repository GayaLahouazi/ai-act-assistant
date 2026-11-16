"""Expose the regulation tools over the Model Context Protocol (MCP).

    python -m aiact.mcp_server          # stdio transport

Any MCP client (an IDE assistant, a desktop chat application, another agent) can then search and
read the AI Act with the very same functions the built-in agent uses. Requires ``pip install mcp``.
"""

from __future__ import annotations

try:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # SDK 1.x, where the same class was called FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer

from .config import get_settings
from .retriever import load_retriever
from .tools import Toolbox

mcp = MCPServer("ai-act")
_toolbox: Toolbox | None = None


def toolbox() -> Toolbox:
    global _toolbox
    if _toolbox is None:  # loaded on first use, so that listing the tools stays instantaneous
        settings = get_settings()
        _toolbox = Toolbox(load_retriever(settings), settings.top_k, settings.retrieval_mode)
    return _toolbox


@mcp.tool()
def search_regulation(query: str, scope: str = "binding") -> str:
    """Hybrid search (BM25 + embeddings) in the French text of the EU AI Act, Regulation (EU) 2024/1689.

    scope: "binding" (articles and annexes, default), "recitals" or "all".
    Returns the most relevant passages, each prefixed by its reference.
    """
    return toolbox().search_regulation(query, scope)


@mcp.tool()
def read_source(reference: str) -> str:
    """Full text of one article, recital or annex, e.g. "article 50", "annexe III", "considérant 27"."""
    return toolbox().read_source(reference)


@mcp.tool()
def lookup_definition(term: str) -> str:
    """Official definition of a term from Article 3, e.g. "déployeur", "fournisseur", "système d'IA"."""
    return toolbox().lookup_definition(term)


if __name__ == "__main__":
    mcp.run()
