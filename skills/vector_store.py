"""Semantic memory over Qdrant -- the meaning-based half of shared memory.

Where ``obsidian_vault.search`` finds notes by exact text, this finds them by
*meaning*: "cooling design" surfaces a note about thermal management even if it
never uses that phrase. It's the backbone of Forge's RAG and Axiom's
duplicate detection.

Architecture choice that matters for this stack: embeddings are computed by
**Ollama over HTTP**, not by an in-process model. Ollama holds the embedder in
memory once and every agent calls it -- so eleven agents don't each load torch.
Qdrant stores the vectors; this module is a thin, safe wrapper over both.

Typical use:

    upsert("engineering", doc_id="proj/power.md", text=note_body, payload={"project": "psu"})
    hits = query("engineering", "how did I handle heat?", limit=5)

IDs are arbitrary strings (a note path, a job id). Qdrant requires int/UUID
point ids, so each string id is hashed to a deterministic UUID -- upserting the
same ``doc_id`` again updates in place rather than duplicating.
"""

from __future__ import annotations

import json
import urllib.request
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

# Stable namespace so the same doc_id always maps to the same Qdrant point id.
_ID_NAMESPACE = uuid.UUID("a3f1c0de-1234-4abc-9def-0123456789ab")
_EMBED_TIMEOUT = 30

_client: QdrantClient | None = None


def _qdrant() -> QdrantClient:
    """Lazily create one shared Qdrant client for the process."""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            api_key=config.qdrant_api_key,
        )
    return _client


def _embed(text: str) -> list[float]:
    """Turn text into a vector via Ollama's embedding endpoint."""
    payload = json.dumps({"model": config.embed_model, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{config.ollama_host}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_EMBED_TIMEOUT) as resp:
        vector = json.loads(resp.read())["embedding"]
    if not vector:
        raise ValueError("Ollama returned an empty embedding")
    return vector


def _point_id(doc_id: str) -> str:
    """Deterministic UUID for an arbitrary string id (so re-upsert updates)."""
    return str(uuid.uuid5(_ID_NAMESPACE, doc_id))


def _ensure_collection(name: str, dim: int) -> None:
    """Create the collection on first use, sized to the embedding dimension."""
    client = _qdrant()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("created Qdrant collection '%s' (dim=%d)", name, dim)


# -- operations ------------------------------------------------------------

@skill
def upsert(collection: str, doc_id: str, text: str, payload: dict | None = None) -> Result:
    """Embed ``text`` and store/update it under ``doc_id`` in ``collection``.

    The collection is created automatically on first use. The original id and
    text are kept in the payload so query results are self-describing.
    """
    vector = _embed(text)
    _ensure_collection(collection, len(vector))
    body = {**(payload or {}), "doc_id": doc_id, "text": text}
    _qdrant().upsert(
        collection_name=collection,
        points=[PointStruct(id=_point_id(doc_id), vector=vector, payload=body)],
    )
    log.info("upserted '%s' into '%s'", doc_id, collection)
    return Result.success({"doc_id": doc_id, "collection": collection})


@skill
def query(collection: str, text: str, limit: int = 5) -> Result:
    """Semantic search: return the ``limit`` closest entries to ``text``.

    Each hit is {'doc_id', 'score', 'payload'} with score in [0,1] (cosine).
    Returns an empty list if the collection doesn't exist yet.
    """
    client = _qdrant()
    if not client.collection_exists(collection):
        return Result.success([])
    vector = _embed(text)
    found = client.query_points(
        collection_name=collection, query=vector, limit=limit, with_payload=True
    ).points
    hits = [
        {"doc_id": p.payload.get("doc_id"), "score": round(p.score, 4), "payload": p.payload}
        for p in found
    ]
    return Result.success(hits)


@skill
def delete(collection: str, doc_id: str) -> Result:
    """Remove a single entry by its ``doc_id``."""
    client = _qdrant()
    if not client.collection_exists(collection):
        return Result.failure(f"collection '{collection}' does not exist")
    client.delete(collection_name=collection, points_selector=[_point_id(doc_id)])
    log.info("deleted '%s' from '%s'", doc_id, collection)
    return Result.success({"doc_id": doc_id})


@skill
def count(collection: str) -> Result:
    """How many entries a collection holds (0 if it doesn't exist)."""
    client = _qdrant()
    if not client.collection_exists(collection):
        return Result.success(0)
    return Result.success(client.count(collection_name=collection).count)
