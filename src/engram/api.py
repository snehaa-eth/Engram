"""HTTP API (FastAPI). Run with:  engram serve   (or: uvicorn engram.api:app)

Tenancy: if ENGRAM_API_KEYS is set ("key1:tenantA,key2:tenantB"), every request needs an
X-API-Key header and the tenant comes from the key, so a client can never pick another
tenant. Without keys (local development only), the X-Tenant-Id header is used.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engram import __version__
from engram.brain import Brain
from engram.config import Settings


@lru_cache(maxsize=1)
def get_brain() -> Brain:
    return Brain(Settings())


def get_tenant(x_api_key: str | None = Header(default=None),
               x_tenant_id: str | None = Header(default=None)) -> str:
    keys = get_brain().settings.api_key_map()
    if keys:
        if not x_api_key or x_api_key not in keys:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
        return keys[x_api_key]
    return x_tenant_id or "default"


class RememberIn(BaseModel):
    user_id: str
    content: str = Field(min_length=1)
    speaker: str = "user"
    source: str = "chat"
    trust: str | None = None
    occurred_at: str | None = None


class RecallIn(BaseModel):
    user_id: str
    query: str
    token_budget: int | None = Field(default=None, ge=50, le=20000)


class UserIn(BaseModel):
    user_id: str


app = FastAPI(title="Engram", version=__version__,
              description="Fast, low-token long-term memory for AI agents.")


WEB_DIR = Path(__file__).parent / "web"


@app.get("/", include_in_schema=False)
def brain_map() -> FileResponse:
    """The interactive 3D brain map."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/remember")
def remember(body: RememberIn, tenant: str = Depends(get_tenant)) -> dict:
    try:
        r = get_brain().remember(tenant, body.user_id, body.content, speaker=body.speaker,
                                 source=body.source, trust=body.trust, occurred_at=body.occurred_at)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return asdict(r)


@app.post("/v1/recall")
def recall(body: RecallIn, tenant: str = Depends(get_tenant)) -> dict:
    return asdict(get_brain().recall(tenant, body.user_id, body.query, token_budget=body.token_budget))


@app.post("/v1/sleep")
def sleep(body: UserIn, tenant: str = Depends(get_tenant)) -> dict:
    return asdict(get_brain().sleep(tenant, body.user_id))


@app.get("/v1/profile/{user_id}")
def profile(user_id: str, tenant: str = Depends(get_tenant)) -> dict:
    return {"user_id": user_id, "profile": get_brain().profile(tenant, user_id)}


@app.delete("/v1/episodes/{episode_id}")
def forget_episode(episode_id: str, tenant: str = Depends(get_tenant)) -> dict:
    return {"facts_deleted": get_brain().forget_episode(tenant, episode_id)}


@app.delete("/v1/users/{user_id}")
def forget_user(user_id: str, tenant: str = Depends(get_tenant)) -> dict:
    return {"facts_deleted": get_brain().forget_user(tenant, user_id)}


@app.get("/v1/stats")
def stats(tenant: str = Depends(get_tenant)) -> dict:
    return get_brain().stats(tenant)


@app.get("/v1/brains")
def brains(tenant: str = Depends(get_tenant)) -> dict:
    return {"brains": get_brain().brains(tenant)}


@app.get("/v1/graph/{user_id}")
def graph(user_id: str, tenant: str = Depends(get_tenant)) -> dict:
    return get_brain().graph(tenant, user_id)
