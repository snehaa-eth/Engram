from fastapi.testclient import TestClient

from engram import Brain, Settings
from engram import api


def make_client(tmp_path, monkeypatch, keys=""):
    s = Settings(db_path=str(tmp_path / "api.db"), llm="fake", embedder="hash",
                 cache_dir=str(tmp_path / "cache"), use_cache=False, api_keys=keys)
    brain = Brain(s)
    monkeypatch.setattr(api, "get_brain", lambda: brain)
    return TestClient(api.app)


def test_remember_and_recall_over_http(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    h = {"X-Tenant-Id": "acme"}
    assert c.post("/v1/remember", json={"user_id": "u1", "content": "I live in Delhi."}, headers=h).status_code == 200
    r = c.post("/v1/recall", json={"user_id": "u1", "query": "where does the user live"}, headers=h).json()
    assert "Delhi" in r["context"]


def test_api_key_decides_tenant(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, keys="keyA:acme,keyB:globex")
    assert c.post("/v1/recall", json={"user_id": "u", "query": "x"}).status_code == 401
    c.post("/v1/remember", json={"user_id": "u", "content": "Secret plan is Bluebird launch."},
           headers={"X-API-Key": "keyA"})
    # Tenant B cannot reach tenant A's data, even if it claims tenant A in a header.
    r = c.post("/v1/recall", json={"user_id": "u", "query": "Bluebird launch plan"},
               headers={"X-API-Key": "keyB", "X-Tenant-Id": "acme"}).json()
    assert r["facts"] == []


def test_brain_map_endpoints(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    h = {"X-Tenant-Id": "acme"}
    c.post("/v1/remember", json={"user_id": "sneha", "content": "I live in Delhi with Rahul."}, headers=h)
    c.post("/v1/remember", json={"user_id": "sneha", "content": "I live in Bangalore with Rahul now."}, headers=h)
    c.post("/v1/sleep", json={"user_id": "sneha"}, headers=h)

    assert [b["user_id"] for b in c.get("/v1/brains", headers=h).json()["brains"]] == ["sneha"]
    assert c.get("/v1/brains", headers={"X-Tenant-Id": "globex"}).json()["brains"] == []

    g = c.get("/v1/graph/sneha", headers=h).json()
    facts = [n for n in g["nodes"] if n["type"] == "fact"]
    assert len(facts) == 2 and sum(not f["active"] for f in facts) == 1   # history is kept, marked retired
    assert any(n["type"] == "entity" and n["name"] == "Rahul" for n in g["nodes"])
    assert {l["type"] for l in g["links"]} >= {"mentions", "replaced"}
    assert all(f["episode"]["content"] for f in facts)                     # each neuron knows its source

    assert c.get("/", headers=h).status_code == 200                      # the 3D map page is served
