"""Command line:  engram remember | recall | sleep | profile | stats | forget-user | serve"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from engram.brain import Brain


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="engram", description="Engram memory layer for AI agents")
    p.add_argument("--tenant", default="default", help="company / workspace id")
    p.add_argument("--user", default="me", help="user id inside the tenant")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("remember", help="store a message")
    r.add_argument("text")
    r.add_argument("--speaker", default="user", choices=["user", "assistant", "tool", "document"])
    r.add_argument("--trust", choices=["user", "assistant", "external"])

    q = sub.add_parser("recall", help="get the memory packet for a query")
    q.add_argument("query")
    q.add_argument("--budget", type=int, help="token budget for the packet")

    sub.add_parser("sleep", help="consolidate: retire duplicates/outdated facts, rebuild profile")
    sub.add_parser("profile", help="show the user's profile")
    sub.add_parser("stats", help="counts for the tenant")
    sub.add_parser("forget-user", help="erase everything stored for the user")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        uvicorn.run("engram.api:app", host=a.host, port=a.port)
        return

    brain = Brain()
    if a.cmd == "remember":
        out = asdict(brain.remember(a.tenant, a.user, a.text, speaker=a.speaker, trust=a.trust))
    elif a.cmd == "recall":
        out = asdict(brain.recall(a.tenant, a.user, a.query, token_budget=a.budget))
    elif a.cmd == "sleep":
        out = asdict(brain.sleep(a.tenant, a.user))
    elif a.cmd == "profile":
        out = {"profile": brain.profile(a.tenant, a.user)}
    elif a.cmd == "stats":
        out = brain.stats(a.tenant)
    else:
        out = {"facts_deleted": brain.forget_user(a.tenant, a.user)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
