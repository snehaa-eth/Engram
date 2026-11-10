from engram.router import NONE


def texts(result):
    return " ".join(f["text"] for f in result.facts)


def test_remember_then_recall(brain):
    brain.remember("acme", "sneha", "I live in Delhi. I prefer Python for backend work.")
    r = brain.recall("acme", "sneha", "where does the user live")
    assert "Delhi" in texts(r)
    assert r.context.startswith("Relevant memory:")
    assert r.context_tokens > 0


def test_exact_id_found_by_keyword_search(brain):
    brain.remember("acme", "sneha", "My order number is ZX-9931 from the Mumbai store.")
    brain.remember("acme", "sneha", "I enjoy hiking on weekends with friends.")
    r = brain.recall("acme", "sneha", "ZX-9931")
    assert "ZX-9931" in texts(r)


def test_small_talk_skips_retrieval(brain):
    brain.remember("acme", "sneha", "I live in Delhi with my family.")
    r = brain.recall("acme", "sneha", "thanks!")
    assert r.route == NONE
    assert r.context == "" and r.context_tokens == 0


def test_unrelated_query_returns_nothing(brain):
    brain.remember("acme", "sneha", "I live in Delhi with my family.")
    r = brain.recall("acme", "sneha", "quantum chromodynamics lattice gauge")
    assert r.facts == []


def test_tenants_are_isolated(brain):
    brain.remember("acme", "sneha", "The launch codename is Bluebird for the March release.")
    r = brain.recall("globex", "sneha", "launch codename Bluebird")
    assert r.facts == []
    assert brain.stats("globex")["valid_facts"] == 0


def test_users_are_isolated_but_shared_memory_is_visible(brain):
    brain.remember("acme", "alice", "Alice keeps her salary details private.")
    brain.remember("acme", "_shared", "The office wifi network is called AcmeNet.")
    r = brain.recall("acme", "bob", "Alice salary details office wifi network")
    found = texts(r)
    assert "AcmeNet" in found
    assert "salary" not in found


def test_sleep_retires_outdated_fact(brain):
    brain.remember("acme", "sneha", "I live in Delhi.", occurred_at="2026-01-01T00:00:00+00:00")
    brain.remember("acme", "sneha", "I live in Bangalore.", occurred_at="2026-06-01T00:00:00+00:00")
    report = brain.sleep("acme", "sneha")
    assert report.outdated_retired == 1
    r = brain.recall("acme", "sneha", "where does the user live")
    assert "Bangalore" in texts(r) and "Delhi" not in texts(r)
    assert brain.stats("acme")["retired_facts"] == 1  # history kept, not deleted


def test_sleep_removes_duplicates(brain):
    brain.remember("acme", "sneha", "I prefer dark mode in every editor.")
    brain.remember("acme", "sneha", "I prefer dark mode in every editor.")
    report = brain.sleep("acme", "sneha")
    assert report.duplicates_retired == 1
    assert brain.stats("acme")["valid_facts"] == 1


def test_external_content_cannot_overwrite_user_fact(brain):
    brain.remember("acme", "sneha", "I live in Delhi.", occurred_at="2026-01-01T00:00:00+00:00")
    brain.remember("acme", "sneha", "I live in Paris.", speaker="document", trust="external",
                   occurred_at="2026-06-01T00:00:00+00:00")
    report = brain.sleep("acme", "sneha")
    assert report.outdated_retired == 0
    r = brain.recall("acme", "sneha", "where does the user live")
    assert "Delhi" in texts(r)


def test_token_budget_is_respected(brain):
    for i in range(40):
        brain.remember("acme", "sneha", f"The user visited project site number {i} in Pune today.")
    r = brain.recall("acme", "sneha", "project site visited in Pune", token_budget=80)
    assert 0 < r.context_tokens <= 100  # budget plus the short header line


def test_forget_episode_erases_derived_facts(brain):
    ep = brain.remember("acme", "sneha", "My passport number is K1234567.").episode_id
    assert brain.forget_episode("acme", ep) == 1
    assert brain.recall("acme", "sneha", "passport number K1234567").facts == []


def test_profile_built_during_sleep(brain):
    brain.remember("acme", "sneha", "I prefer short answers. I live in Delhi.")
    report = brain.sleep("acme", "sneha")
    assert "Delhi" in report.profile
    assert brain.recall("acme", "sneha", "anything").profile == report.profile
