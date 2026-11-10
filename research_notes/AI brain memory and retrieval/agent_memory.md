# Long-Term Memory Architectures for LLM Agents ("AI Brain"), 2014–2026

Verification legend: **[V]** means checked against the primary source (arXiv abstract/HTML, official blog, or GitHub) during this research session (Nov 2026). **[M]** means it comes from the researcher's prior knowledge of the primary paper and was NOT re-fetched this session. Treat [M] numbers as needing a spot-check before publication. The source link is still the primary paper.

---

## Q1. How does each system write, consolidate, forget, and retrieve memories, and what data structure does it use?

### Takeaway
The field went from differentiable memory inside the network (2014–2016), to non-parametric retrieval memory for LMs (2019–2023), to LLM-managed external stores (2023 onward). By 2025–2026 the frontier systems differ mainly in (a) **when consolidation happens**: at write time (Mem0, Zep, A-Mem, fact stores) or at read time (raw logs plus agentic search, as in Letta's filesystem result); and (b) **how they represent time and contradiction**: delete (Mem0 base), mark invalid (Mem0g), or keep bi-temporal validity intervals (Zep/Graphiti). The surveys now frame memory as a *trainable policy* rather than passive storage.

### Cited Findings

**Precursors (differentiable / parametric-adjacent memory)**
- **Memory Networks** (Weston, Chopra, Bordes, 2014). An explicit memory array with I/G/O/R components (input map, generalization/write, output/read, response). Reads were hard attention, supervised on supporting facts. The follow-up End-to-End Memory Networks (Sukhbaatar et al., 2015) replaced this with soft attention over memory slots, trainable without supporting-fact supervision. [M] — [arXiv 1410.3916](https://arxiv.org/abs/1410.3916); [arXiv 1503.08895](https://arxiv.org/abs/1503.08895)
- **Neural Turing Machines** (Graves, Wayne, Danihelka, 2014). An RNN controller with differentiable read/write heads over an external matrix, using content- and location-based addressing, plus erase/add writes. The Differentiable Neural Computer (Nature, 2016) added dynamic allocation and temporal link matrices. These were the first "write/forget/read" memory primitives, but they did not scale to LLMs. [M] — [arXiv 1410.5401](https://arxiv.org/abs/1410.5401)
- **kNN-LM** (Khandelwal et al., ICLR 2020). It interpolates the LM's next-token distribution with a k-nearest-neighbour distribution over a datastore of (context-embedding → next-token) pairs, with no retraining. It reported a new SOTA perplexity of 15.79 on WikiText-103 (≈2.9 points better than the base). The memory is a flat vector datastore, written once and never forgotten. [M] — [arXiv 1911.00172](https://arxiv.org/abs/1911.00172)
- **RETRO** (Borgeaud et al., DeepMind, 2021). It retrieves chunks from a ~2-trillion-token database by frozen-BERT kNN and conditions on them through chunked cross-attention. The authors report performance comparable to GPT-3/Jurassic-1 with ~25× fewer parameters. This is "memory" as a frozen external corpus. [M] — [arXiv 2112.04426](https://arxiv.org/abs/2112.04426)
- **LongMem** (Wang et al., NeurIPS 2023). It uses a frozen backbone LLM as a memory encoder and caches past attention key/value pairs in a memory bank (up to ~65k tokens). A trainable residual "SideNet" retrieves and fuses those cached KV pairs. The memory is latent/KV, with FIFO-style eviction when the bank is full. [M] — [arXiv 2306.07174](https://arxiv.org/abs/2306.07174)

**LLM-agent era (external, LLM-managed memory)**
- **Generative Agents** (Park et al., UIST 2023). The **memory stream** is an append-only natural-language log of observations with timestamps.
  - *Retrieval:* score = α·recency + α·importance + α·relevance. All weights are 1, and each term is min-max normalized. Recency is an exponential decay (factor 0.995 per sandbox game-hour since last access). Importance is an LLM-rated 1–10 "poignancy" score assigned at write time. Relevance is embedding cosine similarity to the query.
  - *Consolidation:* **Reflection** fires when the summed importance of recent events exceeds a threshold (150). The LLM asks itself salient high-level questions, retrieves evidence, and writes higher-level "reflections" back into the stream, forming trees of abstractions.
  - *Forgetting:* no explicit forgetting. Only recency decay in ranking.
  - [M] — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- **Reflexion** (Shinn et al., NeurIPS 2023). "Verbal reinforcement learning": after a trial fails, the agent writes a natural-language self-reflection into an episodic buffer that conditions later trials. The buffer is bounded (a sliding window of the last ~1–3 reflections). It reported 91% pass@1 on HumanEval vs 80% for GPT-4 at the time. This is procedural/experiential memory as text, not user memory. [M] — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
- **MemoryBank / SiliconFriend** (Zhong et al., AAAI 2024).
  - *Store:* conversation logs, daily event summaries, and a global user-personality summary, indexed with FAISS dense retrieval.
  - *Forgetting:* uses the **Ebbinghaus forgetting curve**, R = e^(−t/S). t is time since last recall. S is memory strength, incremented by 1 and with t reset each time the memory is recalled. Memories with low retention are forgotten or deprioritized.
  - [M] — [arXiv 2305.10250](https://arxiv.org/abs/2305.10250)
- **MemGPT → Letta** (Packer et al., 2023). An "LLM as operating system" design.
  - *Tiers:* **main context** (system instructions, a writable "working context"/core memory block, and a FIFO message queue) and **external context**. External context has *recall storage* (full message history, searchable) and *archival storage* (a vector DB for arbitrary text).
  - *Self-editing:* the LLM pages data in and out through function calls (e.g., core_memory_append/replace, archival_memory_insert/search, conversation_search).
  - *Memory pressure:* a system warning fires when the context nears capacity (~70%). When the queue overflows, evicted messages are recursively summarized.
  - The project became the company Letta (letta-ai/letta).
  - [M] — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560); DMR score for MemGPT (93.4% w/ gpt-4-turbo) is restated in the Zep paper [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- **A-Mem** (Xu et al., NeurIPS 2025). A **Zettelkasten**-style network of notes.
  - *Write:* each new memory becomes a note with structured attributes (contextual description, keywords, tags, embedding).
  - *Linking:* the system analyzes historical memories and creates links where it finds meaningful similarity.
  - *Memory evolution:* adding a new note can **rewrite the context and attributes of existing linked notes**.
  - The abstract claims superiority over SOTA baselines across six foundation models, with no numbers in the abstract.
  - [V] — [arXiv 2502.12110](https://arxiv.org/abs/2502.12110)
- **Mem0** (Chhikara et al., Apr 2025).
  - *Extraction phase:* the prompt holds a rolling conversation summary (refreshed asynchronously), the last 10 messages, and the new message pair. The LLM extracts candidate facts from it.
  - *Update phase:* for each candidate, the system retrieves the top-10 most similar existing memories by vector similarity. The LLM then picks one tool-call operation:
    - **ADD:** no equivalent memory exists.
    - **UPDATE:** augment an existing memory, only if the new fact is more informative.
    - **DELETE:** the new information contradicts an existing memory.
    - **NOOP:** no change.
  - *Data structure:* a vector store of natural-language facts. Forgetting happens only through DELETE on contradiction.
  - [V] — [arXiv 2504.19413 (HTML)](https://arxiv.org/html/2504.19413)
- **Mem0g (graph variant).** A directed labeled graph in Neo4j: entity nodes carry type, embedding, and creation timestamp, and edges are (source, relation, destination) triplets.
  - *Extraction:* an LLM entity extractor, then an LLM relationship generator.
  - *Entity resolution:* by embedding-similarity threshold.
  - *Conflicts:* a conflict detector plus an LLM resolver **mark obsolete edges invalid rather than deleting them**, which preserves history for temporal reasoning.
  - GPT-4o-mini is used for extraction and updates.
  - [V] — [arXiv 2504.19413 (HTML)](https://arxiv.org/html/2504.19413)
- **Zep / Graphiti** (Rasmussen et al., Jan 2025). A three-tier temporally aware knowledge graph:
  - **Tiers:**
    - (1) An **episode subgraph** of raw messages, text, or JSON, a non-lossy source linked to extracted entities by episodic edges.
    - (2) A **semantic entity subgraph** of resolved entities and fact edges, which can act as hyper-edges.
    - (3) A **community subgraph** of label-propagation clusters with map-reduce summaries, extended dynamically.
  - **Bi-temporal model:** timeline T (when events happened) records t_valid/t_invalid. Transactional timeline T′ (when ingested) records t′_created/t′_expired.
  - **Contradictions:** an LLM compares a new edge with related edges. When there is a temporally overlapping contradiction, the old edge's t_invalid is set to the new edge's t_valid. Nothing is deleted.
  - **Retrieval:** cosine + BM25 (Neo4j/Lucene) + BFS n-hop. Rerankers are RRF, MMR, episode-mentions, node-distance, and cross-encoder.
  - [V] — [arXiv 2501.13956 (HTML)](https://arxiv.org/html/2501.13956)
- **HippoRAG** (Gutiérrez et al., NeurIPS 2024). Modelled on hippocampal indexing theory.
  - *Write:* the LLM performs OpenIE to build a schemaless KG of noun-phrase nodes and triples (the "hippocampal index").
  - *Retrieval:* query entities seed **Personalized PageRank** over the KG, and passages are ranked by PPR mass. This gives single-step multi-hop retrieval.
  - Reported up to ~20% better on multi-hop QA, with single-step retrieval 10–30× cheaper and 6–13× faster than iterative retrieval (IRCoT).
  - **HippoRAG 2** ("From RAG to Memory", 2025) adds passage nodes and recognition-memory filtering, and frames itself explicitly as non-parametric continual-learning memory.
  - [M] — [arXiv 2405.14831](https://arxiv.org/abs/2405.14831); [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- **Letta filesystem baseline (Aug 2025).** It stores raw conversation history as files, and the agent must call `search_files` repeatedly until it chooses to answer. There is no write-time consolidation at all. All "memory intelligence" lives in the agent's retrieval policy. [V] — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)

**Cognitive-architecture framing**
- **CoALA** (Sumers, Yao, Narasimhan, Griffiths, TMLR 2024). It defines **working memory** (current-cycle variables/context) and three long-term stores: **episodic** (experiences), **semantic** (facts about the world/user), and **procedural** (skills: LLM weights plus agent code/prompts). Actions are split into internal (retrieval, reasoning, learning = writing to memory) and external (grounding), inside a decision cycle. [M] — [arXiv 2309.02427](https://arxiv.org/abs/2309.02427)
- **"Memory in the Age of AI Agents"** (Hu et al., Dec 2025, 47 authors). It argues that long/short-term categories are insufficient and proposes a new taxonomy:
  - **Forms:** token-level, parametric, latent.
  - **Functions:** factual, experiential, working.
  - **Dynamics:** formation, evolution, retrieval.
  - It separates agent memory from LLM memory, RAG, and context engineering, and compiles benchmarks and open-source frameworks.
  - [V] — [arXiv 2512.13564](https://arxiv.org/abs/2512.13564)
- **"A Survey of Agent Memory in the Second Half"** (Huang et al., v1 Jan 2026, v4 Aug 2026, 60 authors, TMLR Survey Certification). Its taxonomy has three axes:
  - **Substrate:** internal parametric vs external retrieval stores.
  - **Cognitive mechanism:** sensory, working, episodic, semantic, procedural.
  - **Subject:** user-centric personalization vs agent-centric experience.
  - It also covers learned policies over memory operations: RL-trained context curation, decision-time experience consolidation, and portable/shareable skills.
  - It claims "hundreds" of memory papers appeared in 2025, and that memory management is becoming a **trainable capability**.
  - [V] — [arXiv 2602.06052](https://arxiv.org/abs/2602.06052)

### Inferences
- A useful classification of the systems by data structure:
  - flat log + scored retrieval: Generative Agents
  - bounded text buffer: Reflexion
  - vector store + decay: MemoryBank
  - tiered OS-style paging: MemGPT/Letta
  - linked note graph with mutation: A-Mem
  - vector fact store with CRUD: Mem0
  - entity graph with soft invalidation: Mem0g
  - bi-temporal KG + episodes + communities: Zep/Graphiti
  - OpenIE KG + PPR: HippoRAG
  - raw files + agentic search: Letta filesystem
  - latent KV cache: LongMem
  - frozen corpus: RETRO/kNN-LM
- The decisive design axis for correctness under change is **how contradiction is handled**: delete (lossy, can't answer "what did I used to…"), invalidate (keeps history), or bi-temporal (can answer "what was true when"). Of the reviewed systems, only Zep models event-time and ingest-time separately.
- Explicit *forgetting* (as opposed to supersession) is rare in production systems. MemoryBank's Ebbinghaus decay is the main principled example. Most systems rely on ranking decay or never forget.

### Gaps
- I did not re-fetch the Generative Agents, MemGPT, MemoryBank, Reflexion, CoALA, HippoRAG, LongMem, kNN-LM, RETRO, NTM or Memory Network papers this session. Their mechanisms and numbers are [M] and should be spot-checked (especially the 0.995 decay, the 150 reflection threshold, MemGPT's ~70% warning threshold, and HippoRAG's 10–30×/6–13× figures).
- The current (2026) Letta and Mem0 product architectures (e.g., Letta "sleep-time compute"/memory blocks, Mem0 platform changes after the paper) were not verified against repos this session.

---

## Q2. Reported accuracy, latency, and token cost on LoCoMo / LongMemEval, and the benchmark disputes

### Takeaway
On LoCoMo, Mem0's paper puts Mem0g at 68.44 J and Mem0 at 66.88, both **below its own full-context baseline (72.90)**. Mem0's advantage is latency and tokens (≈1.8k tokens/query vs 26k; p95 total 1.44 s vs 17.1 s). Zep self-reports 75.14 after a correction. Mem0 re-ran Zep at 58.44. Letta reports 74.0 with a plain filesystem plus gpt-4o-mini. Vendor-run LoCoMo numbers are not mutually comparable. LongMemEval is the harder and more credible benchmark: Zep reports 71.2% vs 60.2% full-context with gpt-4o, at ~90% lower latency.

### Cited Findings

**Benchmarks themselves**
- **LoCoMo** (Maharana et al., ACL 2024). Very long-term multi-session dialogues generated by LLM agents grounded in personas and event graphs. The paper describes ~300 turns and ~9K tokens on average over up to 35 sessions, and covers QA, event-summarization and multimodal tasks. QA categories are single-hop, multi-hop, temporal, open-domain, and adversarial (cat. 5). [M] — [arXiv 2402.17753](https://arxiv.org/abs/2402.17753)
  - Conflicting reports on conversation length: Mem0's paper says ~26k tokens per conversation [V]; Zep says the conversations average ~16k–26k tokens [V].
  - Mem0 paper: [arXiv 2504.19413](https://arxiv.org/html/2504.19413); Zep blog: [getzep.com](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- **LongMemEval** (Wu et al., ICLR 2025). 500 curated questions embedded in scalable user–assistant chat histories. It tests five abilities: information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, and **abstention**.
  - Commercial assistants and long-context LLMs show a ~30% accuracy drop over sustained interactions.
  - Proposed optimizations: session decomposition (value granularity), fact-augmented key expansion (indexing), and time-aware query expansion.
  - [V] — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
  - The history sizes (LongMemEval_S ≈115k tokens; _M ≈1.5M tokens / ~500 sessions) are [M]. The 115k figure is corroborated by Zep's table, which shows 115k avg context tokens for full-context [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- **MSC – Multi-Session Chat** (Xu, Szlam, Weston, "Beyond Goldfish Memory", ACL 2022). Crowdsourced chats of up to 5 sessions, with session summaries as memory. It is the origin of the DMR (Deep Memory Retrieval) evaluation used by MemGPT and Zep. [M] — [arXiv 2107.07567](https://arxiv.org/abs/2107.07567)
- **DMR is saturated.** Full-conversation gpt-4o-mini scores 98.0%, Zep 98.2%, MemGPT 93.4%, and Zep with gpt-4-turbo 94.8%. Zep's own paper concedes that DMR uses single-turn fact retrieval and that conversations fit in context. [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)

**LoCoMo results from the Mem0 paper (gpt-4o-mini, J = LLM-as-judge, cats 1–4) [V]** — [arXiv 2504.19413 HTML](https://arxiv.org/html/2504.19413)

| System | Overall J | Single-hop | Multi-hop | Open-dom | Temporal | Search p50/p95 (s) | Total p50/p95 (s) | Retrieved tokens/query |
|---|---|---|---|---|---|---|---|---|
| Full-context | 72.90 | – | – | – | – | – | 9.87 / 17.12 | ~26,031 |
| Mem0g | 68.44 | 65.71 | 47.19 | 75.71 | 58.13 | 0.476 / 0.657 | 1.09 / 2.59 | ~3,616 |
| Mem0 | 66.88 | 67.13 | 51.15 | 72.93 | 55.51 | 0.148 / 0.200 | 0.71 / 1.44 | ~1,764 |
| Zep (as run by Mem0) | 65.99 | 61.70 | 41.35 | 76.60 | 49.31 | 0.513 / 0.778 | 1.29 / 2.93 | ~3,911 |
| Best RAG (k=2, 256-tok chunks) | 60.97 | – | – | – | – | 0.255 / 0.699 | 0.80 / 1.91 | – |
| LangMem | 58.10 | 62.23 | 47.92 | 71.12 | 23.43 | 17.99 / 59.82 | 18.53 / 60.40 | ~127 |
| OpenAI (ChatGPT memory) | 52.90 | 63.79 | 42.92 | 62.29 | 21.71 | n/a | 0.47 / 0.89 | ~4,437 |
| A-Mem (re-run) | 48.38 | 39.79 | 18.85 | 54.05 | 49.91 | 0.668 / 1.485 | 1.41 / 4.37 | ~2,520 |

- Additional figures from the Mem0 paper:
  - Memory store footprint per conversation: Mem0 ~7k tokens, Mem0g ~14k, Zep >600k (raw conversation ≈26k). MemGPT/Letta was not reported in the tables. [V]
  - The abstract's headline claims are a "26% relative improvement" over OpenAI, "91% lower p95 latency" and ">90% token savings" vs full-context. [V] — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413)

**The Zep vs Mem0 dispute (Apr–May 2025) [V]**
- **Zep's critique** ("Lies, Damn Lies, & Statistics", May 6, 2025):
  - **Setup errors by Mem0:** Mem0 assigned the *user* role to both speakers, appended timestamps to message text instead of using `created_at`, and ran Zep's two searches *sequentially*, which inflated latency.
  - **LoCoMo flaws:** conversations are short enough to fit in context (full-context beat Mem0), there are no knowledge-update questions, cat. 5 lacks ground truth, BLIP image captions omit the needed info, and some questions misattribute speakers or are ambiguous.
  - **Corrected numbers:** Zep's corrected score is **75.14% ± 0.17**, and its corrected p95 search latency is 0.632 s vs Mem0g 0.657 s.
  - The post itself contains an internal inconsistency: the page description says 24%, but the body says ~10% over Mem0.
  - — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- **Mem0's rebuttal** (CTO Deshraj Yadav, GitHub issue on getzep/zep-papers):
  - Zep's original **84%** claim included adversarial cat. 5 in the numerator (≈+25.56 pts). Mem0's re-run gives **58.44% ± 0.20** (cats 1–4, 10 runs).
  - Zep altered the system prompt and template relative to the other baselines, and reported a single run vs Mem0's 10.
- **Zep's reply** (Daniel Chalef, May 12, 2025):
  - Accepted the calculation error, and corrected to 75.14 ± 0.17.
  - Said `created_at` has been in the SDK since May 2024, provided `asyncio.gather` concurrent code, and proposed LongMemEval instead.
  - The issue was closed for inactivity on May 19, 2025 **with no agreed resolution**.
  - — [getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5)
- **Letta's result** (Aug 12, 2025): a gpt-4o-mini agent with only file storage plus `search_files` scored **74.0%** on LoCoMo, beating Mem0g's reported ~68.5%. Letta says Mem0 did not respond to requests about how its numbers were computed. Letta argues that "memory is more about how agents manage context than the exact retrieval mechanism used." [V] — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- Net spread for "Zep on LoCoMo" across sources: 58.44 (Mem0 re-run) / 65.99 (Mem0 paper) / 75.14 (Zep corrected) / 84 (Zep original, retracted). [V] — sources above

**LongMemEval_S results from the Zep paper [V]** — [arXiv 2501.13956 HTML](https://arxiv.org/html/2501.13956)

| Setup | Model | Accuracy | Latency | Context tokens |
|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 31.3 s | 115k |
| Zep | gpt-4o-mini | 63.8% | 3.20 s | 1.6k |
| Full-context | gpt-4o | 60.2% | 28.9 s | 115k |
| Zep | gpt-4o | 71.2% | 2.58 s | 1.6k |

- Per-type deltas, Zep vs full context:
  - Gains: single-session-preference +77.7% (mini) / +184% (4o); temporal reasoning +48.2% / +38.4%; multi-session +16.7% / +30.7%.
  - Losses: **single-session-assistant fell** (−9.06% / −17.7%). Knowledge-update was mixed (−3.36% mini, +6.52% 4o).
  - The table has apparent arithmetic/copy errors. For example, the gpt-4o single-session-assistant drop of 94.6→80.4 is ≈15% relative, not 17.7%, and the single-session-user rows are duplicated.
  - The abstract's "up to 18.5%" and "90% latency reduction" headline figures derive from this table.
  - [V] — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)

**Cost-aware evaluation: MERIT (arXiv 2609.05441, Mishra & Mishra, Jul 2026) [V]** — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441)
- **Motivation:** LoCoMo and LongMemEval test QA over dialogue history, not whether remembered facts change a *tool-using agent's actions*.
- **Benchmark design:** MERIT uses leak-checked episodic tool-use tasks in 3 domains. It has a difficulty ladder up to updated-fact recall, controlled memory corruption, and full token and dollar metering.
- **Scale:** 23,440 scored episodes for a total of $42.57. Models: a gpt-4.1-mini pilot, then a preregistered 3 models × 3 seeds grid including GPT-4.1 and Claude Haiku 4.5, plus a Claude Sonnet 5 spot-check.
- **Results:**
  - Memory raised dependent-task success from 0.00 to 0.55–1.00.
  - On **updated facts**, embedding retrieval was unstable (0.30–0.95 across models, max seed gap 0.45). Update-on-write stores (structured fact store, LLM summarization) stayed at 0.70–1.00. **Hybrid (retrieval + fact store) did worse than the fact store alone.**
  - Agents **acted on a correctly retrieved value only 55% of the time**.
  - Changing the memory implementation shifted success by up to 60 points.
  - **Full replay was never cost-effective.** The best condition per domain delivered 2.7–3.9× its marginal utility per dollar.
  - No latency numbers are reported in the abstract.

### Inferences
- On LoCoMo, "memory systems" mostly buy **cost and latency**, not accuracy. Mem0's own paper shows full context winning by 4.5–6 J points. LoCoMo is too short (≈9k–26k tokens) to stress memory, so it is a weak SOTA arbiter.
- On LongMemEval_S (115k), compressed retrieval memory *beats* full context in both accuracy and latency. The losses on single-session-assistant ("what did *you* tell me") show that extraction pipelines biased toward user facts drop assistant-side content.
- Because the same system's LoCoMo score ranges over ~26 points depending on who runs it, prompt, role mapping and judge, only same-harness, multi-seed comparisons should be trusted. MERIT's finding that implementation choice swings task success by 60 pts is consistent with this.
- Mem0's 7k store vs Zep's >600k store per conversation shows a real tradeoff: Zep keeps episodes non-lossily plus a graph, while Mem0 compresses aggressively.

### Gaps
- No independent, neutral-party LoCoMo or LongMemEval leaderboard covering Mem0, Zep, Letta and A-Mem under one harness was found in this session.
- I did not verify Mem0's own LongMemEval numbers or any later Mem0 paper versions or blog claims (e.g., [mem0.ai/blog/ai-memory-benchmarks](https://mem0.ai/blog/ai-memory-benchmarks) exists but was not read).
- A-Mem's self-reported LoCoMo numbers (from its own paper) were not retrieved. Mem0's re-run (48.38) may differ from them.
- 2026 results for newer systems (e.g., third-party "Hebb Mind vs mem0" pages, MemOS, Memory-R1, etc.) were not verified.

---

## Q3. When does memory help versus hurt? (Stale facts, contradictions, injection attacks)

### Takeaway
Memory helps most when history exceeds the context window or is costly to replay, and for preference, temporal, and multi-session questions. It hurts or fails to help in four situations: (1) facts change and the store does not update on write; (2) extraction drops content (assistant-side utterances); (3) the agent retrieves the right value but doesn't act on it; (4) the store is poisoned. Persistent memory is a new attack surface: MINJA shows query-only injection.

### Cited Findings
- Memory raises dependent tool-use task success from 0.00 to 0.55–1.00. However, embedding-retrieval memory is **unstable on updated facts** (0.30–0.95). Update-on-write stores hold at 0.70–1.00. Adding retrieval on top of a fact store *hurt*. Agents used correctly retrieved values only 55% of the time. [V] — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441)
- Full-context replay was never cost-effective in MERIT. The best per-domain condition had 2.7–3.9× its marginal utility per dollar. [V] — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441)
- On LoCoMo, full context (72.90) beat every memory system in Mem0's own paper. Memory is a net accuracy cost when the history fits in context. [V] — [arXiv 2504.19413](https://arxiv.org/html/2504.19413)
- On LongMemEval_S, Zep beat full context overall (+11 pts with gpt-4o) but *lost* on single-session-assistant questions. Knowledge-update was mixed (−3.36% with gpt-4o-mini). [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- LongMemEval found ~30% accuracy drops for commercial assistants and long-context LLMs on sustained interactions. It explicitly tests knowledge updates and abstention (knowing you don't know). [V] — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
- Contradiction handling differs by system:
  - Mem0 base **deletes** contradicted memories.
  - Mem0g **marks edges invalid** to preserve history.
  - Zep sets t_invalid on the old edge via bi-temporal intervals.
  - [V] — [arXiv 2504.19413](https://arxiv.org/html/2504.19413); [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- Zep criticized LoCoMo for not testing knowledge updates (e.g., a user changing jobs). Benchmarks that ignore change will overrate stores that never update. [V] — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- **MINJA** (Dong et al., Mar 2025, rev. Feb 2026) is a memory injection attack in which the attacker has **no direct memory access**.
  - Through ordinary queries, the attacker gets the agent to store malicious records. These include "bridging steps" and use a progressively shortened indication prompt.
  - The records are later retrieved for a *different user's* query and induce harmful reasoning.
  - Reported as effective across diverse agents. The specific success-rate numbers were not in the abstract.
  - [V] — [arXiv 2503.03704](https://arxiv.org/abs/2503.03704)
- A-Mem's "memory evolution" lets new memories rewrite existing notes. This adds adaptivity, but it also gives one bad input a path to corrupt older memories. [V for mechanism] — [arXiv 2502.12110](https://arxiv.org/abs/2502.12110)

### Inferences
- **Stale facts:** pure append-plus-similarity retrieval returns both old and new values with no ordering signal. Write-time supersession (fact store) or validity intervals (bi-temporal) are necessary for any domain where facts change.
- **Retrieval ≠ use:** MERIT's 55% act-on-retrieved-value rate suggests that the bottleneck is increasingly the *policy* (prompting, provenance, recency labelling) rather than recall. This supports Letta's "context management over retrieval mechanism" thesis.
- **Self-modifying memories** (A-Mem evolution, Generative Agents reflections, Mem0 UPDATE/DELETE) amplify both consolidation benefits and error/poisoning propagation. For a single user, the main threat is *indirect* injection: content from web pages, emails, or documents processed by the agent getting written as "facts". Cross-user retrieval is less of a concern.

### Gaps
- MINJA's quantitative injection and attack success rates were not retrieved. Other memory-poisoning work (e.g., AgentPoison, NeurIPS 2024, arXiv 2407.12784) was not verified this session.
- I found no controlled study isolating harm from *stale summaries/reflections* (hallucinated abstractions) in Generative-Agents-style reflection.

---

## Q4. Which design patterns matter most for a personal single-user "second brain"?

### Takeaway
The evidence favours a **hybrid**: a non-lossy episodic log as ground truth, plus a small write-time-consolidated semantic fact/profile store with explicit supersession and time validity, plus agentic, multi-tool retrieval. Graph structure pays off mainly for temporal and multi-hop questions. Full replay is never cost-optimal once history is large.

### Cited Findings
- Keep raw episodes as a non-lossy source and derive facts and entities from them. Zep's episode subgraph does exactly this. [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- Update-on-write fact stores are robust on changed facts (0.70–1.00), while pure embedding retrieval is not (0.30–0.95). [V] — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441)
- Track time with two clocks (event time vs ingest time) and invalidate rather than delete. This enables "what did I believe in March?" and contradiction resolution. [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- Temporal-reasoning gains are the largest category win for graph/temporal memory:
  - Zep: +38–48% relative on LongMemEval temporal [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
  - Mem0g vs Mem0 temporal: 58.13 vs 55.51 [V] — [arXiv 2504.19413](https://arxiv.org/html/2504.19413)
  - Plain Mem0 had the best multi-hop score (51.15 vs Mem0g 47.19) [V]
- Hybrid lexical + semantic + graph retrieval with reranking: Zep combines cosine, BM25 and BFS with RRF/MMR/cross-encoder. [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- Agentic iterative search over plain files reached 74.0% on LoCoMo with gpt-4o-mini. Simple storage plus a good retrieval loop is a strong baseline. [V] — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- Small compact memory makes retrieval cheap: Mem0 needs ~1.8k tokens/query and has a 0.2 s p95 search, vs 26k tokens for full context. [V] — [arXiv 2504.19413](https://arxiv.org/html/2504.19413)
- LongMemEval's indexing tricks are directly reusable:
  - Session decomposition (store at round/fact granularity)
  - Fact-augmented keys (index chunks by extracted facts)
  - Time-aware query expansion (restrict the search window by the time mentioned in the query)
  - [V] — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
- An always-in-context core memory/profile block plus searchable recall and archival tiers (MemGPT/Letta) suits stable user facts that should be present on every turn. [M] — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- Importance scoring and reflection (Generative Agents) are a template for periodic "sleep" consolidation into higher-level insights. [M] — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- Use CoALA's split of memory into episodic (journal/events), semantic (facts/profile/knowledge), and procedural (skills, prompts, workflows) types as the schema. [M] — [arXiv 2309.02427](https://arxiv.org/abs/2309.02427); the "user-centric vs agent-centric" subject axis is from [arXiv 2602.06052](https://arxiv.org/abs/2602.06052) [V]

### Inferences (recommended pattern for a single-user second brain)
1. **Append-only episodic log** (notes, chats, documents), timestamped and immutable. This is the ground truth and the audit trail.
2. **Write-time extraction** into a small semantic store of atomic facts and entities. Use Mem0-style ADD/UPDATE/NOOP, but **invalidate rather than DELETE**, keeping valid_from/valid_to and source pointers back to the episode.
3. **Pinned profile/core block** (preferences, ongoing projects) always in context. Keep it small and editable by the user.
4. **Hybrid retrieval**: BM25 + embeddings + entity-graph neighbourhood, time-filtered, and reranked. Expose it to the agent as tools so it can iterate (Letta-style), not as a single top-k shot.
5. **Periodic offline consolidation** ("sleep"): reflections/summaries and community summaries, each linked to their evidence so they can be regenerated or discarded.
6. **Provenance and trust tags** on every memory (user-stated vs inferred vs ingested from external content). Ingested content should not be able to write semantic "facts" without review. This mitigates MINJA-style and indirect injection.
7. **Evaluate on your own data** with update and abstention probes (LongMemEval categories), not on LoCoMo alone. Meter tokens and dollars (MERIT-style).
- Single-user removes cross-tenant leakage but raises the importance of *long horizons* (years of data ≫ LongMemEval_S). It also raises the need for *user-visible, editable* memory, since the user is the only arbiter of truth.

### Gaps
- No primary study benchmarks memory systems on multi-year, single-user personal knowledge corpora (notes + email + docs) rather than synthetic dialogues.
- The value of explicit Ebbinghaus-style forgetting for a personal second brain has not been quantitatively compared to supersession-only approaches in the sources reviewed.

---

## Q5. Open problems as of late 2026

### Takeaway
Six problems remain open: evaluation (benchmarks too short, vendor-run, and QA-centric), update/contradiction correctness, the retrieval-to-action gap, learned (RL) memory policies, security/trustworthiness of persistent memory, and cost-aware design. The two 2025–2026 surveys converge on memory becoming a *trainable, self-evolving* component.

### Cited Findings
- **Benchmarks:**
  - LoCoMo fits in context, has no knowledge-update questions, and has label errors and an unusable category 5 [V] — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/).
  - DMR is saturated (≈98%) [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956).
  - QA benchmarks don't measure effects on agent actions [V] — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441).
- **Reproducibility:** the Zep–Mem0 dispute ended without resolution, with Zep's LoCoMo score spanning 58.44–75.14 depending on harness [V] — [GitHub zep-papers#5](https://github.com/getzep/zep-papers/issues/5). Letta could not reproduce or explain Mem0's numbers [V] — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory).
- **Updated facts and the retrieve-vs-act gap:** see the MERIT results above [V] — [arXiv 2609.05441](https://arxiv.org/abs/2609.05441).
- **Single-session-assistant and knowledge-update weaknesses** remain even in temporal KG memory [V] — [arXiv 2501.13956](https://arxiv.org/html/2501.13956).
- **Learned memory policies:** RL-trained context curation, decision-time experience consolidation, and portable/shareable skills are listed as directions. Memory management is described as "becoming a trainable capability." [V] — [arXiv 2602.06052](https://arxiv.org/abs/2602.06052)
- **Future directions** in Hu et al.: memory automation, RL integration, multimodal memory, multi-agent memory, and trustworthiness [V] — [arXiv 2512.13564](https://arxiv.org/abs/2512.13564)
- **Security:** query-only memory injection (MINJA) [V] — [arXiv 2503.03704](https://arxiv.org/abs/2503.03704)
- **Terminology fragmentation:** agent memory vs LLM memory vs RAG vs context engineering [V] — [arXiv 2512.13564](https://arxiv.org/abs/2512.13564)

### Inferences
- The field's centre of gravity has moved from *storage structure* (vector vs graph) to the *policy*: what to write, when to consolidate, what to trust, and whether to act on a memory. The evidence: the Letta filesystem result, MERIT's 60-point implementation swings and 55% act rate, and the surveys' emphasis on RL.
- Parametric/latent memory (LongMem-style KV memory, and test-time-training or continual-learning approaches mentioned in the surveys' "parametric" category) remains less mature for personal use than token-level external stores, because external stores are inspectable and editable.

### Gaps
- No verified quantitative results this session for 2026 RL-trained memory systems (e.g., Memory-R1, MemAgent, or similar). The surveys name these directions, but I did not extract numbers.
- No verified source on privacy/right-to-be-forgotten mechanics (provable deletion from derived summaries/graphs) for agent memory.
- MERIT is a single preprint by two authors (July 2026). Its findings are not yet independently replicated.
