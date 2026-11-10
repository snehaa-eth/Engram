# Adaptive Retrieval, Query Routing, and Caching for Cutting LLM Token Cost and Latency

Notes as of Nov 2026. Items marked **[unverified]** come from prior knowledge and were NOT re-checked against the primary source in this session. Treat them as likely but confirm before quoting. Everything else was read from the arXiv abstract/HTML page or the official provider docs this session.

---

## Q1. Measured token/cost savings and accuracy impact for each method

### Takeaway
Gating retrieval saves a lot (TARG: 70–90% fewer retrievals, accuracy equal or better). Routing between RAG and long context (Self-Route) cuts tokens 39–65%, about 0–3 points behind full long context. Routing between models (RouteLLM, FrugalGPT) cuts cost 1.1x–3.7x on realistic benchmarks; the "98%" FrugalGPT figure is a best case. Provider prompt caching gives a fixed, near-zero-risk 90–97.5% discount on repeated prefix tokens. KV-cache reuse systems (CacheBlend, RAGCache, SGLang) only help if you host the model yourself.

### Cited Findings

**A. When to retrieve (adaptive retrieval)**

- **TARG** (Wang, Wei, Ling; arXiv 2511.09803, v1 Nov 12 2025, v2 Apr 14 2026). Training-free. The model drafts a short answer without retrieval, and an uncertainty score from the draft's prefix logits decides whether to retrieve. Three scores: mean token entropy, top-1/top-2 logit margin, and variance across a few stochastic prefixes. The margin score is the recommended default for instruction-tuned LLMs; small-N variance is the conservative, budget-first option. Results: **70–90% fewer retrievals than Always-RAG while matching or improving EM/F1**, end-to-end latency close to Never-RAG, and draft overhead of "tens to hundreds" of tokens. Tested on NQ-Open, TriviaQA, PopQA, MuSiQue and ASQA. The abstract gives no absolute EM/F1 or latency numbers. — [arXiv 2511.09803](https://arxiv.org/abs/2511.09803)
  - Practical caveat: margin and entropy scores need token logprobs. Anthropic's API does not expose them; OpenAI exposes top logprobs for some models. On Claude you would have to substitute self-consistency (variance) or a verbalized-confidence proxy. (Inference; check current API docs.)
- **Adaptive-RAG** (Jeong et al., NAACL 2024 **[venue unverified]**; arXiv 2403.14403). A small trained classifier (T5-based **[size unverified]**) labels each query No-retrieval (A), Single-step (B) or Multi-step (C). Labels come automatically from which strategy answered correctly, plus dataset inductive bias. Averages across 6 open-domain QA datasets (Table 1); time is relative to single-step = 1.00 — [arXiv HTML 2403.14403v2](https://arxiv.org/html/2403.14403v2):
  - FLAN-T5-XL: No-retrieval F1 21.12 (time 0.11); Single-step F1 44.31 (1.00); Multi-step IRCoT F1 48.85 (time **8.81**, 4.69 steps); **Adaptive-RAG F1 46.94 (time 3.60, 2.17 steps)**; Oracle classifier F1 56.28 (time 2.11).
  - GPT-3.5: No-retrieval F1 48.56 (0.71); Single-step 46.99 (1.00); IRCoT 50.87 (3.33, 2.81 steps); **Adaptive-RAG 50.91 (1.46, 1.03 steps)**; Oracle 62.80 (1.03).
  - So with GPT-3.5, Adaptive-RAG matched multi-step F1 at **about 44% of its time**. With FLAN-T5-XL it reached 96% of multi-step F1 at 41% of the time.
  - The classifier is weak: **54.52% overall accuracy** (No 30.52%, One 66.28%, Multi 65.45%). The large oracle gap (+6–12 F1 at lower cost) shows routing quality is the bottleneck.
  - Predicted label mix: 8.6% No (0.35 s/query), 53.3% One (3.08 s), 38.1% Multi (27.18 s). The paper does not say which LLM these numbers come from.
  - Note: with GPT-3.5, no-retrieval (F1 48.56) beat single-step retrieval (46.99). A strong model plus a noisy retriever can make retrieval worse than nothing.
- **Self-RAG** (Asai et al., ICLR 2024 **[venue unverified]**). Fine-tunes Llama-2 7B/13B to emit reflection tokens (Retrieve, ISREL, ISSUP, ISUSE) that decide retrieval and critique passages **[mechanism details unverified]**. In the Adaptive-RAG comparison Self-RAG is weak and not directly comparable because it uses a different base LLM: F1 20.79 (FLAN-T5 setting) / 22.98 (GPT-3.5 setting), with 0.72–0.74 steps — [arXiv HTML 2403.14403v2](https://arxiv.org/html/2403.14403v2). It needs a fine-tuned model, so it cannot be used with closed API models.
- **FLARE** (Jiang et al., EMNLP 2023 **[unverified]**). Active retrieval during long-form generation: generate a provisional next sentence, and if any token's probability falls below a threshold, use that sentence as the query, retrieve, and regenerate **[unverified]**. Like TARG it needs token probabilities. Its cost is multiple retrieval and regeneration rounds per answer. I did not fetch quantitative results this session.
- **SKR** (Self-Knowledge guided Retrieval; arXiv 2310.05002, Oct 2023; Findings of EMNLP 2023 **[venue unverified]**). Elicits the model's self-knowledge, partly by referring to similar questions it has seen before, and retrieves only for questions it does not know. The abstract says it beats CoT and always-retrieve with InstructGPT/ChatGPT but gives no numbers. — [arXiv 2310.05002](https://arxiv.org/abs/2310.05002)
- **CRAG, Corrective RAG** (Yan et al. 2024; arXiv 2401.15884). A lightweight retrieval evaluator scores the retrieved documents and triggers different actions; web search supplements a static corpus when retrieval is poor. Tested on 4 short- and long-form datasets. The abstract gives no numbers. — [arXiv 2401.15884](https://arxiv.org/abs/2401.15884). **[unverified details]**: the evaluator is a fine-tuned T5-large (~0.77B), the actions are Correct / Incorrect / Ambiguous, a decompose-then-recompose step filters strips of the documents, and the datasets are PopQA, Biography, PubHealth and ARC-Challenge. Cost note: CRAG *adds* cost (an evaluator pass on every query, plus web search on some). It is a quality fix, not a cost saver.
- **RAGRouter-Bench baseline** (Bansal & Agarwal; arXiv 2604.03455, Apr 3 2026). 7,727 queries across 4 domains, each labeled factual / reasoning / summarization, so the router picks a retrieval paradigm per query. 15 classifier × feature combinations tested. **TF-IDF + SVM was best: macro-F1 0.928, accuracy 93.2%, simulated token savings 28.1%** compared with always using the most expensive paradigm. TF-IDF beat MiniLM embeddings by 3.1 macro-F1. Medical queries were hardest to route and legal easiest. The authors argue corpus-aware routing is needed next. — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455). Caveat: the savings are simulated, and the labels reflect query type rather than measured end-to-end answer correctness.
- **Budget-aware Active RAG evaluation** (Qian et al.; arXiv 2607.24010, Jul 27 2026; KDD 2026 Workshop on Evaluation and Trustworthiness of Agentic AI). Defines retrieval as useful only when it *changes correctness* relative to the no-retrieval answer. Findings, quoted from the abstract: retrieval harm is "non-negligible"; router rankings flip across datasets and budgets; thresholds calibrated on past data miss their target usage rate on new data; and **"simple uncertainty or retrieval-score baselines often perform about as well as learned utility routers."** Recommends reporting utility frontiers, realized usage, threshold-transfer error, harm rates and cost decompositions. The abstract page gives no numbers. — [arXiv 2607.24010](https://arxiv.org/abs/2607.24010)

**B. Model routing and cascades**

- **FrugalGPT** (Chen, Zaharia, Zou; arXiv 2305.05176). Three strategies: prompt adaptation, LLM approximation (including caching), and an LLM cascade with a learned scorer that decides when to stop. Claims: API prices differ by about two orders of magnitude; FrugalGPT matches GPT-4 with **"up to 98%" cost reduction**, or gains 4% accuracy at equal cost. — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176). "Up to" means a best case on specific 2023 datasets. Do not treat it as typical.
- **RouteLLM** (Ong et al., LMSYS; arXiv 2406.18665; ICLR 2025 **[venue unverified]**). Binary strong/weak routers (matrix factorization, BERT classifier, causal-LLM classifier, similarity-weighted ranking) trained on Chatbot Arena preferences with augmentation. Strong model GPT-4 and weak model Mixtral-8x7B. Table 6 — [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665):
  - MT Bench: 95% of GPT-4 quality with **13.4% GPT-4 calls (3.66x cheaper)** using MF with Arena data plus LLM-judge augmentation.
  - MMLU: 92% quality with 35.4% GPT-4 calls (**1.41x**).
  - GSM8K: 87% quality with 33.6% GPT-4 calls (**1.49x**).
  - Router overhead per million requests: MF $3.32, BERT $3.19, causal LLM $5.23, SW ranking $39.26 (≤0.4% of GPT-4 generation cost). It matches Martian and Unify AI with up to 40% fewer GPT-4 calls. Routers still worked when the strong and weak models were swapped at test time. — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
  - Lesson: savings are large on chat-style queries (MT Bench) and small on knowledge and math benchmarks (1.1–1.5x), where weak-model failures are harder to predict.

**C. Caching**

*Provider prompt caching (prefix caching, exact match)*
- **Anthropic** (official docs, read Nov 2026) — [Claude prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
  - Pricing multipliers on base input: **5-minute write 1.25x, 1-hour write 2x, read 0.1x** (0.05x for Opus 5.5 / Sonnet 5.5; 0.025x for Fable 5.1 / Mythos 5.1; 0.1x for Haiku 5.5 and others). The multipliers stack with the Batch API discount and long-context pricing.
  - TTL: 5 minutes by default, refreshed free on each hit; `"ttl": "1h"` is optional. The TTL starts at the beginning of the request, so generation time counts against it. Longer-TTL blocks must come before shorter ones.
  - Minimum cacheable length: 512 tokens on current models (Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable/Mythos 5.x); 1,024–4,096 on older ones. Prompts below the minimum are processed uncached **with no error**. Check that `cache_creation_input_tokens` and `cache_read_input_tokens` are not both 0.
  - Up to 4 explicit `cache_control` breakpoints, with a 20-block lookback per breakpoint. A top-level `cache_control` turns on **automatic caching**, which moves the breakpoint forward as the conversation grows and uses 1 of the 4 slots.
  - Invalidation order is tools → system → messages: a change at one level invalidates that level and every later one. Changing tool definitions invalidates everything. Toggling web search or citations, or switching speed mode, invalidates system and messages. Changing tool_choice, images, the thinking config or effort invalidates messages. Inserting a mid-conversation system message (on supported models) keeps the cached prefix.
  - Output tokens are unaffected. The docs say TTFT improves for long prompts but give no percentage, and the 5-minute and 1-hour caches have the same latency. Cache pre-warming with `max_tokens: 0` is supported, except with streaming, extended thinking, structured outputs, forced tool_choice or batches. The docs also say cache hits don't count against rate limits **[stated on the page; check scope]**.
- **OpenAI** (official docs, read Nov 2026) — [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching):
  - Caching is on by default (implicit). GPT-5.6+ adds opt-in explicit breakpoints (`prompt_cache_options.mode: "explicit"`).
  - Minimum is 1,024 visible input tokens on GPT-5.6+. On earlier models, `cached_tokens` is rounded down to multiples of 128, and GPT-5.5 places implicit breakpoints every 2,048 tokens.
  - Read price is **0.1x input** (0.05x for GPT-6.1 Sol and Terra); the docs say "up to 95%" off. Write cost on GPT-5.6+: **30-minute TTL (default) 1.25x, 12-hour TTL 2.5x**. Older models have no write surcharge.
  - Retention on older models: `in_memory` (about 5–10 minutes idle, up to 1 hour) or `24h`. Non-ZDR organizations default to 24h.
  - `prompt_cache_key` improves routing to the same cache (the docs suggest about 15 requests per minute per key) but does not guarantee a hit. Example deployments report token cache-hit rates of about 70% and over 90%. The docs give no latency percentage.
  - Note: these model names and terms (GPT-5.6, 12h TTL, write surcharge) are new in 2026 and differ from the 2024 scheme (automatic caching, 50% discount, no write fee). Older blog posts are out of date.

*Semantic caching*
- **GPTCache** (Zilliz, open source). Embeds each query, looks up a vector store, and returns a stored answer when similarity exceeds a threshold. The README claims large cost and speed gains **[figures unverified; vendor claim, not a measured study]**. Core risk: false-positive hits, where similar wording with different intent (e.g., "my meeting tomorrow" vs. "my meeting today") returns a stale or wrong answer. For a personal assistant whose answers depend on time and personal state, this risk is high. (Inference.)
- **Agentic Plan Caching** (Zhang, Wornow, Wan, Olukotun; arXiv 2506.14852; NeurIPS 2025). Extracts plan templates from completed agent runs, matches new requests to them by keyword extraction, and has a lightweight model adapt the template. Results: **50.31% average cost reduction and 27.28% average latency reduction** "while maintaining performance" (no accuracy delta on the abstract page). The authors contrast it with chatbot semantic caching, which does not suit agents. — [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)

*KV-cache reuse (self-hosted serving only)*
- **vLLM PagedAttention** (Kwon et al., SOSP 2023): paged KV memory gives 2–4x throughput over FasterTransformer/Orca, and it supports automatic prefix caching **[numbers unverified this session]**.
- **SGLang RadixAttention** (Zheng et al.; arXiv 2312.07104; NeurIPS 2024 **[venue unverified]**). Keeps a radix tree of KV caches so requests sharing a prefix reuse it. Reports **up to 6.4x throughput** over other inference systems. — [arXiv 2312.07104](https://arxiv.org/abs/2312.07104)
- **CacheBlend** (Yao et al.; arXiv 2405.16444; EuroSys 2025 **[venue unverified]**). Reuses precomputed KV caches for RAG chunks even when they are *not* the prefix, by selectively recomputing a small subset of tokens to restore cross-attention (**~15% of tokens [unverified]**). Reports **2.2–3.3x lower TTFT and 2.8–5x higher throughput** than full recompute "without compromising generation quality." — [arXiv 2405.16444](https://arxiv.org/abs/2405.16444)
- **RAGCache** (Jin et al.; arXiv 2404.12457). Caches the KV states of retrieved documents in a knowledge tree across GPU and host memory. Reports **up to 4x lower TTFT and up to 2.1x higher throughput** than vLLM + Faiss. — [arXiv 2404.12457](https://arxiv.org/abs/2404.12457)

**D. Long context vs. RAG**
- **Self-Route** (Li et al. 2024, "Retrieval Augmented Generation or Long-Context LLMs?"; arXiv 2407.16833; EMNLP 2024 Industry **[venue unverified]**). With enough resources, long context (LC) beats RAG on average. Self-Route sends each query to RAG first; the model either answers or replies "unanswerable", and unanswerable queries escalate to full LC. Averages over 9 LongBench/∞Bench datasets with the Contriever retriever (Table 1) — [arXiv HTML 2407.16833v2](https://arxiv.org/html/2407.16833v2):
  - Gemini-1.5-Pro: LC 49.70, RAG 37.33, **Self-Route 46.41**; uses **38.39% of LC tokens**; 76.78% of queries handled by RAG.
  - GPT-4o: LC 48.67, RAG 32.60, **Self-Route 48.89**; uses **61.40% of LC tokens**; 57.36% handled by RAG.
  - GPT-3.5-Turbo: LC 32.07, RAG 30.33, **Self-Route 35.32**; uses 38.85% of LC tokens; 74.10% handled by RAG.
  - The paper says cost falls by about 65% (Gemini) and 39% (GPT-4o). RAG and LC give identical predictions on 63% of queries.
  - RAG failure modes: multi-step reasoning (dominant in HotpotQA, 2WikiMQA, MuSiQue), general or summarization queries (QMSum), long or complex queries, and implicit queries (NarrativeQA).
  - The paper is internally inconsistent: Section 4.3 gives slightly different numbers than Table 1 (e.g., Gemini gap −2.2 vs. −3.29; RAG-answered 81.74% vs. 76.78%). Quote Table 1.
- **Lost in the Middle** (Liu et al., TACL 2024 **[unverified]**). Multi-document QA and key-value retrieval show a U-shaped accuracy curve: best when the relevant document is at the start or end of the context, worst in the middle. For GPT-3.5-Turbo with about 20 documents, mid-context accuracy dropped below its closed-book accuracy **[specific numbers unverified]**.
- **Context Rot** (Chroma technical report; Hong, Troynikov, Huber; Jul 14 2025). 18 models tested (Claude Opus 4 / Sonnet 4 / 3.7 / 3.5 / Haiku 3.5; o3, GPT-4.1 family, GPT-4o, etc.; Gemini 2.5 Pro / Flash; Qwen3). Findings — [Chroma Context Rot](https://www.trychroma.com/research/context-rot):
  - Performance degrades as input grows, even on simple tasks.
  - Needle–question pairs with lower similarity degrade faster.
  - One distractor hurts, and four hurt more.
  - Shuffled haystacks *beat* coherent ones across all 18 models.
  - On LongMemEval, focused prompts (~300 tokens) consistently beat full prompts (~113k tokens). Claude showed the largest gap, mostly from abstaining.
  - The report gives no single headline percentage; results are in plots.

### Inferences
- Rough ranking by measured saving and risk: prompt caching (90–97.5% off cached tokens, no accuracy risk) > retrieval gating (70–90% fewer retrievals, accuracy equal or better) > RAG-first/LC-fallback (35–60% fewer tokens, 0–3 points behind LC) > model routing (1.1–3.7x, a few points of quality) > semantic caching (high savings on repeated queries, correctness risk).
- Adaptive-RAG's oracle rows and the 2607.24010 finding point the same way: the router's *accuracy* matters more than its sophistication. Cheap signals (uncertainty, retrieval score, TF-IDF) compete with learned routers.
- For a strong frontier model, "retrieve always" can be net harmful on general-knowledge questions (Adaptive-RAG with GPT-3.5: no-retrieval F1 48.56 vs. single-step 46.99). For a personal assistant, though, most valuable questions concern private data the model cannot know, so the no-retrieval branch applies mainly to general-knowledge and chit-chat turns.

### Gaps
- Not fetched this session: full numeric results for FLARE, Self-RAG (own paper), CRAG, SKR, Lost in the Middle, vLLM, and GPTCache. They are marked [unverified] above.
- TARG and 2607.24010 abstracts give no absolute accuracy, token or latency numbers; the full PDFs are needed.
- No primary source quantifies semantic-cache false-positive rates for personal-assistant workloads.
- Provider docs give no TTFT reduction percentage. Older Anthropic announcements cited up to ~85% latency reduction for long prompts **[unverified, not on the current docs page]**.

---

## Q2. Which techniques give the best savings per engineering hour for a personal assistant built on API models?

### Takeaway
1. Restructure prompts for provider prefix caching and turn on automatic caching (hours of work, 90%+ off the stable prefix).
2. Add a cheap retrieve / don't-retrieve gate plus RAG-first with escalation (a day or two).
3. Add model routing with a simple cascade (days, plus an eval set).

Semantic caching and KV-level systems (CacheBlend, RAGCache, SGLang) are poor value for an API-only assistant: semantic caching risks correctness, and the KV systems require self-hosting.

### Cited Findings
- Anthropic caching is a single top-level `cache_control` field (automatic caching). Reads cost 0.1x (0.05x on Opus/Sonnet 5.5), and 5-minute writes cost 1.25x, so a cached prefix pays for itself after one reuse inside the TTL. — [Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- OpenAI caching is automatic by default; 30-minute writes cost 1.25x and reads 0.1x on GPT-5.6+. — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching)
- Cache-busting pitfalls that matter for assistants: editing tool definitions invalidates the whole cache; toggling web search or citations invalidates system and messages. — [Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- A TF-IDF + SVM query router reaches 93.2% accuracy and 28.1% simulated token savings, which is cheap to build. — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)
- Simple uncertainty baselines are about as good as learned routers. — [arXiv 2607.24010](https://arxiv.org/abs/2607.24010)
- Self-Route needs only a prompt ("answer or say unanswerable") plus a fallback, and saves 39–65% of tokens relative to LC. — [arXiv HTML 2407.16833v2](https://arxiv.org/html/2407.16833v2)
- Agentic plan caching gives −50% cost and −27% latency for repeated agent workflows, but it needs a plan-template store and an adapter model. — [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)
- RouteLLM routers cost $3–5 per million requests to run (MF/BERT), but their savings on knowledge and math tasks are only 1.4–1.5x. — [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665)

### Inferences
Ranked by savings per engineering hour for a personal assistant on Claude or OpenAI APIs:
1. **Prompt layout for caching (highest ROI).** Put stable content first in the order tools → system persona → user profile/memory summary → long-lived documents, and volatile content last (current time, retrieved snippets, user turn). Use automatic caching or a breakpoint after the stable block. Use the 1-hour TTL if the user's sessions have gaps over 5 minutes, since the 2x write pays back after about 2 reads. Never put timestamps or per-request IDs in the system prompt. Keep tool definitions frozen. Monitor `cache_read_input_tokens`.
2. **Retrieval gate.** Use rules plus a cheap classifier (TF-IDF or a Haiku-class LLM call): skip retrieval for chit-chat and general knowledge; always retrieve for personal-data questions ("my", names, dates). Logprob gates like TARG are hard on Claude because the API exposes no logprobs.
3. **RAG-first, escalate on "unanswerable"** (Self-Route pattern) instead of stuffing the whole memory store.
4. **Model cascade.** Use a small model for classification, extraction and memory writes, and a frontier model for synthesis. Hand-assigning models by task type beats a learned router at personal scale.
5. **Plan or result caching for recurring agent tasks** (morning brief, weekly summaries). Prefer exact or keyed caches with invalidation on data change over fuzzy semantic matching.
6. Skip GPTCache-style semantic caching of final answers for personalized, time-sensitive questions. Skip self-hosted KV systems unless you are running an open model locally.

### Gaps
- No published study measures these techniques together on a single-user personal assistant workload. The ROI ranking is a judgment call.
- Real hit rates for a single user with sparse traffic, where the 5-minute TTL often expires between turns, are unmeasured. The 1-hour TTL and pre-warming tradeoff needs empirical testing.

---

## Q3. How should a system decide: answer directly vs. retrieve once vs. multi-hop vs. stuff long context?

### Takeaway
Use a cascade ordered by cost: (0) cache lookup → (1) classify query type and private-data need → (2) answer directly if the model is confident and no private data is involved → (3) single retrieval, with the model allowed to say "unanswerable" → (4) multi-hop or iterative retrieval for compositional questions → (5) long-context stuffing of the relevant corpus slice only as a fallback, or when the corpus is small and stable enough to be cached. Calibrate thresholds to a usage budget and re-check them on fresh data, because thresholds drift.

### Cited Findings
- Three-way complexity routing (none / single / multi) beats both fixed strategies on the efficiency–accuracy frontier. With GPT-3.5 it matched multi-step F1 (50.91 vs. 50.87) at 1.46x versus 3.33x single-step time. — [arXiv HTML 2403.14403v2](https://arxiv.org/html/2403.14403v2)
- Multi-step questions are expensive: 27.18 s/query vs. 3.08 s for single-step in Adaptive-RAG. Route to multi-hop only when needed. — [same](https://arxiv.org/html/2403.14403v2)
- RAG-then-LC fallback handles 57–77% of queries via RAG at 38–61% of LC tokens. RAG fails mostly on multi-step, general/summarization and implicit queries, which are exactly the cases to send to LC or multi-hop. — [arXiv HTML 2407.16833v2](https://arxiv.org/html/2407.16833v2)
- Query-type labels (factual → light retrieval; reasoning or summarization → heavier paradigm) can be predicted lexically with about 93% accuracy. — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)
- Draft-then-gate with an uncertainty margin cuts retrievals 70–90% without accuracy loss. — [arXiv 2511.09803](https://arxiv.org/abs/2511.09803)
- Retrieval can hurt correctness, and thresholds don't transfer, so measure realized usage and harm rate. — [arXiv 2607.24010](https://arxiv.org/abs/2607.24010)
- Long context is not free accuracy: focused ~300-token prompts beat ~113k-token prompts on LongMemEval across 18 models, and distractors hurt. — [Chroma Context Rot](https://www.trychroma.com/research/context-rot)
- When a weak retriever is wrong, CRAG-style evaluation can trigger corrective actions such as web search, at extra cost. — [arXiv 2401.15884](https://arxiv.org/abs/2401.15884)

### Inferences
A concrete decision policy for a personal assistant:
- **Direct answer:** no private entity or time reference, a general-knowledge or chit-chat intent, and high self-consistency on 2–3 cheap samples (or high verbalized confidence).
- **Retrieve once:** a private-data lookup ("when is my dentist appointment") or a single factual hop. Return top-k chunks placed at the *end* of the prompt, after the cached prefix (this also suits Lost-in-the-Middle), and allow "unanswerable" to trigger escalation.
- **Multi-hop / agentic:** compositional ("compare what X said last month with my notes on Y"), temporal aggregation, or the single retrieval returned "unanswerable". Cap the number of hops.
- **Long-context stuffing:** summarization over a bounded set (one long document or one email thread), or when RAG reports unanswerable and the candidate set fits. Cache the stuffed document with a 1-hour TTL when follow-up questions are likely. Avoid stuffing the whole memory store; context rot plus distractors makes it worse than focused retrieval.
- Set the gate threshold against a token budget and re-check it monthly on logged queries, measuring realized retrieval rate, harm rate (cases where retrieval flipped a right answer to wrong) and cost per answered query.

### Gaps
- No paper tests the full 4-way choice (direct / single / multi / LC) jointly with prompt caching economics. With 90% cache discounts, LC stuffing of a *stable* corpus may cost less than the numbers in Self-Route (written before 2026 caching prices) suggest. This is untested.
- Uncertainty gates that need logprobs are hard to apply on Claude, and verbalized confidence has not been validated as a substitute in these papers.

---

## Q4. Open problems as of 2026

### Takeaway
The main open issues: routers that generalize across domains and budgets, calibration that survives distribution shift, measuring when retrieval *hurts*, logprob-free gating for closed APIs, safe semantic and plan caching (correctness and staleness), and joint optimization of retrieval and routing with provider cache pricing.

### Cited Findings
- Router rankings change across datasets and budgets; nominal thresholds miss target usage; retrieval harm is non-negligible; learned utility routers barely beat simple baselines. — [arXiv 2607.24010](https://arxiv.org/abs/2607.24010)
- Query-only routing leaves a gap; corpus-aware routing is needed. Medical queries are hardest. — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)
- Adaptive-RAG's classifier is only 54.5% accurate, while an oracle router would gain 6–12 F1 *and* cut cost, so routing accuracy is the bottleneck. — [arXiv HTML 2403.14403v2](https://arxiv.org/html/2403.14403v2)
- RouteLLM gains shrink on MMLU and GSM8K (1.1–1.5x), so predicting weak-model failure on knowledge and reasoning tasks remains hard. — [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665)
- Long-context reliability is non-uniform: coherence and distractor effects, and abstention behavior differ by model family. — [Chroma Context Rot](https://www.trychroma.com/research/context-rot)
- Semantic caching designed for chatbots does not transfer to agents; plan-level caching is a new direction. — [arXiv 2506.14852](https://arxiv.org/abs/2506.14852)
- Non-prefix KV reuse (CacheBlend) shows that position-independent caching is possible, but providers expose only prefix caching. — [arXiv 2405.16444](https://arxiv.org/abs/2405.16444); [Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Provider caching rules keep changing: new TTL tiers (OpenAI 30m/12h with write fees on GPT-5.6+), model-specific read discounts (0.025x–0.1x on Anthropic), and model-specific minimums. Cost models go stale fast. — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching); [Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### Inferences
- Open: gating signals for closed APIs without logprobs (verbalized confidence, self-consistency, retrieval-score-only gates) need benchmarking against TARG-style logit gates.
- Open: cache-aware routing. Choosing between RAG and LC should account for whether the LC prefix is already warm in the provider cache; current router papers assume uncached per-token prices.
- Open: memory-store-specific routing for personal assistants (temporal queries, entity resolution) is not represented in RAGRouter-Bench's 4 domains or in the open-domain QA datasets used by TARG and Adaptive-RAG.
- Open: invalidating semantic and plan caches when the user's underlying data changes.

### Gaps
- I did not find a 2026 survey consolidating these results. Systematic evaluation of semantic-cache false positives was not found in this session.
- Several key numbers (FLARE, Self-RAG, CRAG, Lost in the Middle, vLLM) need confirmation from the full papers.
