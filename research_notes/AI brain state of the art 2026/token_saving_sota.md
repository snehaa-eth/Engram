# Token and Latency Reduction for Agent/Memory Systems: State of the Art (2025 – Nov 2026)

Scope note: about 18 tool calls. Primary sources were fetched where possible (Anthropic blog/docs, OpenAI compaction guide). Several numbers come only from search-result summaries of arXiv abstracts and are marked "(abstract-level)". Numbers that only appear in secondary or vendor sources are marked **[UNVERIFIED]**.

## Q1: Which 2025–2026 techniques produce the largest verified token savings for agents without accuracy loss?

### Takeaway
The biggest documented savings now come from managing the context window (clearing or masking stale tool results, compaction, deferred tool loading, sub-agent isolation, context folding), not from token-level prompt compression. Savings of 50–90% are reported, usually with equal or better accuracy, because a smaller context also reduces "context rot". Most of these figures are vendor-internal or measured on single benchmarks.

### Cited Findings

**Provider-side context management (Anthropic)**
- Anthropic launched context editing and the memory tool on Sep 29, 2025, with Sonnet 4.5. In a 100-turn web-search eval, context editing cut token consumption by **84%** and let agents finish workflows that otherwise failed when the context ran out. On an internal agentic-search eval, performance rose **29%** with context editing alone and **39%** with context editing plus the memory tool. Both were in public beta on the Claude Developer Platform, Bedrock and Vertex AI. These are vendor-internal evals that have not been independently reproduced. — [Claude blog: Managing context](https://claude.com/blog/context-management)
- Server-side compaction. Claude writes a summary on the server that replaces older turns. There are now two modes:
  - **Compaction on demand:** beta header `compact-2026-09-04` and a top-level `compaction` parameter. It can keep recent turns word for word, run in the background, and work with preserved thinking.
  - **Compaction at a token threshold:** the earlier `compact_20260112` edit in `context_management.edits`, with an `input_tokens` trigger and `pause_after_compaction`.
  - The docs give no token-savings figure and say that "response quality degrades as a conversation grows". — [Anthropic compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction); trigger example value of 100,000 tokens: [AWS Bedrock compaction docs](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-compaction.html). The 50k minimum trigger is **[UNVERIFIED]**; it appears only in a third-party Rust crate — [appam](https://docs.rs/crate/appam/0.2.0/source/src/llm/compaction.rs)
- Context editing also covers rule-based clearing of old tool results and old thinking blocks. — [Anthropic compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- Advanced Tool Use (Nov 2025):
  - **Tool Search Tool** (deferred tool definitions): Anthropic reports an **85% reduction in token usage**, with MCP-eval accuracy gains on Opus 4 from 49% to 74% and on Opus 4.5 from 79.5% to 88.1%. — [Anthropic engineering: advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use). The accuracy numbers are quoted via a [GitHub issue summary](https://claudeissues.com/issue/12836-support-tool-search-and-programmatic-tool-use-betas-for-reduced-token-consumptio). The illustration of about 77K tokens of tool definitions dropping to about 8.7K comes from a secondary source and is **[UNVERIFIED]** — [Medium](https://medium.com/@DebaA/anthropic-just-shipped-the-fix-for-tool-definition-bloat-77464c8dbec9)
  - **Programmatic Tool Calling:** intermediate tool results stay in a code sandbox and never enter the context. Reported savings are 43,588 → 27,297 tokens (**37%**) on complex research tasks. This comes from secondary summaries; another source says 24%, so the figure is **[UNVERIFIED]**. — [search summaries of Anthropic post](https://www.anthropic.com/engineering/advanced-tool-use)
- Anthropic's context engineering guidance (Sep 29, 2025):
  - **Compaction:** Claude Code keeps the summary plus the 5 most recently accessed files.
  - **Tool-result clearing:** described as "one of the lightest-touch forms of compaction".
  - **Structured note-taking:** e.g. a NOTES.md file kept outside the context window.
  - **Sub-agents:** each may spend tens of thousands of tokens exploring but returns a **1,000–2,000-token** condensed summary.
  - **Just-in-time retrieval:** the agent keeps lightweight references and loads content at runtime.
  - **Context rot:** treated as a performance gradient as the context grows, not a hard cliff.
  - Source: [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**Provider-side context management (OpenAI)**
- The Responses API offers two kinds of compaction:
  - **Server-side:** set `context_management` with `compact_threshold`. The server compacts inline and emits an encrypted compaction item that is "opaque and not intended to be human-interpretable".
  - **Standalone `POST /responses/compact`:** stateless and ZDR-friendly. Its output is the canonical next context window.
  - With `previous_response_id`, you send only the new message and should not prune by hand.
  - The guide gives **no token-savings numbers** and does not mention how compaction interacts with prompt caching.
  - Source: [OpenAI compaction guide](https://developers.openai.com/api/docs/guides/compaction.md)
- A Fetch Rewards experiment report says GPT-5.4 is "the first mainline model trained to support compaction". A third-party "47% token-cost reduction" claim could not be traced to OpenAI and is **[UNVERIFIED]**. — [Fetch Rewards report](https://rewards-assistant-docs.fetchrewards.com/_synced/consumer-agent/experiment-report-gpt54-compaction/); [search summary](https://developers.openai.com/api/docs/guides/compaction.md)

**Provider-side caching (Google)**
- On Vertex AI, implicit caching gives a 90% discount on cached tokens for Gemini 2.5 models, versus 75% for 2.0. It applies automatically on a cache hit, with no guarantee of a hit. — [Vertex AI context cache overview](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)
- The documented minimum cache size is 2,048 tokens for 2.5 models. — [Vertex AI context cache overview](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)
- Third parties report other minimums, both **[UNVERIFIED]**: 4,096 tokens for the Gemini 3.x series, and an enforced 1,024 for 2.5 Flash. — [APIYI](https://docs.apiyi.com/en/api-capabilities/gemini/prompt-caching); [MemX](https://memx.app/blog/gemini-caching-trap-prompts-under-1024-tokens/)

**Agent context-engineering research**
- **Context rot** (Chroma, Jul 14, 2025): across 18 LLMs (GPT-4.1, Claude 4, Gemini 2.5, Qwen3), performance became less reliable as input length grew, even on simple tasks.
  - Topically similar distractors hurt more than unrelated ones.
  - Models did better on *shuffled* haystacks than on logically structured ones.
  - Tests included LongMemEval conversational QA.
  - Source: [Chroma Context Rot report](https://research.trychroma.com/context-rot). Per-model numbers were not extracted.
- **The Complexity Trap** (JetBrains, NeurIPS 2025 DL4C workshop, arXiv 2508.21433): simply masking old observations (dropping old tool outputs) matched or beat LLM summarization on SWE-bench Verified across 5 model setups.
  - For the best model, masking was **52% cheaper** than the raw agent and raised solve rate by **2.6 points**.
  - Without any context management, cost per instance often more than doubled.
  - A hybrid of masking and summarization (v3) cut cost a further **7% vs masking alone** and **11% vs summarization alone**.
  - Source: [arXiv 2508.21433](https://arxiv.org/abs/2508.21433); [NeurIPS listing](https://neurips.cc/virtual/2025/131688)
- **ACON** (KAIST/Microsoft/Cambridge, arXiv 2510.00615, ICML 2026): compresses both observations and interaction history, using natural-language compression guidelines refined from failure analysis. No fine-tuning is needed, so it **works with closed models**.
  - Peak tokens fell **26–54%** on AppWorld, OfficeBench and multi-objective QA.
  - The PMLR abstract says task success *improved* over compression baselines; the arXiv v1 abstract says performance was "largely preserved".
  - A compressor distilled into a smaller model keeps more than 95% of accuracy.
  - Small LMs used as agents gained up to 46% because the context was less distracting.
  - (abstract-level) — [PMLR v306](https://proceedings.mlr.press/v306/kang26b.html); [arXiv](https://arxiv.org/abs/2510.00615v2); code at github.com/microsoft/acon
- **ACE (Agentic Context Engineering)** (arXiv 2510.04618, ICLR 2026): treats the context as a growing "playbook", built by a generator → reflector → curator loop with incremental delta updates. This avoids "brevity bias" and "context collapse".
  - Reported gains: **+10.6%** on agents and **+8.6%** on finance tasks, with lower adaptation latency and rollout cost.
  - It matches the top AppWorld production agent using a smaller open model.
  - ACE is mainly an *accuracy and self-improvement* method, not a token-reduction method: its contexts grow. It depends on prompt caching to stay cheap.
  - (abstract-level) — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618); [Microsoft Research](https://www.microsoft.com/en-us/research/publication/agentic-context-engineering-evolving-contexts-for-self-improving-language-models/)
- **Context-Folding / FoldGRPO** (arXiv 2510.11967): the agent branches into a subtask and folds it into a summary when done, trained with RL.
  - Pass@1 was **62.0%** on BrowseComp-Plus and **58.0%** on SWE-Bench Verified, gains of +20.0 and +8.8 points over ReAct baselines.
  - It used an **active context 10× smaller**.
  - Requires RL training, so it is not directly usable with closed APIs. The pattern (branch → fold) can be copied with sub-agents.
  - (abstract-level) — [emergentmind 2510.11967](https://www.emergentmind.com/papers/2510.11967)
- **AgentFold** (arXiv 2510.24699): proactive context management learned with SFT. AgentFold-30B-A3B scored 36.2% on BrowseComp and 47.3% on BrowseComp-ZH.
- **FoldAct** (arXiv 2512.22733): reports a 5.19× training speedup for folding agents.
- (abstract-level) — [AgentFold](https://huggingface-gw-kz.just-ai.com/papers/2510.24699); [FoldAct](https://arxiv.deeppaper.ai/papers/2512.22733v1)

**Structured formats (TOON)**
- **Official TOON benchmark** (209 retrieval questions, 4 models): about **40% fewer tokens** than JSON with similar or better accuracy (73.9% vs 69.7% in one snapshot; 76.4% vs 75.0% in a later one). — [TOON repo](https://github.com/telekom/toon) (snapshot via search)
- **Independent 201-question eval:** TOON ranked 9th of 12 formats at 47.5% accuracy, below JSON at 52.3%. For Claude Haiku 4.5, YAML scored 52.2%, TOON 50.7% and JSON 50.2%. — [DEV analysis](https://dev.to/ikaganacar/toon-benchmarks-a-critical-analysis-of-different-results-5h66)
- **arXiv 2603.03306** (Feb 2026, generation tasks): TOON gave a good accuracy-per-token ratio, but the "prompt tax" (format instructions) often cancels it, and some models degraded significantly. — [arXiv 2603.03306](https://arxiv.org/pdf/2603.03306)
- **arXiv 2605.29676** tests TOON in agent loops; its results were not retrieved. — [arXiv 2605.29676](https://arxiv.org/pdf/2605.29676)

### Inferences
- Ranked by verified size of saving for agent loops:
  1. Deferred tool definitions (about 85%, vendor)
  2. Tool-result clearing / context editing (84% vendor; 52% cost in independent JetBrains work)
  3. Context folding / sub-agent isolation (about 10× smaller active context)
  4. ACON-style learned compression guidelines (26–54% peak)
  5. Format changes such as TOON (about 40% on tabular payloads, accuracy mixed)
- A recurring finding is that smaller contexts often *raise* accuracy (JetBrains +2.6 points, Anthropic +29%, ACON's small models +46%). This fits the context-rot results, so "without accuracy loss" is often an understatement.
- Simple heuristics (observation masking) are a strong baseline. LLM summarization adds cost and latency, and is not clearly better unless it is combined with masking.

### Gaps
- No independent replication of Anthropic's 84%, 29% or 39% figures.
- OpenAI published no savings numbers for compaction.
- Per-model Chroma numbers were not extracted.
- Full-paper verification was not done for ACON, ACE or FoldGRPO.
- No measurement found of how compaction affects prompt-cache hit rates (compaction rewrites the prefix).

## Q2: Which of these work with black-box APIs (Claude, GPT, Gemini)?

### Takeaway
Most context-management wins are now built into the APIs or can be built on the client with no model access:
- Context editing / compaction / the memory tool (Anthropic)
- Compaction (OpenAI Responses)
- Implicit caching (Gemini)
- Tool search and programmatic tool calling (Anthropic)
- Observation masking, sub-agents, ACON guidelines and ACE playbooks (client side)

The RL-trained methods (FoldGRPO, AgentFold) and logprob or hidden-state retrieval gates do not carry over directly.

### Cited Findings
- Anthropic context editing, compaction (on-demand and threshold) and the memory tool are API parameters on Claude, also available on Bedrock and Vertex AI. — [Claude blog](https://claude.com/blog/context-management); [compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- OpenAI compaction is either inline (`compact_threshold`) or the stateless `/responses/compact` endpoint. The compacted state is encrypted, so you cannot inspect or edit it. — [OpenAI compaction guide](https://developers.openai.com/api/docs/guides/compaction.md)
- Gemini implicit caching is automatic, with a 90% discount on cached tokens for 2.5 models. — [Vertex AI caching](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)
- ACON needs no fine-tuning: it optimizes the compression guidelines in natural language, so it works with API models. — [PMLR](https://proceedings.mlr.press/v306/kang26b.html)
- Observation masking is a pure client-side transform of the history. — [arXiv 2508.21433](https://arxiv.org/abs/2508.21433)
- **Retrieval gating without logprobs:**
  - **"Beyond Self-Knowledge"** (arXiv 2607.25600): the model gives a provisional answer plus a verbalized confidence, and retrieval runs only below a validation-tuned threshold. Verbalized confidence is "poorly calibrated in absolute terms", and the extra probe call makes the method "retrieval-saving but not token-saving". — [arXiv 2607.25600](https://arxiv.org/pdf/2607.25600)
  - **arXiv 2604.05306** (Apr 2026): training models to state uncertainty explicitly gives the strongest adaptive-RAG controller. It requires training. — [arXiv 2604.05306](https://arxiv.org/html/2604.05306v1)
  - **Rowen:** a consistency-based uncertainty trigger, explicitly motivated by black-box models that do not expose logits. — [arXiv 2402.10612](https://arxiv.org/pdf/2402.10612v3)
  - **Self-Routing RAG** (Apr 2025): the model verbalizes its own knowledge to decide whether to retrieve. — [arXiv 2504.01018](https://arxiv.org/html/2504.01018v1)
  - **PCC** (Jan 2026): combines certainty with reasoning consistency. — [arXiv 2601.02574](https://arxiv.org/abs/2601.02574)
- **Routing:**
  - LLMRouterBench (Jan 2026; 400K+ instances, 21 datasets, 33 models) found that many routing methods perform similarly under unified evaluation, and that several recent and commercial routers "fail to reliably outperform a simple baseline". — [arXiv 2601.07206](https://arxiv.org/abs/2601.07206)
  - RouteLLM's ">2× cost reduction" is a 2024 claim, reported again in a 2026 guide that notes it was not independently verified. **[UNVERIFIED for 2026 models]** — [IntuitionLabs](https://intuitionlabs.ai/articles/ai-model-routing-cost-quality)
- **Reasoning tokens:** 2025 surveys ("Stop Overthinking", TMLR; "Efficient Reasoning Models", TMLR) catalogue overthinking and length-control methods. Most need training. The black-box levers are prompt-based methods and provider-exposed budgets (thinking budget / effort settings). — [arXiv 2503.16419](https://arxiv.org/abs/2503.16419); [arXiv 2504.10903 overview](https://awesomepapers.io/papers/2504.10903)

### Inferences
- With closed APIs, verbalized-confidence gating mainly saves *retrieval* calls and the tokens that retrieval would inject. It costs one extra short generation, so it pays off only when retrieval payloads are large.
- Logprob-based gates (as in TARG) do not work reliably on Claude, which exposes no logprobs, and work only partially on GPT and Gemini.
- Provider-side compaction makes the summarized state opaque (OpenAI encrypted) or model-written (Anthropic). An external memory store is still needed for anything that must persist across sessions or be inspected.
- Effort / thinking-budget parameters are the main black-box lever on reasoning tokens. No primary-source accuracy-vs-budget curves were retrieved in this pass.

### Gaps
- No search on cache-aware routing turned up any work. It remains unaddressed in what was found.
- No primary numbers were retrieved for Claude's effort parameter, OpenAI's reasoning_effort, or Gemini's thinkingBudget tradeoffs.
- No small-model probe or retrieval-gate paper was found that is tailored to closed APIs, beyond the ones listed.

## Q3: Is a design of LLMLingua-2 + TF-IDF router + prompt caching still current? What should replace or augment it?

### Takeaway
Prompt caching is still essential. LLMLingua-2 and a TF-IDF/SVM router are now secondary. Current practice is:
- Context-window management (tool-result clearing or masking, provider compaction, deferred tool loading, sub-agents with summaries of 1–2K tokens)
- A cache-stable prompt layout
- Retrieval gating via self-assessment where logprobs are not available

LLMLingua-2 conflicts with caching and with context rot only as far as it touches stable prefixes. It is best limited to large, one-off tool outputs or retrieved documents.

### Cited Findings
- The vendor-reported savings from context editing (84%) and tool search (85%) exceed the compression ratios typically cited for token-level compressors. They act on whole stale or unneeded blocks rather than on individual tokens. — [Claude blog](https://claude.com/blog/context-management); [Anthropic advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
- A simple observation-masking baseline matches summarization at about half the cost. Heuristic, structure-aware pruning is therefore competitive with learned or LLM compression for agent histories. — [arXiv 2508.21433](https://arxiv.org/abs/2508.21433)
- ACON shows that task-aware compression guidelines learned from agent failures beat generic compression baselines in agent settings. — [PMLR](https://proceedings.mlr.press/v306/kang26b.html)
- ACE warns that aggressive summarization causes "brevity bias" and "context collapse". It argues for incremental, itemized memory updates over repeated rewrites. — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618)
- Recent unified routing benchmarks show sophisticated routers often do not beat simple baselines. A cheap router (even TF-IDF/SVM) is not clearly outdated, but it should be checked against a simple baseline. — [LLMRouterBench](https://arxiv.org/abs/2601.07206)

### Inferences
- **Keep:**
  - **Prompt caching:** use a stable system prompt, tools and memory-index prefix; put volatile content last. Implicit caching (Gemini) and OpenAI automatic caching reward the same layout.
  - **A cheap router:** but benchmark it against a "always cheap model plus escalate on failure or low self-reported confidence" cascade.
- **Add (highest return first):**
  1. Tool-result clearing or observation masking: Anthropic `clear_tool_uses`, or a client-side mask of old observations.
  2. Provider compaction for long sessions (Anthropic on-demand compaction; OpenAI `compact_threshold`). Pair it with an explicit external memory (Anthropic memory tool, or NOTES.md-style files) so durable facts survive compaction.
  3. Deferred tool loading and tool search when there are many tools or MCP servers. Run tool calls programmatically so large intermediate results never enter the context.
  4. Sub-agent isolation that returns condensed summaries of 1–2K tokens.
  5. ACE-style itemized playbook memory with delta updates instead of whole-memory rewrites.
  6. Retrieval gating via self-consistency or verbalized confidence (Rowen, SR-RAG style) where logprobs are missing. Expect savings in retrieval calls, not necessarily in tokens.
  7. Compact formats (TOON, CSV, YAML) only for large uniform tabular payloads. Validate accuracy per model, because results are mixed.
- **Demote LLMLingua-2:** run it only on volatile, large retrieved or tool text placed after the cache breakpoint. Never run it on cached prefixes, since changing them destroys cache hits. Note that it adds a local model step and some latency.

### Gaps
- No head-to-head study was found that compares LLMLingua-2 with context editing or masking on the same agent benchmark.
- No primary data was found on the combined effect of compaction and prompt caching on cost. A compaction rewrite invalidates the cached prefix after the compaction point.
- Results of the 2026 TOON-in-agent-loops paper (arXiv 2605.29676) were not retrieved.
