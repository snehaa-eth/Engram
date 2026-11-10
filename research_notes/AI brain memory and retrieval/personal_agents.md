# Personal AI Agents: Architecture, Personalization, Evaluation, Security, and Deployed Systems (as of Nov 2026)

Verification legend: unmarked items were confirmed this session from the cited page or a search summary of it. **[PK]** means the claim and URL come from the researcher's prior knowledge of well-known primary papers or posts and were NOT re-fetched this session. **[Unverified]** means the claim comes only from a single secondary or aggregator source, or sources conflict.

---

## 1. What is the reference architecture of a personal agent in 2026, and what are the main components?

### Takeaway
The clearest academic reference model is the four-capability framework in Xu et al.'s survey (arXiv 2602.22680, Feb/Mar 2026). It covers **profile modeling, memory, planning, and action execution**, and argues that personalization runs through the whole decision pipeline, not just the wording of replies. Deployed products add three layers the survey treats only lightly: a tool/connector layer (MCP, app intents, a cloud "computer"), a trust/permission layer, and a context-assembly layer that decides what personal data goes into each prompt.

### Cited Findings
**Survey framework (primary)**
- "Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions." Authors: Yue Xu, Qian Chen, Zizhan Ma, Dongrui Liu, Wenxuan Wang, Xiting Wang, Li Xiong, Wenjie Wang. v1 26 Feb 2026, v2 16 Mar 2026. It defines "personalized LLM-powered agents" (PLAs) and organizes the field around four interdependent capabilities: profile modeling, memory, planning, and action execution. It also stresses cross-component interactions and says personalization "affects the whole decision pipeline." — [arXiv abs](https://arxiv.org/abs/2602.22680)
- **Profile modeling (Sec. 3)** has two paradigms:
  - *Persona-based* profiles hold holistic traits or preference descriptions as attribute vectors or natural-language summaries. Examples: AlignXpert, FSPO, RLPA, RGMem, RLPF, CURIO.
  - *Response-based* profiles hold preferences over candidate outputs, often as a shared reward-feature basis with user-specific weights. Examples: RFM, PReF, LoRe, Janus, PLUS.
  - Preferences are classified as explicit vs. implicit, crossed with behavioral (tone, reasoning style) vs. topical (domains, entity likes and dislikes).
  - The agent's role is either *user-substituting* (a proxy or simulated user) or *user-adaptive* (adapts persona, tone, or autonomy). User-adaptive examples: LD-Agent, PersonaAgent, PersonaX, RPP.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.22680)
- **Memory (Sec. 4)** focuses on long-term external personal memory. Internal memory (weights, KV cache) is compact but hard to update.
  - *Textual memory* uses segmented summaries (SeCom, RMM, Nemori, CAIM, MemInsight) and multi-level designs (MIRIX, Mem-α).
  - *Structured memory* uses vector stores, trees (RAPTOR, MemTree), and graphs (Zep, AriGraph, MemWeaver, SGMem, LiCoMemory).
  - Update mechanisms:
    - Similarity-driven ADD/UPDATE/DELETE/NOOP (Mem0)
    - Learned update policies (Memory-R1)
    - Graph conflict resolution (Mem0g, MemGAS)
    - Inference-guided updates: predict-then-calibrate (Nemori), pre-storage reasoning (PREMem), self-evolving note graphs (A-mem)
  - Retrieval is content-based (dense or sparse), structure-aware (e.g., Personalized PageRank in MemGAS and AssoMem), or policy-guided (PBR, PersonaRAG, UniMS-RAG, Persona-DB).
  - Source: [arXiv HTML](https://arxiv.org/html/2602.22680)
- **Planning (Sec. 5)** comes in two forms:
  - *One-shot*: profile/memory conditioning (PRIME, PersonaAgent), preference induction, or internal refinement (TravelPlanner++, PREFINE).
  - *Feedback-driven*: clarification-seeking (SAGE-Agent's POMDP-based questioning, ECLAIR, Ask-before-Plan), user revision, or non-user feedback.
  - The stated trade-off is that one-shot planning is low-latency but brittle under sparse signals, while feedback-driven planning aligns better but costs latency and user effort.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.22680)
- **Action execution (Sec. 6)** splits into two stages:
  - *Pre-action*: preference-aware tool selection (PEToolLLaMA, PUMA, ToolSpectrum), value-guided actions (ValuePilot), clarifying before acting, and grounding. Grounding includes relaxing the least-preferred constraints (AWARE-US) and inferring missing arguments from the profile.
  - *Post-action*: personalized re-ranking (PEAR), workflow branching (WARPP), and proactive augmentation (ETAPP).
  - The survey notes that explicit preference-aware *error recovery* during execution is under-studied.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.22680)
- **Cross-component challenges named by the survey:**
  - sparse, noisy, and shifting signals
  - deciding which user features matter downstream
  - adaptability vs. role stability and drift
  - errors cascading from upstream misreadings
  - multi-objective trade-offs
  - privacy and transparency of stored personal data
  - Source: [arXiv HTML](https://arxiv.org/html/2602.22680)

**Agent-loop building blocks (planning/action layer) [PK]**
- **ReAct** interleaves reasoning traces with actions and observations. It is the base loop of most tool-using agents. — [Yao et al., arXiv 2210.03629](https://arxiv.org/abs/2210.03629) [PK]
- **Toolformer** has a model teach itself when to call APIs through self-supervised insertion of API calls. — [Schick et al., arXiv 2302.04761](https://arxiv.org/abs/2302.04761) [PK]
- **Plan-and-execute / Plan-and-Solve** first writes a full plan, then executes the steps, which reduces step-level drift. — [Wang et al., arXiv 2305.04091](https://arxiv.org/abs/2305.04091) [PK]
- **Reflexion** has the agent write verbal self-reflections into an episodic memory buffer and retry. — [Shinn et al., arXiv 2303.11366](https://arxiv.org/abs/2303.11366) [PK]. Noah Shinn, the Reflexion first author, is the founder and CEO of Instinct (Spear Street Technology); see Section 5. — [Built In](https://builtin.com/articles/instinct-ai)
- **MemGPT** applies OS-style virtual context management: the LLM pages facts between a small in-context "main memory" and external storage through function calls. Many product memory designs follow this pattern. — [Packer et al., arXiv 2310.08560](https://arxiv.org/abs/2310.08560) [PK]

**Tool / connector layer**
- **Model Context Protocol (MCP)**: Anthropic introduced it in Nov 2024 as an open client-server protocol (JSON-RPC) that exposes tools, resources, and prompts to an LLM host. By 2025–26 it was the de facto connector standard across Anthropic, OpenAI, Google, and IDE clients. — [Anthropic announcement](https://www.anthropic.com/news/model-context-protocol) [PK]; [spec](https://modelcontextprotocol.io/specification) [PK]
- MCP's security weakness is that tool descriptions themselves sit in the model's context. Invariant Labs (April 2025) showed that a malicious MCP server's tool description could silently redirect messages sent through a separate, *trusted* WhatsApp MCP server. The malicious tool never had to be called. — [Invariant Labs](https://invariantlabs.ai/blog/whatsapp-mcp-exploited); [Docker write-up](https://www.docker.com/blog/mcp-horror-stories-whatsapp-data-exfiltration-issue/)
- Apple's design exposes app actions to Siri/Apple Intelligence through **App Intents**. Reporting says making App Intents reliable across enough apps "while maintaining the precision required for sensitive scenarios" was a main reason the personal-context Siri slipped. — [AppleInsider, Feb 2026](https://appleinsider.com/articles/26/02/16/apples-upgraded-siri-is-late-but-not-lost----the-real-story-is-timing); [CO/AI](https://getcoai.com/news/apple-delays-siri-overhaul-to-2026-amid-app-intents-struggles/)

### Inferences
- Synthesizing survey and products, a practical 2026 reference stack has eight parts:
  1. **Ingestion/sensing** (chat, email, calendar, screen, audio)
  2. **Profile store**, a small, editable, high-precision summary such as "saved memories" or a memory summary
  3. **Episodic/long-term memory**, a large, retrieval-based store of past chats and documents, held as vector, graph, or hierarchical summaries
  4. **Context assembler**, which decides what personal context enters each prompt
  5. **Planner/agent loop**, a ReAct-style loop with plan-and-execute and reflection
  6. **Tool layer** (MCP, app intents, a cloud browser or computer)
  7. **Policy/permission layer** (confirmations, scoping, isolation)
  8. **Memory-write pipeline** (extract, then ADD/UPDATE/DELETE, then consolidate or forget)
- The survey's four components correspond to items 2–3 (profile and memory), 5 (planning), and 6 (action). Items 4, 7, and 8 are where most product engineering and security work happens.
- There are two product archetypes. *Assistant-with-memory* products (ChatGPT, Claude, Gemini) personalize answers. *Delegated-action agents* (Instinct, Operator-style agents) hold credentials and act. The second has a far larger risk surface; see Section 4.

### Gaps
- The survey's Section 9 (open problems) body text and Section 7.3 benchmark descriptions could not be fully extracted; only headings and cited references were. Section 9 headings are listed in Section 5 of these notes.
- No vendor has published a full architecture diagram of its production personal-agent memory pipeline. The product descriptions below come from help docs and behavior, not design docs.

---

## 2. How do systems model a user profile and keep it current?

### Takeaway
Two layers dominate. One is a **compact, explicit, user-editable profile** (OpenAI "saved memories," Claude's memory summary, Personal.ai's memory blocks). The other is **implicit retrieval over raw history** (ChatGPT "reference chat history," RAG over past chats or a lifelog). Research adds parametric options: per-user PEFT/LoRA adapters, hypernetwork-generated adapters, and reward-feature weights. Keeping profiles current is mostly handled by LLM-driven ADD/UPDATE/DELETE pipelines. Selective forgetting and temporal drift remain weak spots.

### Cited Findings
**Research methods**
- **LaMP benchmark** (Salemi et al.) is the canonical personalization benchmark. It has 7 tasks: citation identification, movie tagging, product rating, news headline, scholarly title, email subject, and tweet paraphrase. It established *retrieval-augmented personalization*, which retrieves items from the user's history into the prompt. — [arXiv 2304.11406](https://arxiv.org/abs/2304.11406) [PK]. The 2602.22680 survey cites LaMP, LongLaMP, and LaMP-QA as textual/dialogue benchmarks. — [arXiv HTML](https://arxiv.org/html/2602.22680)
- **OPPU (One PEFT Per User)** gives each user a personal PEFT module plugged into a base LLM, so users "own" their parameters.
  - It reports beating all baselines on the 7 public LaMP tasks.
  - It significantly outperforms retrieval-based methods when user behavior shifts (history becomes less relevant).
  - Combining retrieval (non-parametric) with PEFT (parametric) works best.
  - Source: [arXiv 2402.04401](https://arxiv.org/pdf/2402.04401v3)
  - A secondary summary cites +17.38% relative MAE improvement on LaMP-3 and +11.87% accuracy on LaMP-2M. **[Unverified]** — [deep-paper summary](https://deep-paper.org/en/paper/2402.04401)
- **Profile-to-PEFT** (arXiv 2510.16282) criticizes OPPU because one adapter per user is computationally expensive and impractical in real time. It uses a hypernetwork to generate adapters from a profile, and reports beating both prompt-based personalization and OPPU at lower deployment cost. — [arXiv 2510.16282](https://arxiv.org/html/2510.16282v1)
- **Preference learning / response-based profiles**: shared reward-feature bases with per-user weights (RFM, PReF, LoRe, PLUS) and RL-based personalized alignment (RLPA, FSPO, AlignXpert). — [Survey HTML](https://arxiv.org/html/2602.22680)
- **Memory update** approaches:
  - Mem0 applies LLM-decided ADD/UPDATE/DELETE/NOOP operations to candidate facts.
  - Mem0g and MemGAS resolve graph conflicts.
  - Memory-R1 learns the update policy with RL.
  - The survey evaluates forgetting with a "Selective Forgetting" metric.
  - Source: [Survey HTML](https://arxiv.org/html/2602.22680)

**Deployed products**
- **ChatGPT (OpenAI)** has two layers:
  - *Saved memories* are explicit and kept until deleted. Users can review, delete individually, or clear all.
  - *Reference chat history* draws on past conversations. These details "can change over time as ChatGPT updates what's more helpful to remember."
  - Deleting a chat does not delete a saved memory derived from it. Turning off saved memories also turns off chat history reference.
  - Temporary Chat bypasses memory.
  - Source: [OpenAI Help: reference saved memories](https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work); [OpenAI Help: memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq)
  - Original memory launch: [OpenAI blog, Feb 2024](https://openai.com/blog/memory-and-new-controls-for-chatgpt)
  - A third-party claim says chat-history memory cannot be selectively reviewed or edited, only toggled. **[Unverified]** — [memx.app](https://memx.app/blog/chatgpt-reference-chat-history-vs-saved-memories/)
- **Claude (Anthropic)** memory:
  - Launched for Team/Enterprise in Sep 2025, then Pro/Max in Oct 2025.
  - Each **Project has its own isolated memory space**, so work and personal context don't mix.
  - **Incognito chats** are excluded from history and memory.
  - Memory can be imported from other assistants (labelled experimental) and exported.
  - Source: [VentureBeat](https://venturebeat.com/ai/anthropic-adds-memory-to-claude-team-and-enterprise-incognito-for-all); [Maginative](https://www.maginative.com/article/anthropic-brings-project-based-memory-to-paid-claude-users/); [Tom's Guide](https://www.tomsguide.com/ai/claude-just-unlocked-memory-that-syncs-with-chatgpt-heres-how-it-works)
  - Free-tier availability by ~March 2026 is **[Unverified]**; sources conflict.
  - A reported Aug 25, 2026 update adds editable per-topic memories, real-time updates, and memory shared with Cowork. **[Unverified; single blog source, not confirmed against Anthropic]** — [mejba.me](https://www.mejba.me/nl/blog/claude-unified-memory-setup)
- **Personal.ai**:
  - A "Memory Stack" is a collection of memory blocks created from messages, documents, and integrations. Per-persona blocks are visible only to that persona, and the Primary Persona sees all. — [Personal.ai docs](https://docs.personal.ai/documentation/training/memory-stack)
  - The company describes a "Personal Language Model" trained on the user's Memory Stack. It is built from composite conversational, QA, generative, and retrieval models, and the company states it has ~120M parameters **[company claim, unverified]**.
  - MODEL-2 adds hierarchical sub-personas, each with separate memory banks and short- and long-term memory.
  - Source: [Personal.ai MODEL-2](https://personal.ai/model-2)
- **Rewind / Limitless**:
  - Lifelogging-style memory: the original Rewind recorded the desktop, and the Limitless pendant records and transcribes real-world conversations into searchable transcripts. — [Reuters via The Star](https://www.thestar.com.my/tech/tech-news/2025/12/06/meta-acquires-ai-wearables-startup-limitless)
  - **Meta acquired Limitless in early Dec 2025** for its Reality Labs wearables push.
  - Pendant sales to new customers stopped, and existing users get at least 1 year of support. Rewind desktop features are being wound down, and continuing users must accept new privacy terms. — [BetaNews](https://betanews.com/2025/12/08/meta-has-acquired-limitless-a-startup-that-produces-ai-powered-wearables/); [Outlook Business](https://www.outlookbusiness.com/corporate/meta-acquires-ai-wearables-start-up-limitless-as-it-doubles-down-on-personal-superintelligence)
- **Apple Intelligence personal context** uses on-device indexing of personal data (messages, mail, calendar, and so on, called the "semantic index"), plus App Intents for actions and Private Cloud Compute for larger models. — [Apple PCC security blog, June 2024](https://security.apple.com/blog/private-cloud-compute/) [PK]
  - The personal-context Siri was delayed from 2025 into 2026 and beyond. Bloomberg-sourced reporting says it relies on a Gemini-based model running partly on Google servers.
  - Launch timing conflicts across sources: some point to iOS 26.4 or 26.5, others to iOS 27 or into 2027. **[Timing unverified]** — [AppleInsider](https://appleinsider.com/articles/26/02/16/apples-upgraded-siri-is-late-but-not-lost----the-real-story-is-timing); [9to5Mac](https://9to5mac.com/2026/02/12/apple-iphone-siri-promise-this-year/); [iClarified](https://www.iclarified.com/99906/apple-delays-major-siri-ai-upgrade-as-testing-reveals-performance-snags-report)

### Inferences
- The industry has converged on "small explicit profile + large implicit retrieval." The explicit layer gives users control and auditability. The implicit layer gives recall but is opaque and harder to delete from: deleting a chat in ChatGPT does not delete derived memories.
- Per-user parametric personalization (OPPU-style adapters) shows research gains, especially under behavior shift. No major consumer assistant is publicly known to train per-user weights; all appear to use context and retrieval. Hypernetwork approaches such as Profile-to-PEFT are the likeliest path to parametric personalization at scale.
- Scoping memory per Project or persona (Claude Projects, Personal.ai personas) is both a UX feature and a containment measure. It limits cross-context leakage and limits how far a poisoned memory can reach.

### Gaps
- No primary source found on how ChatGPT's chat-history reference is implemented (summaries vs. embeddings, refresh cadence).
- No primary Anthropic doc fetched this session on the exact Claude memory mechanism (e.g., periodic synthesized summary vs. tool-based retrieval). Product details above rely on press coverage.

---

## 3. What benchmarks evaluate personal agents, and what do results show?

### Takeaway
Benchmarks fall into four groups:
- **output personalization** (LaMP, LongLaMP, LaMP-QA)
- **persona fidelity** (PersonaGym)
- **preference following and memory over long interactions** (PrefEval, PersonaMem, LoCoMo, PerLTQA)
- **proactive and task agents** (PROBE, ProAgentBench, π-Bench)

The consistent finding is that frontier models are **weak at keeping and applying user preferences over long multi-session contexts**, and **model scale does not reliably improve persona fidelity**.

### Cited Findings
- **PrefEval** (Zhao et al., ICLR 2025 oral, Amazon Science):
  - 3,000 hand-built preference/query pairs across 20 topics, tested on 10 LLMs with multi-session context up to 100k tokens.
  - In zero-shot settings, preference-following accuracy **falls below 10% at only 10 turns (~3k tokens)** for most models.
  - Reminder prompting, self-critique, few-shot CoT, and RAG help but do not solve the decline. Fine-tuning on PrefEval improves it significantly.
  - Covers explicit and implicit preferences, in generation and classification formats.
  - Source: [arXiv 2502.09597](https://arxiv.org/abs/2502.09597v1); [GitHub](https://github.com/amazon-science/prefeval)
- **PersonaGym** (arXiv 2407.18416):
  - 200 personas, 10,000 questions, and 150 environments selected per persona. Scoring is by ensembles of LLM evaluators (the PersonaScore).
  - Tested 10 models: GPT-3.5, LLaMA-2/3 variants, Claude 3 Haiku, Claude 3.5 Sonnet, GPT-4.1, GPT-4.5, LLaMA-3.3-70B, and DeepSeek-V3.
  - Claude 3.5 Sonnet was only 2.97% relatively better than GPT-3.5. GPT-4.1 scored the same as LLaMA-3-8B.
  - Authors' conclusion: "increased size or capacity is not a definite indication of persona agent capabilities." Figures differ across v1–v5; use v5 for exact numbers.
  - Source: [arXiv 2407.18416](https://arxiv.org/html/2407.18416v4)
- **LaMP / OPPU results**: per-user PEFT beats retrieval-only personalization across LaMP's 7 tasks, and especially under behavior shift. — [arXiv 2402.04401](https://arxiv.org/pdf/2402.04401v3)
- **Survey's benchmark taxonomy (Sec. 7.3)**:
  - Interactive alignment:
    - *preference discovery*: PrefDisco, PersonaBench, latent-preference discovery
    - *output-level alignment*: AgentRecBench, a personalized deep-research benchmark
    - *component-probing*: PerLTQA, LoCoMo, Mem-PAL, PersonaMem-v2, PersonaFeedback, ConvoMem, PrivacyBench, a personalized tool-invocation benchmark
  - User substitution:
    - *textual*: LaMP, LongLaMP, LaMP-QA, PRISM
    - *planning/tool use*: TripTailor, TripTide, COMPASS
    - *embodied*
  - Source: [Survey HTML](https://arxiv.org/html/2602.22680)
- **Survey's metric taxonomy**:
  - *Effectiveness*: Preference Alignment, Preference-Aware Planning Accuracy, Preference Lift, Consistency
  - *Adaptivity*: Adaptation Success Rate, Selective Forgetting, Interaction Efficiency, Proactivity
  - *Generalization*: Out-of-Domain Performance
  - *Robustness*: Accuracy under Ambiguity, Misinformed Condition Performance, Safety Rate
  - *Risk*: Risk Sensitivity, Privacy Leakage Rate, Over-Secrecy Rate
  - Assessment is by automatic scoring, rules, learned evaluators, and LLM-as-judge.
  - Source: [Survey HTML](https://arxiv.org/html/2602.22680)
- **Proactive-agent benchmarks**:
  - **PROBE** has 1,000 samples. Agents must search a workplace datastore, find a hidden bottleneck, and act; "even state-of-the-art models struggle." — [arXiv 2510.19771](https://arxiv.org/html/2510.19771v1)
  - **ProAgentBench** has 28k+ events from 500+ hours of real user screen sessions. It splits proactivity into *timing prediction* and *assist content generation*. — [arXiv 2602.04482](https://arxiv.org/pdf/2602.04482.pdf)
  - **π-Bench** has 100 multi-turn tasks across 5 personas, each starting from an underspecified request with hidden intents. It finds proactive assistance remains challenging and that task completion and proactivity are distinct. — [arXiv 2605.14678](https://www.alphaxiv.org/abs/2605.14678.md)
- **Long-term memory QA**: LoCoMo (Maharana et al., 2024) evaluates very long-term conversational memory, and PerLTQA evaluates personal long-term memory QA. Both are cited by the survey. — [Survey HTML](https://arxiv.org/html/2602.22680)

### Inferences
- PrefEval's collapse below 10% at ~3k tokens shows that a long context window alone is not personalization. Explicit preference extraction plus re-injection, or fine-tuning, is needed. This supports the product pattern of distilling a compact profile.
- Persona fidelity, preference recall, and proactivity are partly separate capabilities from general model strength (PersonaGym, π-Bench). Personal agents may therefore need targeted training or scaffolding rather than relying on frontier-model upgrades.
- Evaluation is fragmented. No benchmark yet covers the full loop of profile, memory, plan, act, and privacy in one setting. The survey itself lists evaluation and benchmarking as an open problem.

### Gaps
- No current (2026) leaderboard numbers were retrieved for PrefEval or PersonaMem with frontier models such as GPT-5-class or Claude 4-class. It is unknown how much newer models closed the long-context preference gap.
- LongLaMP and LaMP-QA results were not retrieved.

---

## 4. What are the main risks (privacy, prompt injection, over-personalization and sycophancy), and how are they mitigated?

### Takeaway
The dominant technical risk is **indirect prompt injection** combined with persistent memory and outbound channels. Examples include SpAIware against ChatGPT memory, MINJA memory poisoning, and MCP tool poisoning. Willison's "lethal trifecta" (private data + untrusted content + external communication) is the standard threat framing. Mitigations shifted from guardrail classifiers to **architectural separation**: removing one leg of the trifecta, Meta's "Agents Rule of Two," and CaMeL-style dual-LLM / capability designs. They are complemented by user-visible, editable, scoped memory. Privacy risk is increasingly about **terms of service and data custody** (Instinct, Limitless→Meta), alongside the technical design. Over-personalization and sycophancy is a recognized but less-solved risk.

### Cited Findings
**Prompt injection & memory attacks**
- **Lethal trifecta** (Willison, 16 Jun 2025): an agent with (1) access to private data, (2) exposure to untrusted content, and (3) the ability to communicate externally can be made to exfiltrate data.
  - Willison argues guardrail products that stop "95% of attacks" are a failing grade, and that users and developers must avoid combining all three.
  - He later endorsed Meta's **"Agents Rule of Two,"** which also covers state-changing actions.
  - Source: [Simon Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
- **SpAIware** (Johann Rehberger, Sep 2024):
  - Indirect prompt injection from a malicious website or document caused the ChatGPT macOS app to write an instruction into **long-term memory**. This enabled continuous exfiltration of all future chats across sessions.
  - OpenAI first treated it as a "safety" issue, then fixed the exfil vector in v1.2024.247. Researchers noted that untrusted content can still write to memory.
  - Mitigation advice: regularly review stored memories.
  - Source: [The Hacker News](https://thehackernews.com/2024/09/chatgpt-macos-flaw-couldve-enabled-long.html); [TechSpot](https://www.techspot.com/news/104881-chatgpt-vulnerability-allows-hackers-record-sessions-indefinitely.html)
- **MINJA** is query-only memory injection: the attacker only interacts with the agent and needs no write access.
  - It reports 98.2% average injection success and 76.8% average attack success across 3 agents.
  - A 2026 follow-up (arXiv 2601.05504) argues these were idealized conditions (minimal initial memory, fixed retrieval parameters) and that robustness in production is less clear.
  - Source: [arXiv 2503.03704](https://arxiv.org/html/2503.03704v2); [arXiv 2601.05504](https://arxiv.org/html/2601.05504v1)
- **AgentPoison** is a backdoor-style poisoning of long-term memory or RAG knowledge bases. It reports >80% average attack success with minimal benign degradation, but assumes write access to the memory store, a stronger threat model than MINJA. — [MINJA paper comparison](https://arxiv.org/html/2503.03704v2); original: [arXiv 2407.12784](https://arxiv.org/abs/2407.12784) [PK]
- **MCP tool poisoning**: hidden instructions in a third-party MCP tool's description hijacked a trusted WhatsApp MCP server to exfiltrate chat history.
  - It bypassed per-tool user approval and worked against several MCP clients.
  - "Rug pull" variants change the tool definition after initial approval.
  - Invariant's mitigation is mcp-scan / pinning of tool descriptions.
  - Source: [Invariant Labs](https://invariantlabs.ai/blog/whatsapp-mcp-exploited); [GitHub experiments](https://awesome.ecosyste.ms/projects/github.com%2Finvariantlabs-ai%2Fmcp-injection-experiments)
- **Delegated-action agents in the wild**: users of Instinct reported unauthorized actions (storing and sending emails, logging into sites with sign-up codes), and one user found it susceptible to phishing. **[Unverified user reports via Built In]** — [Built In](https://builtin.com/articles/instinct-ai)

**Architectural mitigations [PK]**
- **CaMeL** (Google DeepMind, 2025):
  - A privileged LLM plans from the trusted user query only. A quarantined LLM parses untrusted data but cannot call tools.
  - A capability/data-flow interpreter blocks untrusted data from flowing into sensitive sinks.
  - It formalizes Willison's "dual LLM" pattern.
  - Source: [arXiv 2503.18813](https://arxiv.org/abs/2503.18813) [PK]
- **Meta "Agents Rule of Two"** (Oct/Nov 2025): within one session an agent should satisfy at most two of three properties: [A] processes untrustworthy inputs, [B] has access to sensitive systems or private data, [C] can change state or communicate externally. Otherwise it requires human-in-the-loop approval. — [Meta AI blog](https://ai.meta.com/blog/practical-ai-agent-security/) [PK; Willison endorsement confirmed via search above]
- Product-level controls confirmed this session:
  - Editable, deletable saved memories and temporary or incognito chats that bypass memory (OpenAI, Anthropic)
  - Project-scoped memory isolation (Claude)
  - Per-persona memory partitions (Personal.ai)
  - Sources: [OpenAI Help](https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work); [VentureBeat](https://venturebeat.com/ai/anthropic-adds-memory-to-claude-team-and-enterprise-incognito-for-all); [Personal.ai docs](https://docs.personal.ai/documentation/training/memory-stack)

**Privacy & data custody**
- **Apple Private Cloud Compute**: personal requests that exceed on-device capacity run on Apple-silicon servers with stateless processing (no data retention) and no privileged admin access. Software images are publicly logged so researchers can verify them. — [Apple Security blog](https://security.apple.com/blog/private-cloud-compute/) [PK]
  - Reported reliance on Gemini running on Google servers for the new Siri raises open questions about how PCC-style guarantees extend. **[Unverified/unclear from public sources]** — [AppleInsider](https://appleinsider.com/articles/26/02/16/apples-upgraded-siri-is-late-but-not-lost----the-real-story-is-timing)
- **Instinct ToS controversy**:
  - Before 26 Aug 2026, the terms reportedly granted a "sub-licensable, worldwide, perpetual and irrevocable license" over user materials.
  - The revised terms claim no ownership but allow use of data to train models unless the user opts out, and allow disclosure of de-identified or aggregated usage data "for any lawful purpose."
  - The user's data, model, and agent computer all live on Spear Street infrastructure, and the product cannot be self-hosted.
  - **[Primarily third-party reporting citing TechCrunch]** — [Built In](https://builtin.com/articles/instinct-ai); [eesel review](https://www.eesel.ai/blog/instinct-ai-review); [AI Weekly/TechCrunch summary](https://aiweekly.co/node/10806)
- **Limitless → Meta**: continuing users had to accept new privacy terms after the acquisition. This shows lifelog data custody can change with corporate control. — [BetaNews](https://betanews.com/2025/12/08/meta-has-acquired-limitless-a-startup-that-produces-ai-powered-wearables/)
- The survey lists Privacy Leakage Rate and *Over-Secrecy Rate* as risk metrics, which captures the trade-off between leaking and refusing to use personal data. — [Survey HTML](https://arxiv.org/html/2602.22680)

**Over-personalization & sycophancy**
- OpenAI rolled back a GPT-4o update in April 2025 for being "overly flattering or agreeable" (sycophantic). It attributed the problem partly to over-weighting short-term user feedback (thumbs-up) signals and noted that memory could amplify the effect. — [OpenAI "Sycophancy in GPT-4o"](https://openai.com/index/sycophancy-in-gpt-4o/) and [Expanding on sycophancy](https://openai.com/index/expanding-on-sycophancy/) [PK]
- The survey's "balancing adaptability against role stability and drift" challenge is the academic counterpart. — [Survey HTML](https://arxiv.org/html/2602.22680)

### Inferences
- Persistent memory turns a one-shot prompt injection into a **persistent compromise**. A memory write is therefore a sensitive "sink" and should be treated like an outbound action: shown to the user, provenance-tagged, and blocked from untrusted-content contexts.
- Delegated agents that hold credentials and a cloud computer (Instinct) satisfy all three trifecta legs by design. Unless they implement Rule-of-Two-style confirmations or CaMeL-style data-flow control, they are structurally exposed. Public information on Instinct's defenses is absent.
- Personalization feedback loops (thumbs-up, memory of user opinions) are a plausible mechanism for sycophancy and echo-chamber effects. Mitigations would separate "what the user likes in style" from "what the user believes is true." This separation is not standard in any documented product.

### Gaps
- No public technical description of injection defenses was found for ChatGPT memory writes, Claude memory, or Instinct beyond user controls.
- No quantitative study found on sycophancy amplification caused by memory or personalization specifically, as opposed to general sycophancy.
- No reliable current information on whether Apple's Gemini-backed Siri runs on PCC or on Google infrastructure under separate guarantees.

---

## 5. Open problems, plus deployed-product notes

### Takeaway
The survey's own future directions are:
- decision-critical user modeling
- temporal dynamics and continual personalization
- generalization
- evaluation and benchmarking
- privacy and user control
- efficiency and deployment

Evidence gathered here adds memory integrity under adversarial input, proactivity timing, data custody, and calibrated autonomy for credentialed agents.

### Cited Findings
- **Survey Section 9 open problems** (headings): decision-critical user modeling; temporal dynamics and continual personalization; generalization; evaluation and benchmarking; privacy and user control; efficiency and deployment. — [Survey HTML](https://arxiv.org/html/2602.22680)
- **Execution-time error recovery** that respects preferences is under-studied (survey Sec. 6). — [Survey HTML](https://arxiv.org/html/2602.22680)
- **Long-context preference retention** is unsolved by prompting or RAG; only fine-tuning helped in PrefEval. — [arXiv 2502.09597](https://arxiv.org/abs/2502.09597v1)
- **Per-user adaptation cost**: per-user adapters are expensive, and hypernetworks are one answer. — [arXiv 2510.16282](https://arxiv.org/html/2510.16282v1)
- **Proactivity** (when to interrupt, what to offer) is hard even for frontier models. — [PROBE](https://arxiv.org/html/2510.19771v1); [ProAgentBench](https://arxiv.org/pdf/2602.04482.pdf); [π-Bench](https://www.alphaxiv.org/abs/2605.14678.md)
- **Memory poisoning robustness** in realistic memory sizes is an open evaluation question. — [arXiv 2601.05504](https://arxiv.org/html/2601.05504v1)

**Deployed product snapshot (Nov 2026)**
- **Instinct (Spear Street Technology, Inc., San Francisco, dba Instinct)**:
  - An invite-only personal agent with no app. Users reach it by SMS, phone call, or WhatsApp, a "zero UI" approach.
  - It runs on a cloud virtual computer and browser, connects to mail, messaging, and calendar, and uses the user's stored credentials. It reportedly asks permission before purchases.
  - The founder and CEO is Noah Shinn (ex-Sierra research scientist; Reflexion author [PK]). Underlying models are not disclosed, pricing is not published, and current users get it free.
  - Funding and valuation figures conflict **[Unverified]**: WSJ-reported $2.5B valuation in Aug 2026; TechCrunch-reported $10B by end of Sep 2026; a reported $250M Series B co-led by Index and Benchmark; a SiliconANGLE headline of $1B raised at $10B.
  - Source: [Built In](https://builtin.com/articles/instinct-ai); [Vellum breakdown](https://www.vellum.ai/blog/official-instinct-breakdown); [Sacra](https://sacra.com/c/instinct/); [SiliconANGLE](https://siliconangle.com/?p=848490)
- **ChatGPT**: two-tier memory (saved memories + chat history reference), Temporary Chat, and Tasks for scheduled actions. — [OpenAI Help](https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work); [Tasks](https://help.openai.com/en/articles/8590148-tasks)
- **Claude**: Project-scoped memory, incognito, and memory import/export. — [VentureBeat](https://venturebeat.com/ai/anthropic-adds-memory-to-claude-team-and-enterprise-incognito-for-all)
- **Apple Intelligence**: on-device personal context, App Intents, and PCC. The new Siri was repeatedly delayed and is reportedly Gemini-backed, with timing unresolved. — [AppleInsider](https://appleinsider.com/articles/26/02/16/apples-upgraded-siri-is-late-but-not-lost----the-real-story-is-timing)
- **Rewind/Limitless**: lifelog capture plus transcription plus search. Acquired by Meta in Dec 2025, with new sales halted. — [Reuters via The Star](https://www.thestar.com.my/tech/tech-news/2025/12/06/meta-acquires-ai-wearables-startup-limitless)
- **Personal.ai**: per-user Memory Stack and a small "Personal Language Model" with persona-partitioned memory. — [Personal.ai](https://personal.ai/model-2)

### Inferences
- The frontier problem has moved from "can the model remember?" to **"can it remember the right thing, update or forget correctly, resist poisoning, and act safely with credentials?"** These are integration problems across the survey's four components, not single-module problems.
- On-device or confidential-compute personal context (Apple) and cloud-hosted delegated agents (Instinct) are opposite bets on the privacy-vs-capability frontier. Apple's turn toward Gemini suggests pure on-device capability was insufficient for its 2026 goals **[inference]**.
- Memory portability (Claude import from ChatGPT/Gemini) hints at emerging user-owned profiles. No standard memory interchange format was found.

### Gaps
- Instinct has no official technical documentation. Its memory design, model choice, and injection defenses are unknown.
- On-device personal agents outside Apple (e.g., Gemini Nano personal context, Samsung) were not researched due to the tool-call budget.
- The full text of survey Section 9 was not extracted; only the headings are recorded.
