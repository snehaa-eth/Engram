# Token Saving for LLMs: Prompt Compression and Context Reduction (2016–2026)

Scope note: numbers below were checked against arXiv abstract pages or arXiv HTML full text, the official microsoft/LLMLingua README, and the Anthropic/OpenAI prompt caching docs as fetched on 2026-11-08. Anything I could not verify against a primary source is marked **[UNVERIFIED]**. Pre-2023 history (2016–2022) is mostly context for the compression lineage and was not separately source-checked.

Rough timeline (dates are arXiv v1 submissions where I fetched them):
- 2016–2022 (background, [UNVERIFIED] as a timeline): extractive and abstractive summarization, and the KV-cache and sparse-attention ideas the later methods build on. "Prompt compression" as a named field begins in 2023.
- Apr 2023: Gisting (soft gist tokens). May 2023: AutoCompressors (summary vectors). Jun 2023: H2O (KV eviction). Jul 2023: ICAE. Sep 2023: StreamingLLM. Oct 2023: Selective Context, LLMLingua, RECOMP, LongLLMLingua.
- Mar 2024: LLMLingua-2. Apr 2024: SnapKV. May 2024: xRAG. Aug 2024: 500xCompressor.
- 2025: LLM-DCP (RL), SCOPE (generative chunk summarization), info-preservation critique (EMNLP Findings 2025), Cmprsr (Nov 2025), SecurityLingua (CoLM 2025).
- 2026: critique of benchmark-dependent output expansion (Mar 2026). Over the same period, providers turned prompt caching into the cheapest and most dependable way to cut token cost.

---

## Q1. For each major method: mechanism, compression ratio, quality retained, latency overhead, black-box compatibility

### Takeaway
The methods fall into four families. **Hard-prompt pruning or rewriting** (Selective Context, LLMLingua family, LLM-DCP, SCOPE, Cmprsr, RECOMP) outputs text, so it works with black-box APIs; typical safe ratios are 2–5x, with claims of up to 20x. **Soft-prompt or embedding compression** (Gisting, AutoCompressors, ICAE, 500xCompressor, xRAG) reaches 4–480x but needs white-box access to the target model's embeddings or KV cache, so it cannot be used with Claude or GPT. **KV-cache compression** (H2O, StreamingLLM, SnapKV) is a serving-side technique that saves memory and latency, not billed input tokens. **Provider prompt caching** does not shrink the prompt; it cuts the price of repeated prefixes to 0.05–0.1x the base input rate.

### Cited Findings

**Hard-prompt pruning (extractive, token or sentence deletion)**
- **Selective Context** (Li et al., Oct 2023): a small causal LM computes self-information, and low-information lexical units are dropped. Reported results: "50% reduction in context cost", "36% reduction in inference memory usage", "32% reduction in inference time", and a drop of only ".023 in BERTscore and .038 in faithfulness" across four downstream applications. — [arXiv:2310.06201](https://arxiv.org/abs/2310.06201)
- **LLMLingua** (Jiang et al., EMNLP 2023): coarse-to-fine compression with a budget controller, a token-level iterative algorithm that models dependencies between compressed segments, and instruction-tuning to align the small LM's distribution with the target LM's. Claims "up to 20x compression with little performance loss" on GSM8K, BBH, ShareGPT and Arxiv-March23. The abstract reports no speedup figure. — [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
  - README quick-start example: 2,365 tokens compressed to 211 ("11.2x"), saving about $0.1 per GPT-4 call. Compressor models include GPT2-small and LLaMA-7B. — [microsoft/LLMLingua README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)
- **LongLLMLingua** (ACL 2024 Long Papers): question-aware compression for long and RAG contexts, with document reordering and dynamic per-document ratios. README claim: "improving RAG performance by up to 21.4% using only 1/4 of the tokens". Example parameters: rate=0.55, dynamic_context_compression_ratio=0.3–0.4. — [microsoft/LLMLingua README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)
  - Per-benchmark numbers (NaturalQuestions, LooGLE cost reduction, latency) are **[UNVERIFIED]**; I did not read the LongLLMLingua results tables.
- **LLMLingua-2** (Pan et al., ACL 2024 Findings): treats compression as token classification (keep or drop) with a bidirectional encoder (XLM-RoBERTa-large or mBERT), trained on an extractive dataset distilled from GPT-4. Reported: "3x-6x faster than existing prompt compression methods" and 1.6x–2.9x faster end-to-end latency at 2x–5x compression. Evaluated on MeetingBank, LongBench, ZeroScrolls, GSM8K and BBH. It is task-agnostic (question-agnostic). — [arXiv:2403.12968](https://arxiv.org/abs/2403.12968)
  - The LLMLingua-2 compressor is about 0.55B parameters (stated in the Cmprsr paper). — [arXiv:2511.12281 HTML](https://arxiv.org/html/2511.12281)
  - Integrations: LangChain, LlamaIndex, Prompt flow. — [README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)
- **SecurityLingua** (CoLM 2025): applies the LLMLingua idea to jailbreak defense. It claims "100x less token costs compared to state-of-the-art LLM guardrail approaches". — [README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)

**RL-based pruning**
- **LLM-DCP, "Dynamic Compressing Prompts"** (Hu et al., Apr 2025, preprint, under review): models compression as an MDP in which a DCP-Agent (an xlm-roberta-large actor with a keep/discard head, plus a critic) removes tokens step by step. The reward balances compression rate, output quality and key-information retention. "Hierarchical Prompt Compression" is a curriculum that gradually increases compression difficulty. — [arXiv:2504.11004](https://arxiv.org/abs/2504.11004)
  - Results, target GPT-4o-mini (black-box), from the HTML tables. — [arXiv:2504.11004 HTML](https://arxiv.org/html/2504.11004)
    - ShareGPT at about 3.4x: BLEU 64.93 vs 61.97 for LLMLingua-2 and 38.71 for LLMLingua; BERTScore F1 91.80 vs 90.87.
    - Arxiv-March23 at 12.9x: ROUGE-2 15.94 vs 14.62 for LLMLingua-2 (which ran at 12.0x).
    - GSM8K, 1-shot: EM 77.03 at 6.9x vs 76.87 for LLMLingua-2 at 5.7x. Half-shot: 77.03 at 15.5x vs LLMLingua-2's 76.80 at 16.9x.
    - BBH, 1-shot: EM 83.16 at 3.1x.
  - Training uses Llama-3-8B as a distribution-alignment proxy for the target model rather than querying the black-box during RL. The paper reports no latency figures. Compressed prompts are "poorly readable". Only GPT-4o-mini and GLM-4-Plus were tested. — [arXiv:2504.11004 HTML](https://arxiv.org/html/2504.11004)

**Abstractive or generative compressors**
- **RECOMP** (Xu et al., Oct 2023): compresses retrieved RAG documents before they reach the LM. Two compressors: an extractive one (sentence selection) and an abstractive one (multi-document summary). Reports "a compression rate of as low as 6%" with "minimal loss in performance" on language modeling and open-domain QA. In selective augmentation, it "can return an empty string" when the retrieved documents are irrelevant. — [arXiv:2310.04408](https://arxiv.org/abs/2310.04408)
- **SCOPE** (Zhang, Wang & Wang, Aug 2025): training-free chunk-level rewriting. It does semantic chunking (400–700-token chunks), summarizes each chunk with a small public summarizer such as bart-large-cnn, and assigns dynamic per-chunk ratios while preserving keywords. It needs no training and no LLM calls. On GovReport it stays close to the uncompressed baseline at 2x and 3x, with some drop at 3x. — [arXiv:2508.15813](https://arxiv.org/abs/2508.15813) (as summarized in search results; I did not read its tables, so exact scores are **[UNVERIFIED]**)
- **Cmprsr** (Zakazov et al., EPFL/West lab, Nov 2025, revised Jan 2026): abstractive, token-level, question-agnostic compression. It benchmarks 25 LLMs as compressors, applies TextGrad to optimize gpt-4.1-mini's meta-prompt, and post-trains Qwen3-4B with SFT then GRPO (summary-quality reward plus length reward). — [arXiv:2511.12281](https://arxiv.org/abs/2511.12281)
  - Results with gpt-5-mini as the black-box solver (precision), at compression rates of 0.1, 0.3 and 0.5 of the original length. — [arXiv:2511.12281 HTML](https://arxiv.org/html/2511.12281)
    - MeetingBank QA (uncompressed 0.841): Cmprsr 0.443 / 0.688 / 0.767; LLMLingua-2 0.448 / 0.681 / 0.786; GPT-4.1-mini 0.503 / 0.697 / 0.746.
    - GSM8K (uncompressed 0.930): Cmprsr 0.130 / 0.467 / 0.720; LLMLingua-2 0.003 / 0.120 / 0.513; GPT-4.1-mini 0.493 / 0.807 / 0.910.
  - Rate adherence: vanilla LLMs adhere poorly (Qwen3-4B ΔCR -0.26 at 0.5, so it over-compresses). LLMLingua-2 stays within ±0.03, and Cmprsr is about 0.
  - Model size and release date don't predict compressor quality: gpt-4.1-mini beat gpt-4.1 and gpt-5-mini.
  - A QA-based reward was "gamed" by the model listing many possible answers.
  - Autoregressive generation adds latency, so the authors position it for precomputed compression. It was trained only on MeetingBank, so code and medical domains are untested.

**Soft-prompt or latent methods (white-box only)**
- **Gisting** (Mu et al., Apr 2023): fine-tunes the LM to compress a prompt into "gist" tokens via attention masking. Reports "26x compression", "40% FLOPs reductions" and "4.2% wall time speedups" on LLaMA-7B and FLAN-T5-XXL. — [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)
- **AutoCompressors** (Chevalier et al., May 2023): OPT and Llama-2 models fine-tuned to recursively produce "summary vectors" over sequences up to 30,720 tokens. Summary vectors can replace in-context demonstrations, raising accuracy while cutting cost, and can be precomputed for RAG corpora. The abstract gives no numeric figures. — [arXiv:2305.14788](https://arxiv.org/abs/2305.14788)
- **ICAE** (Ge et al., Jul 2023): an LLM-based encoder (LoRA) compresses context into memory slots for a frozen Llama decoder. Reports "4×" compression with "about 1%" additional parameters. — [arXiv:2307.06945](https://arxiv.org/abs/2307.06945)
- **500xCompressor** (Li et al., Aug 2024): compresses context into as few as one special token, at ratios of "6x to 480x". The LLM "retained 62.26-72.89% of its capabilities" vs uncompressed prompts, with about 0.3% extra parameters. KV values preserve information better than embeddings at high ratios. — [arXiv:2408.03094](https://arxiv.org/abs/2408.03094)
- **xRAG** (Cheng et al., May 2024): projects a retriever's document embedding into the LM's representation space as one token ("one token" per document), training only a modality bridge while the retriever and LM stay frozen. Reports FLOPs reduced "by a factor of 3.53" and an average gain of more than 10% across six knowledge-intensive tasks (relative to no-retrieval baselines; the abstract's phrasing). Backbones range from 7B dense to 8x7B MoE. — [arXiv:2405.13792](https://arxiv.org/abs/2405.13792)

**KV-cache compression (serving side; white-box or self-hosted only)**
- **H2O** (Zhang et al., Jun 2023): keeps a 20% "heavy hitter" plus recent-token KV budget. Throughput rises up to 29x vs DeepSpeed Zero-Inference and HF Accelerate, and 3x vs FlexGen, on OPT-6.7B/30B. Latency falls by up to 1.9x. — [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- **StreamingLLM** (Xiao et al., Sep 2023): keeps KV for the initial "attention sink" tokens plus a sliding window, enabling stable modeling of "up to 4 million tokens and more" and a 22.2x speedup over sliding-window recomputation (Llama-2, MPT, Falcon, Pythia). It does not extend the usable context: mid-context information is evicted. — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **SnapKV** (Li et al., Apr 2024): an observation window at the end of the prompt votes for which KV positions to keep, per head. At 16K input it reports 3.6x faster generation and 8.2x better memory efficiency, handles up to 380K tokens on one A100-80GB, and shows comparable performance on 16 long-sequence datasets. — [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- Microsoft's related serving work, MInference (Jul 2024), claims pre-filling "up to 10X" faster on an A100 at 1M tokens. — [README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)

**Provider prompt caching (black-box, no quality loss)**
- **Anthropic:**
  - Pricing, as a multiple of base input price: 5-minute cache write 1.25x; 1-hour write 2x; cache read 0.1x. Reads are 0.05x for Opus 5.5 and Sonnet 5.5, and 0.025x for Fable 5.1 and Mythos 5.1. Example: Opus 5.5 at $4/MTok base costs $5 for a 5-minute write, $8 for a 1-hour write and $0.20 per MTok read.
  - Minimum cacheable prefix: 512 tokens on current models, and 1,024–4,096 on older ones.
  - Up to 4 breakpoints; lookback of 20 blocks. Automatic caching is available as a single top-level `cache_control` field.
  - Cache hits require an exact prefix match, and any change cascades tools → system → messages. Output is identical to uncached output.
  - Source: [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **OpenAI:**
  - Caching is automatic for prefixes of at least 1,024 tokens (GPT-5.6 and later).
  - Reads cost 0.1x (0.05x for GPT-6.1 Sol and Terra), and the docs say cached input is "discounted up to 95%". Writes cost 1.25x (30-minute default) or 2.5x (12-hour option).
  - `prompt_cache_key` helps routing. The docs advise putting stable content first and appending rather than rewriting history.
  - The docs warn that "summarization, compaction, or truncation can change the prefix and reduce reuse."
  - Source: [OpenAI prompt caching docs](https://developers.openai.com/api/docs/guides/prompt-caching)

### Inferences
- **Black-box compatibility:**
  - Works with Claude or GPT: Selective Context, LLMLingua, LongLLMLingua, LLMLingua-2, LLM-DCP, RECOMP, SCOPE, Cmprsr, and prompt caching. All of these output text or operate at the API.
  - Does not: Gisting, AutoCompressors, ICAE, 500xCompressor, xRAG, H2O, StreamingLLM and SnapKV. All need weights, embeddings or KV access.
- **Latency:** extractive encoders (LLMLingua-2, about 0.55B) add the least latency. Causal-LM scorers (Selective Context, LLMLingua with LLaMA-7B) add more. Autoregressive abstractive compressors (Cmprsr at 4B, LLM summarizers) add the most and suit offline precomputation.
- **Interaction with caching:** prompt compression and prompt caching partly conflict. Compressing content that varies per query breaks prefix identity, so compression is best applied to the dynamic tail (retrieved docs, tool outputs) while the stable head stays cached.

### Gaps
- I did not read the LongLLMLingua or LLMLingua-2 per-task results tables, so per-dataset retention at each ratio is unverified.
- No primary source gave LLMLingua's wall-clock speedup. A frequently cited 1.7–5.7x figure is **[UNVERIFIED]**.
- I did not read SCOPE's tables or AutoCompressors' numeric results.
- No primary-source number for latency reduction from Anthropic or OpenAI caching. The current docs give no percentage; older "up to 85% / 80% latency" figures are **[UNVERIFIED]** for current models.
- I found no fetched primary source on structured outputs (JSON schema) as a token-saving method, so that is a gap.

---

## Q2. Which methods are practical for a single developer building a personal AI assistant on API models (Claude/GPT) today?

### Takeaway
In order of payoff: (1) prompt caching with a cache-friendly prompt layout (stable system prompt, tools and memory first; append-only history). (2) Retrieval with tight top-k plus selective augmentation, i.e. retrieve less and allow "no context". (3) Rolling summarization or compaction of old turns, done sparingly because it breaks the cache. (4) Optionally, LLMLingua-2 on bulky retrieved documents or tool outputs at a conservative 2–3x, but not on instructions, code or numbers. Soft-prompt and KV methods don't apply to closed APIs.

### Cited Findings
- **Caching economics:**
  - A cache read costs 0.1x base input (0.05x on Opus/Sonnet 5.5). A 5-minute write costs 1.25x and a 1-hour write 2x, so caching pays off after roughly one reuse within the TTL. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
  - The minimum cacheable length is 512 tokens on current Claude models. Changing tool definitions invalidates everything, and toggling thinking or effort settings invalidates the message cache. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
  - OpenAI caches automatically at 1,024 tokens or more, recommends putting stable content first and dynamic content (timestamps, user details) last, and recommends `tool_choice: "none"` rather than removing tools. — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching)
- **Summarization vs caching:** OpenAI explicitly warns that summarization, compaction or truncation changes the prefix and reduces cache reuse. — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching)
- **LLMLingua-2 as a drop-in:** a pip-installable, 0.55B encoder that integrates with LangChain and LlamaIndex, offers 1.6–2.9x end-to-end speedup at 2–5x compression, and is task-agnostic. — [arXiv:2403.12968](https://arxiv.org/abs/2403.12968); [README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md); [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- **Using a cheap LLM as the compressor:** a small API model (gpt-4.1-mini) used as a vanilla abstractive compressor kept GSM8K at 0.910 vs 0.930 uncompressed at CR 0.5, while LLMLingua-2 fell to 0.513. — [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- **Selective retrieval:** RECOMP's empty-string option, which adds no context when retrieval is irrelevant, is a cheap pattern to copy in any RAG pipeline. — [arXiv:2310.04408](https://arxiv.org/abs/2310.04408)

### Inferences
- For a personal assistant whose costs are dominated by a repeated system prompt, tool schemas and long-term memory, caching alone likely saves more than any compression method, with zero quality risk.
  - Layout: `[tools][system + persona][long-term memory snapshot: cached, 1h TTL][conversation history: append-only, automatic breakpoint][retrieved context + new user message]`.
- Update the memory snapshot in batches (e.g. once per session) rather than every turn, so the cached prefix survives.
- When compacting history, do it rarely and in large steps (e.g. summarize the oldest 50% once the window passes a threshold). Each compaction costs one cache rewrite.
- For abstractive compression of retrieved docs, a cheap API model (Haiku-class or mini-class) with an explicit target length is a reasonable black-box option. It costs an extra call, so it only pays off when the main model is much more expensive or the same doc is reused.
- Don't bother with RL compressors (LLM-DCP) or soft prompts. The former needs training infrastructure for marginal gains (about 0.2 EM on GSM8K); the latter is incompatible with APIs.

### Gaps
- I found no primary benchmark measuring end-to-end dollar savings of caching plus compression in a conversational assistant.
- I did not fetch provider docs on context editing or compaction features or on structured outputs; whether they reduce billed tokens is unverified here.

---

## Q3. Where does compression break (reasoning, code, numbers, hallucination)?

### Takeaway
Extractive token pruning breaks at high ratios on short, dense prompts such as math and code, where every token matters: LLMLingua-2 collapsed to 0.003 accuracy on GSM8K at 10x. Compression also drops entities and details needed for complex tasks. And it can paradoxically increase total cost, because models whose instructions were truncated produce much longer outputs (up to about 56x output expansion on MBPP).

### Cited Findings
- **Math or reasoning collapse with extractive pruning:** at compression rates of 0.1 / 0.3 / 0.5, LLMLingua-2 scored 0.003 / 0.120 / 0.513 on GSM8K (uncompressed 0.930), vs 0.493 / 0.807 / 0.910 for abstractive gpt-4.1-mini. — [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- **Information loss:** state-of-the-art soft and hard methods "lose key details", which limits performance on complex tasks. Controlling compression granularity in a soft method gave up to +23% downstream, +8 BERTScore of grounding, and 2.7x more entities preserved. The authors conclude that soft prompting with sequence-level training gives the best trade-off. — [arXiv:2503.19114](https://arxiv.org/abs/2503.19114) (EMNLP 2025 Findings)
- **Output expansion and benchmark dependence:** across 5,400 API calls, DeepSeek at r=0.3 expanded outputs about 56x on MBPP (instruction survival probability Ψ≈0.15) but only about 5x on HumanEval (Ψ≈0.72). GPT-4o-mini was relatively stable. The author argues that prompt structure, not provider, moderates the effect, that single-benchmark evaluation misleads, and that "token savings can overstate joule savings". — [arXiv:2603.23527](https://arxiv.org/abs/2603.23527) (single-author v1 preprint, not peer-reviewed)
- **Abstractive compressor failure modes:**
  - Vanilla LLM compressors don't hit target lengths: Qwen3-4B's ΔCR was -0.26 at a target of 0.5.
  - RL with a QA reward was reward-hacked (the compressor listed many candidate answers).
  - Cmprsr is untested on code and medical domains.
  - Source: [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- **Soft-compression capability ceiling:** even the best extreme compression keeps only 62–73% of capability (500xCompressor). — [arXiv:2408.03094](https://arxiv.org/abs/2408.03094)
- **Readability and trust:** token-pruned prompts are poorly readable, which hampers debugging. LLM-DCP deliberately excludes fluency from its reward. — [arXiv:2504.11004 HTML](https://arxiv.org/html/2504.11004)
- **KV eviction is irreversible:** StreamingLLM's sink plus window design keeps generation stable but discards mid-context KV, so information evicted from the middle cannot be recalled. This is my inference from the mechanism described in [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).

### Inferences
- Never compress instructions, output-format specs, code, numbers, IDs, names or dates. Compress only narrative or reference material, such as retrieved prose and verbose tool logs.
- Measure total cost (input plus output) and task success, not the input compression ratio. 2603.23527 shows output growth can wipe out input savings.
- Hallucination risk rises when compressed context drops grounding details (2503.19114's grounding metric). Abstractive compressors add a second risk: the compressor itself can hallucinate content that wasn't in the source. I found no source that quantifies this for Cmprsr or SCOPE.

### Gaps
- I found no systematic study of prompt compression on code generation beyond 2603.23527's MBPP and HumanEval output-length analysis. Accuracy on code under compression is unverified here.
- I found no primary quantitative study of hallucination rates introduced by abstractive compressors.

---

## Q4. What are open problems as of 2026?

### Takeaway
Open problems as of late 2026 include: evaluation that goes beyond the input compression ratio (information preservation, output-length effects, energy, cross-benchmark robustness); keeping instructions and structure intact when compressing; question-agnostic compression that holds up at 10x+ on dense content; reconciling compression with exact-prefix caching; and domain coverage (code, medicine, multi-turn agents).

### Cited Findings
- New evaluation axes beyond compression ratio:
  - Downstream performance, grounding and information preservation. — [arXiv:2503.19114](https://arxiv.org/abs/2503.19114)
  - Instruction survival probability and a Compression Robustness Index, plus energy (NVML) measurements. — [arXiv:2603.23527](https://arxiv.org/abs/2603.23527)
- Compression-rate controllability is unsolved for generative compressors without RL post-training, and current compressors are trained narrowly (MeetingBank-only for Cmprsr). — [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- Abstractive methods win at high compression (CR 0.1) and extractive methods track the target rate better, so the field is split on which to use. — [Cmprsr HTML](https://arxiv.org/html/2511.12281)
- Soft compression still caps at 62–73% retention at extreme ratios (500xCompressor). — [arXiv:2408.03094](https://arxiv.org/abs/2408.03094) Granularity control is a promising fix. — [arXiv:2503.19114](https://arxiv.org/abs/2503.19114)
- Caching rewards exact, append-only prefixes, which conflicts with summarization or compaction. — [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching); [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Microsoft's own research has shifted toward KV-cache-centric evaluation and serving (SCBench, Dec 2024; RetrievalAttention, Sep 2024; MInference, Jul 2024), which suggests the frontier moved from prompt text to KV state. — [README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md)

### Inferences
- Since providers cut cached reads to 0.025–0.1x of input price, the economic case for lossy prompt compression on API models has narrowed. It now matters mainly for non-repeated, large, dynamic context (RAG docs, tool outputs, logs) and for latency or context-window limits.
- A likely research direction is compression that knows about structure and instructions (protecting Ψ), with adaptive ratios per segment type. SCOPE's per-chunk ratios and keyword maintenance and LLM-DCP's key-information reward are early steps.
- Black-box users still lack a standard way to compress KV state. Provider-side caching is the only KV-level saving exposed through the API.

### Gaps
- I did not survey 2026 papers beyond 2603.23527, so there may be newer methods (e.g. agent-trajectory or tool-output compression) not covered here.
- I found no shared leaderboard for prompt compression as of Nov 2026.
