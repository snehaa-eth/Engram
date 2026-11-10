# Retrieval and Ranking State of the Art (2025 to Nov 2026): Does It Beat the Classics?

Scope: first-stage retrieval and ranking for an "AI brain" memory service. The planned classic stack is BM25, a dense bi-encoder with cosine similarity, RRF hybrid fusion, cross-encoder reranking and MMR diversity. These notes are limited to that stack. Memory-specific benchmarks such as LongMemEval and LoCoMo are out of scope and are listed under Gaps.

Verification legend: **[V]** means I read the number in the primary source (paper table, abstract or official blog). **[S]** means the number came from a search snippet or secondary summary and I did not check it against the primary table. **[U]** means unverified or conflicting.

---

## Q1. Which classic components are still best practice in 2026, which are superseded, and by how much?

### Takeaway
Nothing in the classic stack is obsolete as a *pattern*: hybrid lexical plus dense retrieval, then a reranker, is still the stack that every strong 2025 result builds on. What has changed is the models in each slot.
- **Dense embedder:** the 2023-era embedders (BGE, e5, text-embedding-3) are clearly beaten. Use an LLM-based embedder such as Qwen3-Embedding (open) or Gemini Embedding (API). The gain is about 5 to 12 points on the MMTEB mean and up to about 16 points on MMTEB retrieval.
- **Cross-encoder reranker:** the BERT-era cross-encoder is superseded by small LLM-backbone pointwise rerankers. Qwen3-Reranker-0.6B and 4B beat bge-reranker-v2-m3 by about 8 to 14 points on MMTEB-R.
- **RRF:** still a fine zero-tuning default. A tuned convex combination beats it, and RRF *hurts* when one retriever is much stronger than the other.
- **BM25:** still a necessary cheap component. It is no longer competitive on its own for semantic or reasoning queries (BRIGHT 14.8 vs 36.9 for a reasoning embedder).
- **MMR:** has no consensus successor. Budget-aware and adaptive variants exist, but none is established.

### Cited Findings

**Dense embeddings (successors to the classic bi-encoder)**
- MMTEB (multilingual) mean-task score, with the retrieval sub-score in parentheses [V]:
  - Qwen3-Embedding-8B: 70.58 (retrieval 70.88)
  - Qwen3-Embedding-4B: 69.45 (69.60)
  - Qwen3-Embedding-0.6B: 64.33 (64.64)
  - Gemini Embedding: 68.37 (67.71)
  - gte-Qwen2-7B: 62.51 (60.08)
  - multilingual-e5-large-instruct: 63.22 (57.12)
  - BGE-M3: 59.56 (54.60)
  - text-embedding-3-large: 58.93 (59.27)
  - NV-Embed-v2: 56.29 (56.72)
  — [Qwen3 Embedding tech report, arXiv 2506.05176 Table 2](https://arxiv.org/html/2506.05176v3)
- MTEB English v2 mean [V]: Qwen3-Embedding-8B 75.22, 4B 74.60, 0.6B 70.70; Gemini Embedding 73.30; NV-Embed-v2 69.81; text-embedding-3-large 66.43 — [arXiv 2506.05176 Table 3](https://arxiv.org/html/2506.05176v3)
- Qwen3-Embedding-8B ranked No. 1 on the MTEB multilingual leaderboard as of 5 June 2025 (70.58). It also scored 80.68 on MTEB Code [V]. Sizes are 0.6B, 4B and 8B, and the models are Apache-licensed open weights. — [Qwen blog](https://qwenlm.github.io/blog/qwen3-embedding); [arXiv 2506.05176](https://arxiv.org/html/2506.05176v1)
- Margin for the memory-service decision: Qwen3-Embedding-0.6B scores about the same as or above multilingual-e5-large-instruct (also 0.6B) on the multilingual mean (64.33 vs 63.22) and is about 7.5 points higher on retrieval (64.64 vs 57.12) [V, computed from Table 2]. Upgrading the embedder at the *same model size* is therefore close to free. — [arXiv 2506.05176](https://arxiv.org/html/2506.05176v3)
- Later entrants claim the top of the leaderboard but are self-reported: QZhou-Embedding (Aug 2025) claims No. 1 on MTEB and CMTEB, and LGAI-Embedding-Preview also reports leaderboard results. A 2026 guide says gemini-embedding-001 (GA mid-2025) "currently holds the top spot" on MTEB Multilingual, which conflicts with the Qwen3 tables above. The leaderboard churns every few months. [U] — [QZhou tech report](https://arxiv.org/pdf/2508.21632); [LGAI report](https://arxiv.org/pdf/2506.07438); [PremAI 2026 guide](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)
- I could not verify numbers for Voyage (voyage-3.5/4), jina-embeddings-v4/v5, NV-Embed successors or a "Gemini Embedding 2". See Gaps.

**Reasoning-intensive retrieval (BRIGHT): where the classic stack breaks**
- On the original BRIGHT paper (ICLR 2025, 1,398 queries), the best retriever scored only 24.3 nDCG@10. SFR-Embedding-Mistral scored 59.0 on MTEB retrieval but 18.3 on BRIGHT. Adding LLM reasoning to the query improved results by up to 12.2 points [V, abstract/snippet]. — [BRIGHT, arXiv 2407.12883](https://arxiv.org/abs/2407.12883)
- Reproducible BRIGHT baselines from Waterloo (Anserini/Pyserini/RankLLM, arXiv 2509.02558), overall nDCG@10 [V, from the HTML tables]:
  - BM25 (query-side variant BM25Q): 14.8
  - SPLADE-v3: 15.6
  - BGE-large-en-v1.5: 13.8
  - Diver-Retriever-4B: 29.1
  - Reason-Embed-4B: 36.9
  — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- **Fusion result from the same paper [V]:** RRF with BM25Q *helps* weak dense retrievers (SPLADE-v3 15.6 → 17.2; BGE 13.8 → 17.3). It *hurts* strong reasoning retrievers (Diver 29.1 → 28.5; Reason-Embed 36.9 → 35.9). Their normalized-score fusion ("NAF") hurts even more for Reason-Embed (31.1). — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- **Reranker result from the same paper (top-100 candidates) [V]:**
  - Over a weak first stage, LLM reranking adds a lot: SPLADE+BM25 at 17.5 rises to 29.3 with Qwen3-8B, 33.4 with gpt-oss-120b and 30.9 with ReasonRank-32B.
  - Over a strong first stage, it adds little or hurts: Reason-Embed-4B at 36.9 becomes 35.7 with Qwen3-8B (worse), 38.0 with ReasonRank-32B and 40.3 with gpt-oss-120b.
  - Implication: with a strong embedder, much of the reranker's value disappears.
  — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- Systems using LLM-based query expansion score more than 10 points higher nDCG@10 on BRIGHT than those using the original queries (citing the DIVER work) [S]. — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- ReasonIR-8B (Meta, Apr 2025) on BRIGHT [S, consistent across sources]:
  - 24.4 nDCG@10 with the original queries
  - 29.9 with GPT-4-rewritten queries
  - 36.9 when an LLM reranker is added. Sources disagree on which reranker: "QwenRerank" or zero-shot Qwen2.5-32B [U].
  — [ReasonIR, arXiv 2504.20595](https://arxiv.org/pdf/2504.20595); [HF paper page](https://huggingface.co/papers/2504.20595)
- Reason-ModernColBERT (LightOn, 150M parameters, late interaction) is reported to outperform all models up to 7B on BRIGHT. It is also reported to beat ReasonIR-8B, which was trained on the same data, by more than 2.5 nDCG@10 on the StackExchange splits. These are vendor-reported claims [S], and the license may be CC-BY-NC [U]. — [HF model card](https://huggingface.co/lightonai/Reason-ModernColBERT); [LightOn blog](https://www.lighton.ai/lighton-blogs/lighton-releases-reason-colbert)

**Learned sparse retrieval (successor to BM25)**
- BEIR-13 average nDCG@10, author-reported and collected in the Sentence-Transformers docs [S; these come from different papers and setups, so they are not strictly comparable]:
  - BM25: 45.6
  - naver/splade-v3: 51.7
  - splade-v3-distilbert: 50.0
  - splade-v3-doc (inference-free): 47.0
  - opensearch-neural-sparse-encoding-doc-v3-gte (inference-free): 54.6
  - lightonai "SlideOn": 58.7 [U on whether it is inference-free]
  — [SBERT sparse pretrained models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)
- In other words, SPLADE-v3 beats BM25 by about 6 points on BEIR, but on BRIGHT it beats BM25 by under 1 point (15.6 vs 14.8 [V]). — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- Inference-free LSR (Amazon/OpenSearch, arXiv 2411.04403): beats SPLADE-v3-Doc by 3.3 nDCG@10 on 13 BEIR datasets, with client-side latency only 1.1× that of BM25 [S]. Li-LSR (SIGIR 2025) beats SPLADE-v3-Doc by 1.8 nDCG@10 on BEIR [S]. — [arXiv 2411.04403](https://arxiv.org/pdf/2411.04403); [Li-LSR arXiv 2505.01452](https://arxiv.org/pdf/2505.01452)

**Late interaction / multi-vector**
- MUVERA (Google, NeurIPS 2024) reduces multi-vector search to single-vector maximum inner product search (MIPS) using fixed-dimensional encodings. Against PLAID across six BEIR datasets it reports [S, consistent with the abstract and the Google blog]:
  - on average 10% higher recall and 90% lower latency
  - up to 56% higher recall and up to 5.7× lower latency
  - about 32× compression with product quantization
  — [MUVERA arXiv 2405.19504](https://arxiv.org/html/2405.19504v2); [Google Research blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
- GTE-ModernColBERT-v1 has a BEIR average of about 54.7 to 54.9, vs 53.79 for ColBERT-small. The figures vary by source [S]. — [HF model card](https://huggingface.co/lightonai/GTE-ModernColBERT-v1); [mxbai-edge-colbert report arXiv 2510.14880](https://arxiv.org/pdf/2510.14880)

**Rerankers (cross-encoder vs LLM)**
- Qwen3-Reranker results [V]. Each reranks the top 100 from Qwen3-Embedding-0.6B, whose own MMTEB-R baseline is 64.64.

  | Reranker | Size | MMTEB-R | MTEB-R (English) | MLDR |
  |---|---|---|---|---|
  | Qwen3-Reranker-0.6B | 0.6B | 66.36 | 65.80 | 67.28 |
  | Qwen3-Reranker-4B | 4B | 72.74 | 69.76 | 69.97 |
  | Qwen3-Reranker-8B | 8B | 72.94 | 69.02 | 70.19 |
  | bge-reranker-v2-m3 | 0.6B | 58.36 | 57.03 | 59.51 |
  | jina-multilingual-reranker-v2-base | 0.3B | 63.73 | 58.22 | — |
  | gte-multilingual-reranker-base | 0.3B | 59.44 | — | — |

  Two important points:
  - The classic cross-encoders *lower* the score of a strong modern first stage (64.64 down to 58.36 for bge-reranker-v2-m3).
  - Qwen3-Reranker-0.6B adds only about 1.7 points over a strong embedder. The 4B model adds about 8 points.
  — [arXiv 2506.05176 Table 4](https://arxiv.org/html/2506.05176v3)
- A Naver Labs study (2024) found that "cross-encoders remain competitive against LLM-based re-rankers – in addition to being way more efficient" when reranking SPLADE results. GPT-4 was the strongest model, and open LLMs did worse [S, abstract]. — [arXiv 2403.10407](https://ar5iv.labs.arxiv.org/html/2403.10407)
- An Innsbruck study (Aug 2025) compared 22 methods and 40 variants. LLM rerankers are best on familiar queries, but their generalization to novel queries varies, and lightweight models offer comparable efficiency [S, abstract]. — [arXiv 2508.16757](https://arxiv.org/abs/2508.16757v1)
- FIRST (single-token listwise decoding) reports 54.3 average nDCG@10 on BEIR vs 53.7 for RankZephyr, with 50% lower latency [S]. A Waterloo reproduction measured a smaller latency gain of 21% to 42% [S]. — [FIRST arXiv 2406.15657](https://arxiv.org/html/2406.15657v1); [Reproduction arXiv 2411.05508](https://arxiv.org/html/2411.05508v2)
- ReasonRank (Aug 2025, Qwen2.5-7B/32B backbones) [S]:
  - 32B reaches 38.03 average nDCG@10 on BRIGHT, about 5 points above Rank-K-32B.
  - The abstract claims a leaderboard score of 40.6, which appears to come from a different retriever and window configuration [U].
  — [arXiv 2508.07050](https://arxiv.org/html/2508.07050v1)

**Contextual retrieval and late chunking (chunk-representation upgrades)**
- Anthropic Contextual Retrieval measures failure as 1 − recall@20, averaged across domains, using Gemini text-embedding-004 [V]. Failure rates:
  - baseline: 5.7%
  - contextual embeddings: 3.7% (−35%)
  - contextual embeddings + contextual BM25: 2.9% (−49%)
  - the above + reranking (Cohere; top-150 reranked to top-20): 1.9% (−67%)
  
  Other details:
  - It prepends 50 to 100 tokens of LLM-written context to each chunk.
  - One-time cost is about $1.02 per million document tokens with prompt caching, assuming 800-token chunks and 8k-token documents.
  - Caveats: results are dataset-specific, reranking adds latency, and for knowledge bases under 200k tokens you should consider putting everything in the prompt.
  — [Anthropic engineering blog](https://www.anthropic.com/engineering/contextual-retrieval)
- Late chunking (Jina, arXiv 2409.04701) embeds the whole document first and then pools per chunk, with no extra training [S]:
  - average relative nDCG@10 gain of 2.70% to 3.63% over naive chunking on BEIR (from an alphaXiv summary [U])
  - SciFact example: naive 64.20, late chunking 66.10
  - gains grow with document length
  — [arXiv 2409.04701](https://arxiv.org/abs/2409.04701); [jina-ai/late-chunking repo](https://github.com/jina-ai/late-chunking)

**Fusion: RRF vs learned/convex fusion**
- Bruch, Gai and Ingber (Pinecone; ACM TOIS) found that a tuned convex combination of normalized scores beats RRF both in-domain and out-of-domain. RRF is sensitive to its parameter k, and tuned RRF generalizes poorly. The convex combination needs only a small labeled set to tune α, and α in [0.6, 0.8] (dense weight) consistently helped [S]. — [arXiv 2210.11934](https://arxiv.org/pdf/2210.11934); [Pinecone research page](https://www.pinecone.io/research/an-analysis-of-fusion-functions-for-hybrid-retrieval/)
- On BRIGHT, both RRF and normalized fusion hurt when the dense retriever is much stronger than BM25 (see above) [V]. — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)

**Diversity: MMR successors**
- Several 2025 to 2026 papers propose alternatives to MMR, but there is no consensus winner. I did not check their benchmark numbers [S]:
  - AdaGReS: redundancy-aware greedy selection under a token budget, with adaptive relevance/redundancy weighting
  - DF-RAG: per-query adaptive λ for MMR, training-free
  - MMR combined with farthest-point sampling (arXiv 2502.09017)
  - submodular facility-location selection (a 2025 UW thesis)
  — [AdaGReS arXiv 2512.25052](https://arxiv.org/html/2512.25052v1); [DF-RAG arXiv 2601.17212](https://www.arxiv.org/pdf/2601.17212); [arXiv 2502.09017](https://arxiv.org/abs/2502.09017v1)

**Agentic / iterative search**
- Search-R1 (RL-trained interleaved search and reasoning) reports +41% (Qwen2.5-7B) and +20% (Qwen2.5-3B) relative improvement over RAG baselines on QA benchmarks [S, abstract]. Secondary summaries give different figures (about 24%, or +26% and +21%) [U]. I found no measured latency or search-count figures. — [arXiv 2503.09516](https://arxiv.org/abs/2503.09516)

### Inferences
- **Superseded:**
  - the BERT-era dense embedder, replaced by an LLM-based embedder (+5 to 16 points on MMTEB retrieval)
  - the BERT-era cross-encoder, replaced by an LLM-backbone pointwise reranker (+8 to 14 points on MMTEB-R)
- **Still best practice:** BM25 as a cheap hybrid component (with a contextual-BM25 variant), hybrid retrieval generally, and the two-stage retrieve-then-rerank pattern.
- **RRF is the weakest link of the fusion layer:** it is fine as a zero-label default, but a tuned convex combination is better, and fusion should be down-weighted or skipped when the dense retriever dominates.
- **MMR remains acceptable:** successors are early-stage. For a memory service, a token-budget-aware variant (AdaGReS-style) is the most relevant idea, but it is unproven.
- **The biggest gains for a memory corpus probably come from chunk representation, not the ranker:** contextual retrieval cut failures by 35 to 49% *before* reranking. Memory items, such as short facts extracted from conversations, often lack context, which is the problem contextual retrieval targets.
- **Reasoning retrievers and rerankers (ReasonIR, ReasonRank, Rank1) matter mainly for reasoning-heavy queries** like those in BRIGHT. Typical memory lookups ("what did the user say about X") look more like BEIR/MTEB, where standard strong embedders are sufficient.

### Gaps
- No verified current (Nov 2026) MTEB/MMTEB leaderboard snapshot. Scores for Voyage-3.5/4, jina-embeddings-v4/v5, NV-Embed successors, Gemini Embedding 2 and QZhou are unverified.
- No head-to-head study of learned (non-linear) fusion vs RRF from 2025 to 2026 was found.
- No benchmark results were verified for the MMR successors.
- Memory-specific evaluations (LongMemEval, LoCoMo) of these retrieval components were not covered in this pass.

---

## Q2. What are the latency and cost trade-offs of the new methods?

### Takeaway
The cost ladder per query, from cheapest to most expensive:
1. BM25, inference-free sparse retrieval or MUVERA-style multi-vector search: about 1 to 10 ms of retrieval.
2. A small (0.6B) embedder or reranker on GPU: tens of ms.
3. Listwise LLM rerankers (7B+): hundreds of ms to seconds.
4. Reasoning rerankers (ReasonRank, Rank1): about 0.25 to several seconds per query on multi-GPU hardware.
5. RL-trained search agents: multiple LLM calls plus multiple retrievals, i.e. seconds and many tokens.

Contextual retrieval moves its cost to ingestion time: about $1 per million document tokens, once.

### Cited Findings
- Contextual Retrieval: about $1.02 per million document tokens, one-time, using prompt caching. There is no per-query cost unless you add reranking, which "adds latency and cost" [V]. — [Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)
- ReasonRank-7B: about 0.25 to 0.5 s per query to rerank 100 passages on 4×A800 GPUs. That is reported as 2 to 2.7× faster than the pointwise reasoning reranker Rank1, because it produces one reasoning chain per window rather than one per passage [S, from Emergent Mind; not checked against the paper]. — [arXiv 2508.07050](https://arxiv.org/html/2508.07050v1); [Emergent Mind](https://www.emergentmind.com/papers/2508.07050)
- ReasonIR-8B with GPT-4 query rewriting outperforms LLM-reranker baselines while using over 200× less compute than Rank1-32B. This excludes the cost of the rewrite LLM call [S]. — [ReasonIR arXiv 2504.20595](https://arxiv.org/pdf/2504.20595)
- FIRST cuts listwise LLM reranker latency by 50% vs RankZephyr (original claim). A reproduction measured 21% to 42% [S]. — [arXiv 2406.15657](https://arxiv.org/html/2406.15657v1); [arXiv 2411.05508](https://arxiv.org/html/2411.05508v2)
- Cross-encoders are "way more efficient" than LLM rerankers while remaining competitive [S]. — [arXiv 2403.10407](https://ar5iv.labs.arxiv.org/html/2403.10407)
- MUVERA has 90% lower average latency than PLAID with equal or better recall [S]. — [Google Research](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
- Inference-free learned sparse retrieval has client-side latency of about 1.1× BM25, because the query is only tokenized and needs no neural encoding [S]. — [arXiv 2411.04403](https://arxiv.org/pdf/2411.04403)
- The cost of a cross-encoder or LLM reranker scales linearly with the number of reranked documents (typically k=100), and it is much more expensive than a bi-encoder [S]. — [ReasonIR appendix, arXiv 2504.20595](https://arxiv.org/pdf/2504.20595)
- Search-R1 and other agentic search methods have no published per-query latency in the sources I found. Secondary sources describe them only qualitatively as "slower" [U]. — [arXiv 2503.09516](https://arxiv.org/abs/2503.09516)

### Inferences
- A memory service aiming for p50 latency under about 200 ms and minimal LLM tokens per query cannot afford a listwise or reasoning LLM reranker or agentic search on every query. Those should be opt-in escalation paths: when first-stage confidence is low, or when the query is complex or multi-hop.
- A 0.6B LLM-backbone reranker (Qwen3-Reranker-0.6B class) over 20 to 50 candidates is likely affordable at tens of ms on a GPU. This is an estimate, not measured in the sources. Its gain over a strong embedder is small (+1.7 MMTEB-R), whereas the 4B model gives +8 at roughly 6× the compute.
- Spending LLM tokens at ingestion (contextualizing chunks or memories) is the best accuracy-per-query-latency trade, because it adds zero query-time tokens.

### Gaps
- I found no primary, apples-to-apples ms-per-query table comparing a cross-encoder, Qwen3-Reranker and a listwise LLM reranker on the same hardware.
- No per-query latency or token counts were found for Search-R1-style agents.
- The ReasonRank latency is from a secondary summary.

---

## Q3. What is the best stack for a low-latency, low-token memory service (Nov 2026)?

### Takeaway
Keep the classic architecture, but upgrade the models in each slot and add contextualization at ingestion. Reserve LLM or reasoning reranking and agentic multi-hop search for an escalation path.

### Cited Findings (evidence underpinning the recommended stack)
- **Ingestion:** contextualize each memory or chunk with 50 to 100 tokens of LLM-written context, embed it, and also BM25-index the contextualized text. This gave −49% retrieval failure without a reranker [V]. — [Anthropic](https://www.anthropic.com/engineering/contextual-retrieval). Late chunking is a training-free, zero-LLM-cost alternative for long documents, worth a few percent relative nDCG [S]. — [arXiv 2409.04701](https://arxiv.org/abs/2409.04701)
- **Dense retrieval:** Qwen3-Embedding-0.6B or 4B (open, self-hostable; MMTEB retrieval 64.64 or 69.60) or Gemini Embedding via API (67.71) [V]. — [arXiv 2506.05176](https://arxiv.org/html/2506.05176v3)
- **Lexical retrieval:** BM25, or an inference-free learned sparse model with about 1.1× BM25 latency and +3 BEIR points over SPLADE-v3-Doc [S]. — [arXiv 2411.04403](https://arxiv.org/pdf/2411.04403)
- **Fusion:** RRF as the default, or a convex combination with α tuned on a small labeled set (α around 0.6 to 0.8 toward dense) [S]. — [arXiv 2210.11934](https://arxiv.org/pdf/2210.11934). Down-weight lexical when the dense retriever is much stronger [V]. — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2)
- **Rerank:** Qwen3-Reranker-0.6B, or 4B if the latency budget allows, over the top 20 to 100. Avoid old BGE/Jina-v2 cross-encoders over a strong embedder, because they lowered MMTEB-R from 64.64 to 58.36 [V]. — [arXiv 2506.05176](https://arxiv.org/html/2506.05176v3)
- **Diversity:** MMR, or a budget-aware redundancy selector (AdaGReS) [S]. — [arXiv 2512.25052](https://arxiv.org/html/2512.25052v1)
- **Escalation path for hard or reasoning queries:** LLM query rewriting gives +5.5 nDCG@10 for ReasonIR on BRIGHT (24.4 to 29.9) [S]. — [arXiv 2504.20595](https://arxiv.org/pdf/2504.20595). Then use a reasoning reranker or a large LLM reranker (BRIGHT +3 to +16 depending on first-stage strength [V]). — [arXiv 2509.02558](https://arxiv.org/html/2509.02558v2). Use iterative agentic search for multi-hop questions (Search-R1 +20 to 41% relative over RAG on QA [S]). — [arXiv 2503.09516](https://arxiv.org/abs/2503.09516)
- **Multi-vector option:** a ColBERT-family model (GTE-/Reason-ModernColBERT, 150M) with MUVERA or PLAID indexing gives strong quality at small model size [S]. However, its storage overhead and the immaturity of its tooling compared with single-vector ANN search need weighing. — [MUVERA](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/); [Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT)

### Inferences
- **Recommended default path, with zero LLM tokens per query:**
  1. contextualized memories
  2. hybrid retrieval: Qwen3-Embedding (or Gemini) plus BM25 or inference-free sparse
  3. RRF or tuned convex combination fusion
  4. a small LLM-backbone reranker (0.6B) on the top 30 to 50
  5. MMR or budget-aware selection under a token budget
- **Escalation path** (triggered by low reranker confidence, multi-hop intent, or explicit "deep recall"): LLM query rewrite → larger or reasoning reranker → optional iterative search loop.
- **Should a 2026 production system switch?**
  - Embedder: yes, switch.
  - Cross-encoder: yes, switch to an LLM-backbone pointwise reranker.
  - Contextual retrieval: yes, add it.
  - BM25: no, keep it.
  - RRF: keep it, but tune or weight it.
  - MMR: keep it.
  - Reasoning rerankers and agentic search: switch to them only as an opt-in tier, not by default.

### Gaps
- No end-to-end benchmark of this exact stack on memory-style workloads (LongMemEval/LoCoMo) was found in this pass.
- Optimal reranker depth and latency for the 0.6B and 4B rerankers have not been measured here.
- Licenses for some ColBERT models are unclear.
