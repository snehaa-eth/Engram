# Algorithms and Data Structures for Fast, Low-Cost Retrieval in LLM Systems

Notes date: Nov 2026. Labels used below: **[V]** means checked this session against the primary source (arXiv abstract or HTML, NeurIPS proceedings, or the official blog). **[T]** means taken from training knowledge of the primary paper and not re-fetched this session, so treat it as unverified. **[Inf]** means my own arithmetic or inference.

## Q1: Time/space complexity and practical recall@k vs latency trade-offs for each method

### Takeaway
Vector ANN is a solved engineering problem at almost every scale:
- **HNSW** gives roughly logarithmic query time and >95% recall in RAM, but it uses a lot of memory.
- **IVF-PQ, ScaNN and binary/int8 quantization** trade 4–32x memory for a few points of recall. Rescoring with full-precision vectors recovers most of that loss.
- **DiskANN** moves billion-scale search onto an SSD.

Late interaction (ColBERTv2/PLAID) and cross-encoder reranking buy quality at 10–100x the latency of a single-vector lookup. Graph-RAG methods move the cost into LLM-driven indexing: tens of millions of tokens for a corpus of about 10k passages.

### Cited Findings

**Dense ANN**
- **HNSW** (Malkov & Yashunin, arXiv 1603.09320; later published in IEEE TPAMI [T]):
  - It is a multi-layer proximity graph. Search starts at the top layer and descends, which gives "logarithmic complexity scaling". A neighbor-selection heuristic improves performance at high recall and on clustered data. [V] — [arXiv 1603.09320](https://arxiv.org/abs/1603.09320)
  - Memory: the full float vectors plus about M×2 links per node on layer 0 (roughly 4·2M bytes per vector) [T]. Build cost is about O(N log N) [T].
- **Product quantization** (Jégou, Douze & Schmid, IEEE TPAMI 2011) [T]:
  - A vector is split into m subvectors, and each is quantized to one of 256 centroids, so it is stored in m bytes. For example, 768-d float32 is 3,072 B; PQ with m=64 stores it in 64 B, a 48x reduction [Inf].
  - Distances are computed asymmetrically (ADC) with lookup tables.
  - IVF-ADC adds a coarse k-means inverted file. Each query scans only `nprobe` of the `nlist` cells, so the cost is about O(nprobe·N/nlist·m).
- **FAISS:** I tried to verify its index-selection guidance from the [FAISS wiki "Guidelines to choose an index"](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), but the page returned HTTP 504. The rules of thumb below are recalled guidance and remain unverified [T]:
  - Use Flat (exact search) for small N, or when exact results are required.
  - Use HNSW when memory is not a constraint.
  - Use IVF with about 4√N to 16√N centroids, plus PQ/OPQ, when memory is limited.
  - IVF needs on the order of 30·nlist to 256·nlist training vectors.
- **ScaNN, anisotropic vector quantization** (Guo et al., arXiv 1908.10396; ICML 2020 [T]):
  - Reconstruction-error quantization is a poor fit for maximum inner product search (MIPS). The anisotropic loss instead "more greatly penalizes the parallel component of a datapoint's residual" than the orthogonal component, and the paper reports state-of-the-art results on public ANN benchmarks. [V] — [arXiv 1908.10396](https://arxiv.org/abs/1908.10396)
- **DiskANN/Vamana** (Subramanya et al., NeurIPS 2019) [V] — [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html):
  - Indexes SIFT1B (1 billion points) on one workstation with 64 GB RAM and an inexpensive SSD.
  - Serves more than 5,000 QPS on 16 cores with under 3 ms mean latency at 95%+ 1-recall@1.
  - At a similar memory footprint, FAISS and IVFOADC+G+P plateau at about 50% 1-recall@1.
  - At high recall it serves 5–10x more points per node than HNSW or NSG.
- **Binary and int8 embedding quantization** (Hugging Face / Sentence-Transformers, measured on MTEB Retrieval with mxbai-embed-large-v1) [V] — [HF blog](https://huggingface.co/blog/embedding-quantization):

  | Format | Memory reduction | Mean speedup (range) | Performance retained |
  |---|---|---|---|
  | Binary | 32x | 24.76x (15.05–45.8x) | ~92.5% without rescoring; ~96.45% with rescoring |
  | Int8 | 4x | 3.66x (2.99–4.8x) | ~97% without rescoring; ~99% with rescoring at a 4–5x multiplier |

  - Binary rescoring: Hamming search over the binary index fetches `rescore_multiplier × k` candidates. The float32 query is then dot-producted against those candidates.
  - Demo on 41M Wikipedia texts: the binary index takes about 5.2 GB, versus about 200 GB for float32.
- **Matryoshka Representation Learning** (Kusupati et al., arXiv 2205.13147; NeurIPS 2022 [T]) [V] — [arXiv 2205.13147](https://arxiv.org/abs/2205.13147):
  - Embeddings can be truncated to a nested prefix. Embeddings are up to 14x smaller at the same ImageNet-1K accuracy.
  - Large-scale retrieval is up to 14x faster in practice.
  - This enables coarse-to-fine "adaptive retrieval": shortlist with low-dimensional prefixes, then rerank with full dimensions.
- **ANN-Benchmarks** (ann-benchmarks.com; Aumüller et al., Information Systems 2020) [T]:
  - Graph methods (HNSW variants, NGT, and later Glass/SymphonyQG-style implementations) and ScaNN occupy the recall/QPS Pareto frontier on glove/sift/deep at 1M scale.
  - At recall about 0.9–0.95, these reach thousands to tens of thousands of QPS per core.
  - I did not re-fetch these numbers this session [T].

**Sparse, hybrid and late interaction**
- **BM25:** query cost scales with the posting-list lengths of the query terms. Dynamic pruning (WAND, BlockMax-WAND) gives millisecond latency on millions of documents [T].
- **SPLADE** (Formal et al., SIGIR 2021) [T]: learned sparse expansion that runs on inverted indexes. It is slower than BM25 because the expanded postings are denser [T].
- **Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009) [T]:
  - score(d) = Σ 1/(k + rank_i(d)), with k = 60.
  - It is parameter-light and cost-free, since it fuses rank lists only. It is the default in most hybrid stacks.
- **ColBERT/ColBERTv2** (SIGIR 2020; NAACL 2022) [T]:
  - Stores one vector per token and scores with MaxSim.
  - ColBERTv2's residual compression cuts the index footprint about 6–10x versus ColBERT v1 [T]. Storage still scales with the number of tokens, not passages.
- **PLAID** (Santhanam et al., arXiv 2205.09707) [V] — [arXiv 2205.09707](https://arxiv.org/abs/2205.09707):
  - Up to 7x faster than vanilla ColBERTv2 on GPU and up to 45x on CPU.
  - Latency is tens of ms on GPU and tens to a few hundred ms on CPU, evaluated at up to 140M passages.
- **Cross-encoder rerankers** [T/Inf]: one transformer forward pass per (query, doc) pair. Reranking the top 100 with a roughly 100M–600M-parameter model costs about 100 forward passes, around 50–500 ms on GPU and seconds on a laptop CPU. That is why rerank depth is usually 20–100. I did not find a primary-source benchmark this session.

**Graph and tree RAG: indexing cost vs query cost**
- **GraphRAG** (Edge et al., arXiv 2404.16130) [V] — [arXiv HTML](https://arxiv.org/html/2404.16130):
  - Pipeline: an LLM extracts an entity graph, then Leiden communities are built, then community summaries are generated. At query time a map-reduce runs over the summaries.
  - Setup: 600-token chunks with 100-token overlap.
  - Indexing the Podcast set (1,669 chunks, about 1M tokens) took 281 minutes with gpt-4-turbo and produced 8,564 nodes and 20,691 edges. The News set has 3,197 chunks and about 1.7M tokens.
  - On global sensemaking questions, it beats vector RAG with comprehensiveness win rates of 72–83% (Podcast) and 72–80% (News), and diversity win rates of 75–82% and 62–71%.
  - Root-level (C0) summaries use about 26.7k/39.8k tokens, versus about 1.01M/1.71M for summarizing the source text directly. That is 9–43x fewer tokens (over 97% fewer).
- **LightRAG** (Guo et al., arXiv 2410.05779) [V] — [arXiv HTML](https://arxiv.org/html/2410.05779v3):
  - Retrieval cost on the Legal set: GraphRAG uses about 610 communities × 1,000 tokens, about 610,000 tokens over "hundreds of API calls". LightRAG uses fewer than 100 tokens and a single API call.
  - Incremental updates in GraphRAG rebuild community reports, about 1,399 × 2 × 5,000 tokens. LightRAG's updates add only extraction overhead.
  - Self-reported overall win rates vs GraphRAG are 49.6–54.8%, a narrow margin. Against NaiveRAG they are 60–84.8%.
- **HippoRAG 2** (Gutiérrez et al., arXiv 2502.14802, ICML 2025) [V] — [arXiv HTML](https://arxiv.org/html/2502.14802). Its Table 12 (MuSiQue corpus, 11,656 passages) is an independent cost comparison:

  | Method | Input tokens | Output tokens | Indexing time | QA time per query |
  |---|---|---|---|---|
  | NV-Embed-v2 (dense) | none | none | 12.1 min | 0.3 s |
  | RAPTOR | 1.7M | 0.2M | 100.5 min | 0.6 s |
  | HippoRAG | 9.2M | 3.0M | 57.5 min | 0.9 s |
  | HippoRAG 2 | 9.2M | 3.0M | 99.5 min | 1.2 s |
  | LightRAG | 68.5M | 18.3M | 235 min | 13.3 s |
  | GraphRAG | 115.5M | 36.1M | 277 min | 10.7 s |

- **RAPTOR** (Sarthi et al., arXiv 2401.18059; ICLR 2024 [T]) [V] — [arXiv 2401.18059](https://arxiv.org/abs/2401.18059):
  - Recursively clusters chunks and summarizes the clusters into a tree. Retrieval runs over the collapsed tree.
  - With GPT-4 it gives +20% absolute accuracy on QuALITY.

### Inferences
- [Inf] Each step down the quantization ladder (float32 → int8 → binary, or full → Matryoshka-truncated) multiplies memory savings. Each step costs about 1–8% of retrieval quality, and roughly half or more of that loss comes back by rescoring 4–10x oversampled candidates with higher-precision vectors.
- [Inf] The cost hierarchy per query is roughly: BM25 or HNSW (sub-ms to a few ms) < PLAID (tens to hundreds of ms on CPU) ≈ cross-encoder top-50 (hundreds of ms) < GraphRAG global map-reduce (seconds and hundreds of thousands of tokens).

### Gaps
- I could not fetch the FAISS wiki (HTTP 504), so the IVF/HNSW parameter rules remain [T].
- I did not fetch current ANN-Benchmarks plots, so I have no specific QPS@recall numbers.
- I found no primary benchmark of cross-encoder latency.
- I did not re-verify SPLADE or ColBERTv2 space numbers this session.

## Q2: At personal scale (10k–10M chunks on a laptop), which index choices are optimal?

### Takeaway
Up to about 1M chunks, exact brute force over int8 or float vectors is fast enough for interactive use, simplest, and has 100% recall. HNSW, or binary quantization with rescoring, pays off from about 1M to 10M chunks. DiskANN or IVF-PQ matter only when the vectors don't fit in RAM. Hybrid BM25+dense with RRF, plus a small reranker, usually adds more quality than any ANN tuning.

### Cited Findings
- Binary quantization at 41M documents needs about 5.2 GB, versus about 200 GB for float32. Rescoring retains about 96% of quality. — [HF blog](https://huggingface.co/blog/embedding-quantization)
- DiskANN serves 1B points from 64 GB RAM plus an SSD at under 3 ms latency. A 10M-point SSD index on a laptop is therefore well within its design envelope. — [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- Matryoshka prefixes enable cheap shortlisting followed by full-dimension rerank, with up to 14x real-world speedup. — [arXiv 2205.13147](https://arxiv.org/abs/2205.13147)
- PLAID's CPU latency is "tens to a few hundred ms" at large scale, which makes late interaction usable on a laptop at personal scale. — [arXiv 2205.09707](https://arxiv.org/abs/2205.09707)

### Inferences
Back-of-envelope figures [Inf], assuming 768-d embeddings and a laptop with about 100 GB/s memory bandwidth (Apple M-series class). Brute force is memory-bandwidth-bound.

| Corpus size | Float32 size | Brute-force latency |
|---|---|---|
| 10k chunks | 30 MB | ~0.3 ms |
| 100k chunks | 300 MB | ~3 ms |
| 1M chunks | 3 GB | ~30 ms with float32; ~8 ms with int8; ~1 ms on 96 MB of binary codes with popcount |
| 10M chunks | 30 GB | ~300 ms; does not fit in RAM on most laptops |

Recommended configuration by corpus size [Inf]:

| Corpus size | Recommendation |
|---|---|
| Up to ~200k–1M | Flat index (exact). There is no reason to accept ANN recall loss. |
| 1M–10M | HNSW over int8 vectors (about 1 KB per vector plus links), or binary/Matryoshka shortlisting with float rescoring. |
| ~10M with memory limits | IVF-PQ, or DiskANN on SSD. |

- Embedding throughput for ingestion (for example, about 2 hours to embed 10M chunks on a laptop GPU at roughly 1k chunks/s [Inf, unverified]) is likely a bigger bottleneck than the index.

### Gaps
- I have no measured laptop benchmark from a primary source. The table above is arithmetic, not measurement.
- I have no verified FAISS-official threshold for "use Flat below N".

## Q3: Graph RAG vs vector RAG: when is the extra indexing cost worth it? Verifying HippoRAG's "10–30x cheaper / 6–13x faster than IRCoT"

### Takeaway
HippoRAG's claim is accurate but narrow:
- **What it compares:** only online retrieval for 1,000 queries against iterative IRCoT, with GPT-3.5.
- **What it leaves out:** HippoRAG's indexing is about 10x slower and costs about $15 more per 10k passages than dense retrieval.

Graph structure is worth it for multi-hop or associative questions (2Wiki recall gains of 14–30 points) and for corpus-wide "global" sensemaking (GraphRAG). It is not worth it for single-hop factual lookup, where GraphRAG and especially LightRAG underperform plain dense RAG in independent tests.

### Cited Findings
- **HippoRAG abstract (NeurIPS 2024):** up to 20% better than state of the art on multi-hop QA. Single-step retrieval is "comparable or better" than IRCoT while being "10-30 times cheaper and 6-13 times faster". Note that a rendered section of the paper says 10–20x, so the paper is internally inconsistent. — [arXiv 2405.14831](https://arxiv.org/abs/2405.14831), [HTML](https://arxiv.org/html/2405.14831)
- **What exactly is measured (Table 17):** online retrieval for 1,000 queries with GPT-3.5-turbo. — [arXiv HTML](https://arxiv.org/html/2405.14831)

  | Method | API cost | Time |
  |---|---|---|
  | ColBERTv2 | $0 | 1 min |
  | IRCoT | $1–3 | 20–40 min |
  | HippoRAG | $0.1 | 3 min |

  The ratios therefore come from $1–3 vs $0.1 and 20–40 min vs 3 min. Speed depends on IRCoT's 2–4 retrieval rounds.
- **Offline indexing per 10k passages (Table 18):** — [arXiv HTML](https://arxiv.org/html/2405.14831)
  - HippoRAG: $15 and 60 min with GPT-3.5-turbo-1106; $8 with -0125.
  - Open models: 120 min with Llama-3.1-8B and 250 min with Llama-3.1-70B (about 4 h on 4×H100).
  - ColBERTv2 or IRCoT: $0 and 7 min.
  - The paper itself says indexing is "around 10 times slower and $15 more expensive".
- **HippoRAG retrieval quality (R@2/R@5):** — [arXiv HTML](https://arxiv.org/html/2405.14831)
  - 2Wiki: 71.5/89.5 vs ColBERTv2 59.2/68.2.
  - MuSiQue: 41.0/52.1 vs 37.9/49.2.
  - HotpotQA: 59.0/76.2 vs ColBERTv2 64.7/79.3, so it loses on the dataset with lower knowledge-integration demands.
- **HippoRAG 2 (ICML 2025):** — [arXiv HTML](https://arxiv.org/html/2502.14802)
  - Recall@5 vs NV-Embed-v2: 2Wiki 90.4 vs 76.5; MuSiQue 74.7 vs 69.7; NQ 78.0 vs 75.4.
  - QA F1: 2Wiki 71.0 vs 61.5; NQ 63.3 vs 61.9.
  - The abstract claims a 7% gain on associative memory tasks.
- **Same paper, F1 of other graph methods:** GraphRAG trails NV-Embed-v2 on every dataset (for example NQ 46.9 vs 61.9). LightRAG collapses (average 6.6 vs 57.0). The authors attribute this to LLM-generated noise. — [arXiv HTML](https://arxiv.org/html/2502.14802)
- **Independent systematic comparison** (Han et al., arXiv 2502.11371, revised 2026): with controlled LLMs and embeddings, RAG wins on single-hop and detail questions. GraphRAG wins on multi-hop and reasoning questions and produces more diverse summaries. Adaptive Selection/Integration hybrids give consistent gains. I read only the abstract and summary. — [arXiv 2502.11371](https://arxiv.org/abs/2502.11371)
- **GraphRAG's own evidence** is for global sensemaking over corpora of about 1M tokens (72–83% comprehensiveness win rate). It is not evidence for local factual QA. — [arXiv HTML](https://arxiv.org/html/2404.16130)

### Inferences
- [Inf] Break-even: HippoRAG's extra indexing (about $8–15 per 10k passages) is paid back against IRCoT after roughly 5k–15k queries, given savings of $1–3 per 1,000 queries. Against plain single-step dense retrieval, it is never paid back on cost, only on multi-hop quality.
- [Inf] For a personal "AI brain" with a frequently updated corpus:
  - GraphRAG-style community summarization is the worst fit, since updates trigger expensive rebuilds.
  - HippoRAG/HippoRAG 2 (OpenIE triples, a passage-entity graph and PPR) is the most cost-effective graph option, at about 8x fewer indexing tokens than LightRAG and about 12x fewer than GraphRAG.
  - The most robust default is dense+BM25 hybrid retrieval, with graph retrieval routed only for multi-hop queries.

### Gaps
- I have no verified absolute dollar figures for GraphRAG indexing beyond the tokens and minutes reported above.
- I did not extract the detailed metrics in arXiv 2502.11371.
- I found no primary evaluation of KG path-finding methods (Think-on-Graph, G-Retriever) this session.

## Q4: Hierarchical / coarse-to-fine retrieval and "optimized paths"; open problems as of 2026

### Takeaway
Coarse-to-fine retrieval shows up at every layer:
- **Within a vector index:** HNSW's layer descent, IVF cell routing, Matryoshka or binary shortlisting followed by rescoring, and PLAID's centroid pruning.
- **At the document level:** RAPTOR's summary tree and GraphRAG's community hierarchy.
- **At the query level:** decomposition and iterative retrieval (IRCoT).

Each layer trades recall at the coarse stage for cost. The main open problems are adaptive routing and noise from LLM-built indexes.

### Cited Findings
- **Within-index coarse-to-fine:** — [arXiv 1603.09320](https://arxiv.org/abs/1603.09320), [arXiv 2205.13147](https://arxiv.org/abs/2205.13147), [HF blog](https://huggingface.co/blog/embedding-quantization), [arXiv 2205.09707](https://arxiv.org/abs/2205.09707)
  - HNSW descends layers to find an entry point.
  - Matryoshka uses adaptive shortlist-then-rerank.
  - Binary and int8 indexes rescore oversampled candidates.
  - PLAID prunes with centroid interaction (7x/45x speedups).
- **RAPTOR summary tree:** +20% absolute on QuALITY. Its indexing costs 1.7M input tokens on an 11.7k-passage corpus, the cheapest LLM-built structure in HippoRAG 2's comparison. — [arXiv 2401.18059](https://arxiv.org/abs/2401.18059), [arXiv HTML 2502.14802](https://arxiv.org/html/2502.14802)
- **GraphRAG community hierarchy:** C0 root summaries answer global questions with 9–43x fewer tokens than summarizing the source text. — [arXiv HTML 2404.16130](https://arxiv.org/html/2404.16130)
- **Query decomposition and iteration:** IRCoT's 2–4 retrieval rounds cost 6–13x the time of single-step PPR retrieval. — [arXiv HTML 2405.14831](https://arxiv.org/html/2405.14831)
- **Adaptive routing between RAG and GraphRAG** gives consistent improvements. — [arXiv 2502.11371](https://arxiv.org/abs/2502.11371)

### Inferences (open problems as of 2026)
- [Inf] Indexing noise and verification: LLM-extracted graphs inject errors, as LightRAG's F1 collapse shows. There is no standard way to validate or repair triples cheaply.
- [Inf] Incremental maintenance: community summaries and RAPTOR trees are not cheaply updatable. LightRAG's incremental update is a partial answer. Deletes and edits remain poorly handled across these methods. Filtered and streaming ANN, with deletes in HNSW, is a related index-level issue.
- [Inf] Query routing: deciding per query whether to use dense, sparse, graph or a global map-reduce, at negligible cost, is still heuristic.
- [Inf] Evaluation bias: LLM-as-judge win rates (GraphRAG, LightRAG) versus exact-match F1 (HippoRAG 2) give contradictory pictures. arXiv 2502.11371 explicitly discusses evaluation biases.
- [Inf] Multi-vector cost: late-interaction storage scales with token count, and compressing it further without recall loss is still active work.
- [Inf] Long-context LLMs as a competitor: at personal scale, "retrieve less, read more" may beat elaborate indexes. I found no primary 2026 study.

### Gaps
- I did not find a 2026-dated survey or benchmark of hierarchical or routed retrieval.
- I did not verify claims about agentic or learned routers, or specific KG multi-hop path-finding numbers, this session.
