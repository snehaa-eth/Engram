# Vector Index and Compression State of the Art (2024 – Nov 2026): do the new methods beat HNSW + int8/binary/Matryoshka?

Research date: 2026-11-08. About 20 tool calls. Items marked **[UNVERIFIED]** come from vendor self-reports, AI-generated summaries, or background knowledge that I did not re-check against a primary source in this session.

## Q1. Is HNSW still the default in 2026, or have RaBitQ and graph+quantization hybrids replaced it? Verified recall, QPS and memory numbers

### Takeaway
HNSW, or a close graph relative such as Vamana or SVS, is still the default in-memory graph in every major engine: pgvector, Qdrant, Milvus, Redis and Elasticsearch. What changed in 2024–2026 is the layer that stores the vectors. RaBitQ-style 1–5 bit quantization, with error bounds and a rerank step, is replacing int8 and naive sign-bit binary quantization. Where data lives on object storage or SSD, IVF/centroid indexes combined with RaBitQ (Turbopuffer, LanceDB IVF_RQ, Milvus IVF_RABITQ) have overtaken graphs. So the 2026 default for RAM-resident data is "graph + RaBitQ-class quantization + full-precision rerank". Plain "HNSW + int8" is no longer the default.

### Cited Findings
**RaBitQ / Extended RaBitQ (the main compression advance)**
- RaBitQ (Gao and Long, SIGMOD 2024) turns D-dimensional vectors into D-bit strings and "guarantees a sharp theoretical error bound". The authors report it beats PQ and its variants on the accuracy/efficiency trade-off "by a clear margin". They also say PQ-style methods have no error bound and "fail disastrously on some real-world datasets". — [arXiv 2405.12497](https://arxiv.org/abs/2405.12497)
- The error bound works because a random rotation makes ⟨x̄, o⟩ concentrate around a known constant. This gives an unbiased distance estimator with an error of roughly O(1/√D) with high probability. **[UNVERIFIED: the exact O(1/√D) form comes from background knowledge and a secondary blog. The arXiv abstract does not state it.]** — [secondary blog](https://mohakchugh.is-a.dev/blog/rabitq-binary-quantization-vector-search)
- Extended RaBitQ (Gao, Gou, Xu, Yang, Long, Wong; arXiv 2409.09913, Sep 2024; SIGMOD 2025, PACMMOD 3(3) Art. 202, DOI 10.1145/3725413) generalizes RaBitQ to B bits per dimension (B = 1, 2, …). It keeps the original guarantees and "achieves the asymptotic optimality in terms of the trade-off between space and error bounds". The authors say it "consistently outperforms the state-of-the-art baselines" at equal memory. — [arXiv 2409.09913](https://arxiv.org/abs/2409.09913); [HKUST portal](https://researchportal.hkust.edu.hk/en/publications/practical-and-asymptotically-optimal-quantization-of-high-dimensi/)
- Because RaBitQ's distance estimates come with error bounds, the rerank step can be selective. Turbopuffer ANN v3 (May 5, 2026) stores 1-bit RaBitQ codes (16–32x compression) next to full-precision f16/f32 vectors on NVMe. It reranks only candidates whose distance ranges overlap, which it reports is under 1% of the narrowed set. **[vendor self-report]** — [Turbopuffer ANN v3](https://turbopuffer.com/blog/ann-v3)

**Production adoption of RaBitQ-class quantization**
- Milvus 2.6 adds an IVF_RABITQ index with 1:32 compression by default. Zilliz reports 864 QPS at 94.7% recall against 236 QPS for IVF_FLAT. Recall reaches 94.7% only with SQ8 refinement (96 B + 768 B per vector). Raw 1-bit codes give about 76.3% recall, against 95.2% for IVF_FLAT at 3072 B per vector. Refinement adds about 15 ms per query. **[vendor self-report; a DEV post repeats the same numbers, so it is not independent]** — [Zilliz blog](https://zilliz.com.cn/blog/rabitq-1bit-quantization-for-ai-infra); [DEV](https://dev.to/schiffer_kate_18420bf9766/the-practical-tradeoffs-of-extreme-vector-compression-testing-rabitq-at-scale-10h8)
- LanceDB IVF_RQ is IVF with RaBitQ codes and configurable `num_bits`. A LanceDB newsletter from summer 2026 says multi-bit IVF_RQ reaches 96.2% recall@10 with 5-bit codes and 2.6x lower p99 than unrefined IVF_PQ, "no raw vectors needed". **[vendor; I did not open the underlying post]** — [LanceDB docs](https://docs.lancedb.com/indexing/quantization); [LanceDB newsletter](https://www.lancedb.com/blog/newsletter-july-2026)
- Qdrant 1.15 added asymmetric quantization: binary codes for stored vectors and scalar-quantized queries. It also added 1.5-bit and 2-bit quantization, motivated by binary quantization's "poor handling of values close to zero". — [Qdrant 1.15](https://qdrant.tech/blog/qdrant-1.15.x/); [Qdrant quantization docs](https://qdrant.tech/documentation/manage-data/quantization/index.md)
- Zvec (Sep 2026) ships RaBitQ inside HNSW and benchmarks it against Elasticsearch BBQ, which is also a 1-bit-plus-rerank scheme. The post warns that the comparison measures end-to-end systems. **[vendor]** — [Zvec RaBitQ blog](https://zvec.org/en/blog/2026-09-08-zvec-rabitq/)
- An equal-byte-budget comparison found Milvus IVF_RABITQ and IVF_PQ "genuinely competitive". In other words, RaBitQ does not dominate PQ in every configuration. — [OraCore](https://oracore.dev/en/news/turboquant-does-not-hurt-search-quality-equal-bytes-en)

**LVQ / LeanVec (Intel SVS)**
- LeanVec (arXiv 2312.16335) reports up to 8.5x higher throughput than FAISS-IVFPQfs at 10-recall@10 = 0.90. It also reports graph builds up to 8.6x faster than FP16 and 4.9x faster than LVQ. — [arXiv 2312.16335](https://arxiv.org/pdf/2312.16335)
- In Redis, SVS-LVQ and SVS-LeanVec use 26–37% less total memory and 51–74% less index memory than HNSW, while keeping recall@10 above 0.95. They deliver up to 144% higher QPS on high-dimensional data. **[vendor: Intel/Redis]** — [Intel Redis guide](https://intel.github.io/optimization-zone/software/similarity-search/redis); [Redis blog](https://redis.io/blog/quantization-and-dimensionality-reduction-are-now-available-in-redis-query-engine.md)
- On Intel's large-scale (billion-vector) benchmark, SVS reaches 336k QPS, 5.8x the runner-up, with a 1.4x smaller memory footprint (596 GiB). HNSWlib is among the methods compared. **[vendor; the page is labelled "previous"]** — [SVS large-scale benchmarks](https://intel.github.io/ScalableVectorSearch/benchs/static/previous/large_scale_benchs.html)

**ScaNN / SOAR**
- SOAR (NeurIPS 2023) assigns each vector to a second ("spilled") cluster. A modified loss makes the second residual orthogonal to the first. Google reports state-of-the-art results in the two Big-ANN 2023 tracks where ScaNN applied. On ann-benchmarks glove-100 it reports "by far the best query speed / indexing speed trade-off… while having the smallest memory footprint". **[vendor research blog]** — [Google Research SOAR blog](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)

**GPU (CAGRA / cuVS)**
- Faiss with cuVS (Meta/NVIDIA, May 2025) builds CAGRA graphs up to 12.3x faster than CPU HNSW: 1106 s vs 90 s for 5M×1536, and 3322 s vs 519 s for 100M×96. Search latency is up to 4.7x lower. — [Meta engineering](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- The original CAGRA paper reports 33–77x higher large-batch throughput than HNSW at 90–95% recall, and 3.4–53x faster single queries at 95% recall. — [arXiv 2308.15136](https://arxiv.org/pdf/2308.15136.pdf)
- cuVS can convert a CAGRA graph into an HNSW graph for CPU serving, so a GPU can be used for builds only. — [NVIDIA blog](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs); Elastic reports up to 12x faster indexing with cuVS — [Elastic](https://www.elastic.co/search-labs/blog/elasticsearch-gpu-accelerated-vector-indexing-nvidia)

**DiskANN successors (SSD-resident)**
- PipeANN (Guo and Lu, OSDI 2025) replaces DiskANN's synchronous beam search with an asynchronous pipeline that overlaps CPU work and SSD I/O. On billion-scale data its latency is 1.14–2.02x that of in-memory Vamana and 35% of on-disk DiskANN, with no accuracy loss. The README claims about 2 ms vs 6 ms for DiskANN on SPACEV-1B with 40 GB RAM. **[the README figure is a maintainer claim]** — [USENIX OSDI'25](https://www.usenix.org/conference/osdi25/presentation/guo); [GitHub](https://github.com/thustorage/PipeANN)
- Starling improves on DiskANN in two ways: block shuffling for better data locality on disk, and an in-memory navigation graph that shortens search paths. — summary via [PipeANN-related search](https://arxiv.org/pdf/2509.25487)
- A third-party 2026 paper (Helmsman, arXiv 2606.13145) tested on 96 cores with 12 PCIe 5.0 SSDs. DiskANN and Starling "consistently fail" production average and tail latency SLAs. PipeANN had the lowest latency of the three but still missed tail SLAs at high top-k. The paper argues for clustering-based designs ("The Clustering Strikes Back"). — [arXiv 2606.13145](https://arxiv.org/pdf/2606.13145)
- Turbopuffer explains why it chose centroid indexes: each object-storage round trip costs about 100 ms, and centroid indexes need far fewer round trips than HNSW or DiskANN. ANN v3 uses a hierarchical SPFresh tree with branching factor about 100. It probes about 500 clusters of roughly 100 vectors each per machine and targets 100B+ vectors at p99 ≤ 200 ms and about 1k QPS. The post gives no recall figure. **[vendor]** — [Turbopuffer architecture](https://turbopuffer.com/docs/architecture); [ANN v3](https://turbopuffer.com/blog/ann-v3)

**BigANN NeurIPS'23**
- The official results paper (arXiv 2409.17424; NeurIPS 2025 Datasets and Benchmarks track) covers four tracks: filtered, out-of-distribution, sparse and streaming. — [arXiv 2409.17424](https://arxiv.org/html/2409.17424v1); [NeurIPS proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/63092d79154adebd7305dfd498cbff70-Abstract-Datasets_and_Benchmarks_Track.html)
- Track results: the streaming track was won by PyANNS, which used DiskANN's streaming support with 8-bit scalar quantization, after early results were corrected for evaluation errors. ParlayANN was a strong performer in the filtered track. **[UNVERIFIED: these come from AI-generated summaries, not the paper's tables]** — [AlphaXiv overview](https://alphaxiv.org/overview/2409.17424v1)

### Inferences
- The 2024–26 gains come mostly from the vector representation (bits per dimension and how reranking is done) and the storage layout, not from new graph topologies. HNSW's search algorithm is mature. Compared with naive binary quantization, RaBitQ-class codes buy higher recall at the same number of bits and a principled rule for how many candidates to rerank, so less oversampling guesswork. For example, the pgvector binary-quantization guide suggests 80x oversampling.
- Int8 scalar quantization is still a safe, simple baseline (4x compression with negligible recall loss). The best 2026 option is 1–4 bit RaBitQ for traversal plus an int8/fp16/fp32 rerank.
- Matryoshka truncation and quantization stack, since they compress different axes (dimensions vs bits). No primary source found here benchmarks Matryoshka together with RaBitQ.

### Gaps
- I found no current ann-benchmarks or VectorDBBench leaderboard snapshot with verified numbers. VectorDBBench was not fetched.
- I did not find primary sources for RoarGraph (VLDB 2024), τ-MNG, NSG updates, LM-DiskANN or Filtered-DiskANN numbers in this session. Searches returned nothing usable for RoarGraph.
- I did not see the RaBitQ papers' own head-to-head comparison against HNSW+SQ8. Their abstracts compare against PQ variants.
- Elasticsearch BBQ internals and numbers were not verified.

## Q2. Best approach for filtered search with many tenants and ACLs

### Takeaway
Treat tenancy and ACLs as two different problems. For tenants, partition physically: a tenant-keyed shard, partition or namespace, so each tenant's search runs over that tenant's vectors only. Use tiered placement, with small tenants in a shared payload-indexed segment and large tenants promoted to dedicated shards. For ACLs, which act as arbitrary low-selectivity filters inside a tenant, use filter-aware graph traversal (ACORN-style two-hop expansion) or iterative/relaxed scans. Fall back to exact brute-force search when the filtered set is small.

### Cited Findings
- ACORN (Patel, Kraft, Guestrin, Zaharia; SIGMOD 2024) builds a denser HNSW without depending on the predicate and searches predicate subgraphs. It reports 2–10x higher QPS on earlier benchmarks and over 1,000x on a 25M-vector dataset. Earlier methods such as Filtered-DiskANN support only equality predicates over about 1,000 labels. — [arXiv 2403.04871](https://arxiv.org/abs/2403.04871v1)
- Qdrant 1.16 (Nov 19, 2025) adds ACORN-1 as a per-query option with no index changes. On 5M×96d with about 4% filter selectivity, ef=64 with ACORN reaches 97.2% accuracy at 13.86 ms. Without ACORN, ef=64 gives 53.3% at 1.25 ms and ef=512 gives 71.1% at 3.89 ms. Qdrant recommends ACORN only for multiple low-selectivity filters. — [Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/index.md)
- Qdrant 1.16 tiered multitenancy keeps small tenants on a shared fallback shard and lets large tenants have dedicated shards. Requests are routed automatically, and a tenant is promoted through shard transfer while reads and writes continue. Every request must carry a shard key selector. — [Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/index.md)
- pgvector 0.8.0 (2024-10-30) added iterative index scans (`hnsw.iterative_scan`, `ivfflat.iterative_scan`). They prevent "overfiltering", where too few rows are returned once a WHERE filter is applied. The release also improved cost estimation when filtering. — [pgvector CHANGELOG](https://raw.githubusercontent.com/pgvector/pgvector/master/CHANGELOG.md); [pgsql-announce](https://postgrespro.com/list/id/173133472572.709.14743591049493026466@wrigleys.postgresql.org); [Nile](https://www.thenile.dev/blog/pgvector-080)
- LanceDB recommends IVF_RQ or IVF_PQ over its HNSW-backed variants (IVF_HNSW_*) for filtered workloads, because the HNSW variants "can show higher latency variance". — [LanceDB vector index docs](https://docs.lancedb.com/indexing/vector-index.md)
- Turbopuffer gives each namespace its own object-storage prefix, and any stateless node can serve any namespace. A namespace-per-tenant design therefore needs no tenant filter. — [Turbopuffer architecture](https://turbopuffer.com/docs/architecture)
- An implementation note (vicinity crate, not the paper) says ACORN's two-hop expansion works best when about 2–20% of nodes pass the filter. **[third-party guidance]** — [docs.rs vicinity](https://docs.rs/crate/vicinity/0.8.1/source/src/hnsw/filtered.rs)

### Inferences
- Pre-filtering with a tenant_id payload filter inside one global HNSW is the weakest pattern at scale. Very selective filters disconnect the graph, so recall drops or ef has to be pushed very high. Partitioning by tenant avoids that entirely.
- For tenants with fewer than about 10k–50k vectors, exact search over int8 or binary codes is often cheaper and gives 100% recall. This is an inference: I found no source benchmarking the crossover point.
- ACLs that change per user should be stored as payload or bitmap filters and applied with ACORN or iterative scans. Embedding ACLs in the index structure would force rebuilds whenever permissions change.

### Gaps
- I found no independent 2025–26 benchmark that compares multi-tenant strategies (namespace-per-tenant, partition-per-tenant, payload filter) at 10k–1M tenants.
- I did not verify Milvus partition-key or Weaviate multi-tenancy limits.

## Q3. Best approach for frequently updated (write-heavy) memory

### Takeaway
Graph indexes handle inserts well but deletes and updates poorly. Recall degrades unless deletes are consolidated in batches (FreshDiskANN), handled in place (IP-DiskANN, 2025), or repaired by vacuuming. Centroid/IVF indexes with incremental cluster split and merge (SPFresh, as used by Turbopuffer) and a WAL-plus-memtable front layer suit write-heavy object-storage designs best. For a memory service, the usual pattern is an LSM-style buffer: a fresh segment searched by brute force or a small HNSW, merged into a compacted, quantized main index.

### Cited Findings
- IP-DiskANN (arXiv 2502.13826, Feb 2025) is "the first algorithm to avoid batch consolidation by efficiently processing each insertion and deletion in-place". Its recall stays stable over long update streams, and its update cost is "significantly lower than for the HNSW algorithm" at comparable or better recall. It also beats FreshDiskANN's batch consolidation on query throughput and update speed. — [arXiv 2502.13826](https://www.arxiv.org/pdf/2502.13826)
- One summary gives sliding-window figures on 10M points: Recall@10 of about 94.8% for IP-DiskANN, 94.4% for FreshDiskANN and 91.8% for HNSW. **[UNVERIFIED: from an auto-generated summary]** — [emergentmind](https://api.emergentmind.com/topics/proximity-graph-based-ann-indices)
- Turbopuffer and SPFresh update clusters incrementally as vectors change, so the index is never fully rebuilt. Writes land in a WAL on object storage and are durable once acknowledged. — [Turbopuffer architecture](https://turbopuffer.com/docs/architecture)
- The BigANN'23 streaming track was won by DiskANN streaming with int8 SQ (PyANNS). **[UNVERIFIED: secondary summary]** — [AlphaXiv](https://alphaxiv.org/overview/2409.17424v1)
- pgvector's HNSW update path is still not fully mature. Version 0.8.3 (2026-06-17) fixed "possible index corruption with HNSW vacuuming". Version 0.9.0 (unreleased as of Nov 2026) adds async I/O for HNSW. — [pgvector CHANGELOG](https://raw.githubusercontent.com/pgvector/pgvector/master/CHANGELOG.md)
- Qdrant 1.16 adds conditional updates, where an update applies only if a filter on a field such as `version` matches. This gives optimistic concurrency, useful for re-embedding migrations. — [Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/index.md)
- Intel SVS publishes separate benchmarks for dynamic indexing. — [SVS dynamic benchmarks](https://intel.github.io/ScalableVectorSearch/benchs/dynamic.html) (not examined in detail)

### Inferences
- Make memory writes append-only (new version plus tombstone) and handle deletion by filtering tombstones at query time plus periodic compaction. This is the approach that works with every index family.
- RaBitQ codes need no trained codebook (only a random rotation plus IVF centroids). That makes them friendlier to streaming than PQ, whose codebooks drift as data changes. This is inferred from the method design, not from a benchmark.

### Gaps
- I did not verify a 2025–26 study that measures how much HNSW recall drops under heavy deletes.
- LM-DiskANN and Fresh-DiskANN numbers were not re-checked.

## Q4. Practical recommendation for 10k – 100M vectors per tenant (low-latency, multi-tenant memory service)

### Takeaway
Use tiers by tenant size. HNSW still covers most of the range, but pair it with RaBitQ/BBQ-class or 2-bit quantization and a full-precision rerank. Switch to IVF/centroid + RaBitQ on SSD or object storage once a tenant's vectors no longer fit cheaply in RAM.

### Cited Findings (supporting facts; the recommendation itself is synthesis)
- pgvector timeline: halfvec, bit indexing and `binary_quantize` arrived in 0.7.0 (2024-04-29). Parallel HNSW builds arrived in 0.6.0 (2024-01-29) and iterative scans in 0.8.0 (2024-10-30). The latest release is 0.8.7 (2026-10-01). pgvector has no native RaBitQ, and binary quantization requires manual reranking. — [pgvector CHANGELOG](https://raw.githubusercontent.com/pgvector/pgvector/master/CHANGELOG.md)
- A third-party pgvector guide suggests about 80x oversampling for binary-quantized two-stage search (800 candidates for top-10). — [Timescale skill guide](https://smithery.ai/skills/timescale/pgvector-semantic-search)
- Qdrant inline storage (1.16) packs quantized vectors and original vectors into each HNSW node. On 1M LAION 512d with 2-bit quantization and fp16, low-RAM mode (430 MiB) gave 211 QPS at 86.9% accuracy, against 20 QPS at 53.3% without inline storage. — [Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/index.md)
- Turbopuffer reports cold p50 of 874 ms and warm p50 of 14 ms at 1M docs. A third party reports 343 ms cold, so the figures conflict. — [Turbopuffer architecture](https://turbopuffer.com/docs/architecture); [modern-datatools](https://www.modern-datatools.com/tools/turbopuffer)

### Inferences (recommendation)
- **Tier 0, under about 50k vectors per tenant:** no ANN index. Run exact scans over an fp16 or int8 copy, or 1-bit codes with rerank, inside the tenant's partition. Recall is 100%, latency is sub-millisecond to a few ms, and updates are trivial.
- **Tier 1, 50k to about 5M per tenant:** in-RAM HNSW (Qdrant, pgvector HNSW on halfvec, or Milvus HNSW) on a tenant-scoped shard or partition. Quantize with int8, or with 1.5/2-bit or RaBitQ when memory is tight, and rerank in fp16. Use ACORN or iterative scans for ACL filters.
- **Tier 2, 5M to 100M per tenant:** quantized graph with inline or on-SSD vectors (Qdrant inline storage, DiskANN/PipeANN class), or IVF + RaBitQ (Milvus IVF_RABITQ, LanceDB IVF_RQ with 4–5 bits, Turbopuffer). Give each such tenant a dedicated shard.
- **Write-heavy memory:** use a fresh/memtable segment with exact search, merged asynchronously into the quantized main index. Delete with tombstones and compaction.
- **Matryoshka embeddings:** truncate to 256–512 dimensions for the first pass, then rerank at full dimension. This stacks with bit quantization. It is untested in the sources gathered here, so validate it on your own data.
- **Postgres-centric stack:** pgvector at 0.8.x is viable up to about 10M vectors per tenant with halfvec + HNSW + iterative scans + partitioning by tenant. Beyond that it falls behind dedicated engines on quantization (no RaBitQ) and on-disk efficiency. This is an inference from the changelog's feature set, with no benchmark source.

### Gaps
- I have no verified, apples-to-apples p99 numbers across pgvector, Qdrant, Milvus, LanceDB and Turbopuffer for multi-tenant workloads. VectorDBBench results were not retrieved.
- I found no primary source quantifying the exact-search vs HNSW crossover point per tenant size.
