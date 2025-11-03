# Homework 3
### Congratulations on completing two homeworks where you created and applied [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) to a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from [Ollama](https://ollama.com/search) using [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus). The final homework requires students to build a scalable [GraphRAG](https://microsoft.github.io/graphrag/) pipeline that streams indexed document chunks through [Apache Flink](https://flink.apache.org/), extracts and fuses relation candidates with an Ollama model and rule pass, constructs versioned concept and chunk nodes with typed edges, and atomically upserts the resulting knowledge graph into [Neo4j](https://neo4j.com/) for [REST](https://en.wikipedia.org/wiki/REST)-based dependency queries and inconsistency detection.

Much of the background information is based on the books [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription and books on [A Simple Guide to Retrieval Augmented Generation by Abhinav Kimothi](https://www.manning.com/books/a-simple-guide-to-retrieval-augmented-generation) and [Essential GraphRAG by Tomaž Bratanič and Oskar Hane](https://www.manning.com/books/essential-graphrag). A majority of images in this and other homework descriptions are used from these books.

#### The goal of this homework is for students to gain experience with solving a distributed computational problem using cloud computing technologies by designing and implementing a GraphRAG pipeline built on the [Apache Flink](https://flink.apache.org/) streaming model, deployed on AWS EKS, that extracts relations with an Ollama model and upserts the resulting knowledge graph into [Neo4j](https://neo4j.com/).
#### Grade: 20%

## Preliminaries and Context
Before starting this homework, please make sure that you have completed all preliminary steps designated in [homework 1](https://github.com/0x1DOCD00D/CS441_Fall2025/blob/main/Homeworks/Homework1.md) and [homework 2](https://github.com/0x1DOCD00D/CS441_Fall2025/blob/main/Homeworks/Homework2.md) where you built a batch index over a fixed MSRCorpus using Map/Reduce or CORBA and kept that index fresh as documents and models change using Spark or UIMA. Thus, homework 1 proves you can build a robust batch indexer whereas homework 2 proved you can keep that index fresh, cheap, and correct as both corpus and models change using massively parallel processing in the cloud environment. Now, it is time to create a system to reason logically about these documents.

Organizations sit on piles of PDFs, specs, tickets, and contracts where critical knowledge is buried across documents and versions, so firefights, audits, and upgrades take days instead of minutes. This homework teaches students a valuable skill - how to turn that chaos into an actionable knowledge graph: a Flink-powered pipeline extracts concepts and relations from indexed chunks with an Ollama model, fuses them with rule-based signals, and upserts versioned nodes and edges into Neo4j. Teams can then query grounded dependency chains with citations, surface contradictions, and get data-driven suggestions for architecture and policy, improving delivery speed, reliability, compliance, and the ROI of existing documentation. This homework is the natural capstone after Homeworks 1 and 2: once Homework 1 proved you can build a robust batch indexer and Homework 2 proved you can keep that index fresh, cheap, and correct as both corpus and models change at cloud scale, this assignment lifts the abstraction from documents → index to (documents+index) → knowledge graph. Students will reuse the same disciplined ingestion, versioning, and atomic publish patterns, but extend them with relation extraction (rules + Ollama), fusion, and idempotent upserts into Neo4j, all orchestrated by Flink on [EKS](https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html). In short, homework 1 gave you reliable bulk foundations, homework 2 gave you continuous, cost-aware freshness, and this project turns those capabilities into GraphRAG so teams can query dependencies, contradictions, and evidence paths rather than just retrieve text.

## Overview and Motivation
All three homeworks are created under the general umbrella of a course project that allows students to create and train a RAG index for an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. All homework descriptions are written using a retroscripting technique, in which the homework outlines are generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3Adg7IGPGYyKODxJgBwRT2bRKS0ig_u-IFqzOBkeXbuPo1%40thread.tacv2/conversations?groupId=01f0341d-bb02-4af6-8e0c-ad0b2a320a32&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

Companies are drowning in PDFs, wikis, tickets, specs, contracts, audit reports and compliance binders. Most of this knowledge is unstructured, cross-referential, and time-sensitive. A GraphRAG system converts that pile into an actionable network of concepts and relationships. Instead of full-text search that returns 200 documents, teams get dependency chains like `Service A depends on Library B, which conflicts with Policy C`, with citations to exact passages. That shortens root-cause analysis, change-impact estimation, and due-diligence from days to minutes. It also reduces operational risk, because contradictions and missing links surface automatically rather than during a postmortem.

From a revenue standpoint, GraphRAG accelerates product delivery and partner integration. Engineering leaders can ask `What do we break if we upgrade this SDK?` and receive a grounded subgraph with blast radius estimates. Sales and legal can map clauses that block a deal and find alternatives cited across previous contracts. Support can diagnose incidents by following `uses/depends on` paths across runbooks and architecture docs. The net effect is higher throughput per employee and fewer expensive escalations. For regulated industries, the graph becomes a living evidence ledger that demonstrates compliance with traceable links from requirements to implementation.

Commercial differentiation comes from the system’s ability to propose new connections, not just retrieve known ones. Link prediction and contradiction checks turn static documentation into a recommendation engine for architecture and process. That creates defensible intellectual property (IP): the more a company feeds its domain corpus, the better its private knowledge graph gets, and the harder it is for competitors to replicate the same organizational memory with generic tools. Cost control is built in, since GraphRAG targets exact evidence spans, minimizing LLM token usage while preserving auditability.

Educationally, this project is a perfect capstone for modern computing curricula. It combines IR fundamentals (chunking, BM25), distributed systems (Flink on EKS), knowledge representation (ontologies and schemas), databases (Neo4j/Cypher), cloud deployment (AWS) and prompt/LLM engineering into a single coherent build. Students see how algorithms meet real-world constraints like latency, idempotence, and cost, and they learn to evaluate systems by groundedness, precision/recall of edges, and drift metrics rather than toy benchmarks.

For courses in systems/software, data management, and AI, GraphRAG creates a hands-on lab where learners implement rule-based extractors, design prompts that return strict JSON, write idempotent upserts into Neo4j, and measure the trade-off between precision and recall as prompts or thresholds change. It also builds scientific habits. Every edge is tied to evidence spans, so students must argue from data, not vibes. That mindset transfers to research, where reproducibility and transparent provenance are essential. In short, the project teaches how to turn text into decisions, which is the core skill behind the next wave of AI-augmented work.

## Functionality
Thus, this third homework assignment is to create a GraphRAG program and expose its functionality using RESTful microservices. You will work with the same MSRCorpus dataset that consists of [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus), it is published under the directory MSRCorpus in this repository. Below is an end-to-end project blueprint that turns an existing document index and an Ollama model into a graph stored in Neo4j. Nodes represent concepts and chunk spans, and edges represent relations discovered from text and cross-document references. The execution engine is Apache Flink DataStream, deployed on EKS. The code sketches use Scala 3 and call Flink’s Java API from Scala as a production-minded, end-to-end project that turns an existing text index and an Ollama model into a graph of concepts and relations, stored in Neo4j, built as a Flink DataStream job running on Amazon EKS.

### Inputs
* Ollama model available to workers through a network endpoint or sidecar.
* Constructed index of documents or chunks with stable references to original sources.

### Outputs

* Knowledge graph where nodes represent concepts or text chunks and edges represent relations between them.
* Graph persisted to Neo4j through the Bolt driver with idempotent upserts.

### High-level architecture

```
[S3/Index Store] --> [Flink Source]
                         |
                         v
                 [Chunk Stream] --------------.
                         |                    |
                         v                    |
                [Concept Extraction]          |
                         |                    |
                         v                    |
              [Relation Candidates] <---------'
                         |
                         v
               [LLM Scoring via Ollama]
                         |
                         v
                 [Graph Projection]
                         |
                         v
                   [Neo4j Sink]
```

 A streaming topology lets you incrementally enrich and link content while keeping backpressure and retries under control. Each operator isolates a concern, which simplifies scaling and upgrades.

### Data model

* Node types
    * `Chunk` with fields `chunkId`, `docId`, `text`, `span`, `hash`, `sourceUri`.
    * `Concept` with fields `conceptId`, `lemma`, `surfaceForms`, `origin` (NER, keyphrase, title, tag).
* Edge types
    * `MENTIONS` from `Chunk -> Concept`.
    * `CO_OCCURS` between `Concept -> Concept` with `window`, `frequency`.
    * `RELATES_TO` between `Concept -> Concept` with `predicate`, `confidence`, `evidence`.
* Identity and dedup
    * Content-addressed `chunkId = sha256(docId + span + hash(text))`.
    * `conceptId = normalizedLemma + optional disambiguation key`.

This schema is needed to separate text containers from semantic units, supports provenance, allows multiple relation builders to coexist, and enables safe replay with idempotent upserts.

### End-to-end workflow

#### 1) Ingest the index into a Flink `DataStream`

**What**
Create sources that stream chunks from the index, preserving document references and chunk boundaries.

**Why**
Stable chunk identity is the anchor for all later joins, edges, and idempotent writes.

```scala
// build.sbt key libs
// "org.apache.flink" %% "flink-streaming-scala" % flinkVersion % "provided"
// "org.neo4j.driver" % "neo4j-java-driver" % neo4jDriverVersion

final case class Chunk(chunkId: String, docId: String, span: (Int, Int), text: String, sourceUri: String, hash: String)

val env = StreamExecutionEnvironment.getExecutionEnvironment

val chunks: DataStream[Chunk] =
  env.addSource(IndexSource.fromS3("s3://index-bucket/prefix"))
     .name("index-source")
     .uid("index-source")
```

#### 2) Normalize and language-tag chunks

**What**
Clean whitespace, detect language, compute stable hashes.

**Why**
Normalization avoids duplicate embeddings, language tags route to the right model prompts.

```scala
val normalized: DataStream[Chunk] =
  chunks.map(Normalize.cleanAndTag).name("normalize")
```

#### 3) Extract candidate concepts from each chunk

**What**
Combine fast heuristics with LLM-assisted extraction for high recall. Start with NER, noun phrase extraction, and keyphrase algorithms, then ask the LLM to refine or merge.

**Why**
Heuristics are cheap and deterministic. LLM adds semantic grouping and disambiguation when needed.

```scala
final case class Concept(conceptId: String, lemma: String, surface: String, origin: String)
final case class Mentions(chunkId: String, concept: Concept)

val mentions: DataStream[Mentions] =
  normalized.flatMap(ConceptStage.extractHeuristic)
            .name("concept-heuristics")
            .union(
              normalized.flatMap(ConceptStage.extractWithLLM(Ollama.client("http://ollama:11434")))
                        .name("concept-llm"))
```

#### 4) Build co-occurrence windows to propose relation candidates

**What**
Within a sliding window over the stream, create pairs of concepts that co-occur in proximity, and tally frequencies.

**Why**
Co-occurrence generates high-recall structural candidates without expensive model calls for every pair.

```scala
final case class CoOccur(a: Concept, b: Concept, windowId: String, freq: Long)

val coOccurs: DataStream[CoOccur] =
  mentions
    .keyBy(_.chunkId)
    .process(RelationStage.localCoOccurrence(windowSize = 3))
    .name("cooccur-local")
```

#### 5) Generate semantic relation candidates with light prompts

**What**
For each candidate pair, craft a compact prompt that asks the LLM to identify probable predicate types and supporting snippets, returning a JSON verdict.

**Why**
Moves from statistical correlation to semantic relation with measured cost and latency.

```scala
final case class RelationCandidate(a: Concept, b: Concept, evidence: String)
final case class ScoredRelation(a: Concept, predicate: String, b: Concept, confidence: Double, evidence: String)

val candidates: DataStream[RelationCandidate] =
  RelationStage.makeCandidates(normalized, mentions, coOccurs)
               .name("relation-candidates")

val scored: DataStream[ScoredRelation] =
  candidates.process(
    LLMStage.scoreRelationsWithOllama(
      client = Ollama.client("http://ollama:11434"),
      model  = "llama3:instruct",
      temperature = 0.0
    )
  ).name("relation-scoring")
```

#### 6) Project to graph primitives

**What**
Translate `Chunk`, `Concept`, and `ScoredRelation` into node and edge upserts, including `MENTIONS`, `CO_OCCURS` with a threshold, and `RELATES_TO` with predicate and confidence.

**Why**
A single projection layer decouples semantic logic from the database sink, making it easy to swap Neo4j for other graph stores.

```scala
sealed trait GraphWrite
final case class UpsertNode(label: String, id: String, props: Map[String, Any]) extends GraphWrite
final case class UpsertEdge(fromLabel: String, fromId: String, rel: String, toLabel: String, toId: String, props: Map[String, Any]) extends GraphWrite

val writes: DataStream[GraphWrite] =
  GraphProjector.project(normalized, mentions, coOccurs, scored)
                .name("graph-projector")
```

#### 7) Idempotent writes to Neo4j

**What**
Use MERGE semantics with deterministic primary keys and write-ahead retry on transient failures.

**Why**
Guarantees correctness under retries, scaling, and replays.

```scala
val neo4jSink = Neo4jSink
  .builder[GraphWrite]("bolt://neo4j:7687", "neo4j", sys.env("NEO4J_PASS"))
  .withUpsertMapper(GraphUpsert.mapper)
  .withBatchSize(200)
  .withMaxRetries(8)
  .build()

writes.addSink(neo4jSink).name("neo4j-sink")
```

#### 8) Monitoring and quality gates

**What**
Attach metrics for per-stage throughput, LLM latency, token usage, rejection rates, and graph growth, plus sampling validators.

**Why**
Graph quality is not obvious from counts. You need distributional checks and spot audits to avoid semantic drift.

```scala
Quality.attachMetrics(env, writes)
```

#### 9) Run on EKS

**What**
Package as a container, deploy a Flink Application cluster on EKS. Run Ollama as a DaemonSet or sidecar for low-latency local calls. Mount the index storage through S3.

**Why**
Kubernetes gives resilience and auto-scaling. Locality to Ollama reduces network cost and tail latency.

```
helm repo add flink https://downloads.apache.org/flink/flink-kubernetes-operator-helm
helm install graphrag flink/flink-kubernetes-operator -f deploy/flink-values.yaml
kubectl apply -f deploy/job-graph-rag.yaml
kubectl apply -f deploy/ollama-daemonset.yaml
```

### Relation construction details

#### Candidate generation strategies

1. Windowed co-occurrence
   Pairs from tokens, sentences, or chunk adjacency with frequencies and PMI as features.

2. Cue-pattern mining
   Lexical patterns like “X causes Y”, “X part of Y”, “X also known as Y” from dependency parses.

3. LLM light pass
   Compact prompts that return a predicate shortlist and an evidence span as JSON.

Multiple strategies complement one another. Co-occurrence is cheap, patterns are precise, LLMs generalize and resolve ambiguity.

#### LLM scoring prompt shape

* Input fields: `concept_a`, `concept_b`, `context_snippets[]`, `doc_refs[]`.
* Output fields: `predicate`, `confidence in [0,1]`, `evidence_span`, `ref`.

Why JSON with fixed schema? Because it is easy to validate, version, and evolve. It maps directly to case classes in Scala for type-safe parsing.

```scala
final case class LlmVerdict(predicate: String, confidence: Double, evidence: String, ref: String)
```

### Idempotency, updates, and reprocessing

* Stable IDs for chunks and concepts allow safe MERGE.
* Upsert policy re-writes edges if confidence improves beyond a hysteresis margin.
* Replay-safe sources so you can reprocess without duplication.
* Versioned prompts in metadata to roll forward or back LLM behavior.

Graphs evolve as models and heuristics improve. You want repeatable builds, safe restarts, and well-defined semantics for updates.

### Error handling and backpressure: Optional

* LLM backoff + token budget per parallel instance with a queue and circuit breaker.
* Dead-letter stream for malformed index entries or LLM timeouts.
* Checkpointing with Flink state and exactly-once sinks where feasible.

Stable throughput and predictable cost rely on bounded queues and graceful degradation when LLM latency spikes.

### Example Neo4j upsert logic

```scala
object GraphUpsert:
  def mapper: GraphWrite => Seq[String] = {
    case UpsertNode("Concept", id, props) =>
      Seq(
        "MERGE (c:Concept {conceptId: $id}) SET c += $props",
      )
    case UpsertNode("Chunk", id, props) =>
      Seq(
        "MERGE (ch:Chunk {chunkId: $id}) SET ch += $props",
      )
    case UpsertEdge("Chunk", from, "MENTIONS", "Concept", to, props) =>
      Seq(
        "MERGE (ch:Chunk {chunkId: $from})",
        "MERGE (c:Concept {conceptId: $to})",
        "MERGE (ch)-[r:MENTIONS]->(c) SET r += $props"
      )
    case UpsertEdge("Concept", a, "RELATES_TO", "Concept", b, props) =>
      Seq(
        "MERGE (a:Concept {conceptId: $a})",
        "MERGE (b:Concept {conceptId: $b})",
        "MERGE (a)-[r:RELATES_TO]->(b) SET r += $props"
      )
    case UpsertEdge("Concept", a, "CO_OCCURS", "Concept", b, props) =>
      Seq(
        "MERGE (a:Concept {conceptId: $a})",
        "MERGE (b:Concept {conceptId: $b})",
        "MERGE (a)-[r:CO_OCCURS]->(b) SET r.freq = coalesce(r.freq,0) + $inc"
      )
  }
```

### Configuration

```yaml
# deploy/application.conf
ollama:
  endpoint: "http://ollama:11434"
  model: "llama3:instruct"
  temperature: 0.0
  timeoutMs: 15000

neo4j:
  uri: "bolt://neo4j:7687"
  user: "neo4j"
  passEnv: "NEO4J_PASS"

index:
  source: "s3://index-bucket/prefix"
  format: "jsonl"

relation:
  cooccur:
    window: 3
    minPmi: 0.2
  llm:
    predicateSet: ["is_a","part_of","causes","synonym_of","related_to"]
    minConfidence: 0.65
```

Externalized configuration lets you tune thresholds, swap models, and roll predicates without rebuilding containers.

### Suggested Directory layout

```
graphrag/
  build.sbt
  modules/
    core/
      src/main/scala/...
    ingestion/
      src/main/scala/...
    neo4j/
      src/main/scala/...
    llm/
      src/main/scala/...
  deploy/
    flink-values.yaml
    job-graph-rag.yaml
    ollama-daemonset.yaml
    application.conf
  docs/
    architecture.md
    prompts.md
```

### Operational notes or advice - optional

* Prefer sidecar or DaemonSet Ollama for node-local inference and lower latency.
* Keep temperature at 0 for classification prompts to stabilize outputs.
* Use Flink checkpoints to S3 and enable exactly-once semantics where applicable.
* Run Neo4j Aura or HA cluster for durability and easy scaling.
* Expose Grafana dashboards for throughput, LLM latency, token spend, and graph growth.

### What you can show off to your prospective employers

* A repeatable pipeline that converts indexed text into a queryable knowledge graph.
* Clear separation of concerns between extraction, candidate generation, LLM scoring, and graph persistence.
* Cloud-native deployment on EKS with Flink scalability and Neo4j analytics friendliness.

This repository template and workflow let you process the input index and Ollama model, obtain relation candidates with both statistical and semantic signals, and construct a robust graph of concepts and chunk-level links stored in Neo4j.

---
### Example GraphRAG query and response (MSR-corpus scenario)

Below is a concrete, end-to-end illustration of how a GraphRAG built from Mining Software Repositories (MSR) conference papers can answer a nuanced research question. All paper IDs, titles, and numbers are illustrative placeholders to show the mechanics without relying on specific real papers.

#### User query

> “Since 2018, which techniques improved just-in-time defect prediction on commit-level datasets similar to JITGIT, and by how much compared to common Random Forest baselines? Please group results by technique family, report typical gains, and cite evidence.”

This is a good GraphRAG query because it mixes concepts (task = JIT defect prediction), entities (datasets like JITGIT), time constraints (since 2018), baselines (Random Forest), and requires synthesis beyond keyword search. The graph structure helps expand synonyms, follow citations, filter by time, and aggregate metric deltas.

---

#### What the graph lookup does behind the scenes

**Concepts & nodes**

* `Task: JIT Defect Prediction`
* `Technique` families like `Deep Learning`, `Graph Neural Networks`, `Meta-learning`, `Commit2Vec embeddings`.
* `Dataset` nodes like `JITGIT`, `SEOSS-JIT`, `CommitGuru-style`.
* `Metric` nodes like `AUC`, `F1`, `MCC`.
* `Paper` nodes with year, venue, and stable `paperId`.

**Relations**

* `(Paper)-[:ADDRESSES]->(Task)`
* `(Paper)-[:USES_DATASET]->(Dataset)`
* `(Paper)-[:REPORTS]->(Metric {name:"AUC", value:…})`
* `(Paper)-[:IMPROVES_OVER {delta:…, metric:"AUC"}]->(Baseline {name:"Random Forest"})`
* `(Paper)-[:PROPOSES]->(Technique)`
* `(Paper)-[:EVIDENCE]->(Snippet {text:…, span:…, chunkId:…})`

---

#### Example Cypher plan the GraphRAG might run

```cypher
// 1) Find JIT defect prediction papers since 2018 that compare to Random Forest
MATCH (t:Task {name:"JIT Defect Prediction"})<-[:ADDRESSES]-(p:Paper)
WHERE p.year >= 2018
MATCH (p)-[:PROPOSES]->(tech:Technique)
MATCH (p)-[:USES_DATASET]->(d:Dataset)
WHERE d.name IN ["JITGIT","SEOSS-JIT","CommitGuru-style"]
MATCH (p)-[imp:IMPROVES_OVER {metric:"AUC"}]->(b:Baseline {name:"Random Forest"})
MATCH (p)-[:REPORTS]->(m:Metric {name:"AUC"})
WITH p, tech, d, m, imp
RETURN tech.family AS family,
       collect({
         paperId: p.paperId,
         title: p.title,
         year: p.year,
         dataset: d.name,
         auc: m.value,
         delta: imp.delta
       }) AS results
ORDER BY family;
```

**Why**
* The `Task` node focuses retrieval on the right problem and expands synonyms like “commit-level bug prediction.”
* Dataset nodes constrain evidence to JIT-style corpora.
* `IMPROVES_OVER` edges carry already-computed deltas against a named baseline, so we do not re-parse PDFs on the fly.
* We return grouped results for downstream LLM synthesis.

---

#### Example graph hits (illustrative)
```
family = "Deep Learning (Commit Embeddings)"
results =
  [
    {paperId:"MSR2020-045", title:"Commit2Vec-JIT", year:2020,
     dataset:"JITGIT", auc:0.78, delta:+0.06},
    {paperId:"MSR2021-112", title:"Hybrid CNN-RNN for JIT", year:2021,
     dataset:"SEOSS-JIT", auc:0.80, delta:+0.08}
  ]

family = "Graph Neural Networks"
results =
  [
    {paperId:"MSR2019-073", title:"HeteroGNN-JIT", year:2019,
     dataset:"CommitGuru-style", auc:0.79, delta:+0.05},
    {paperId:"MSR2022-056", title:"CodeGraph-JIT", year:2022,
     dataset:"JITGIT", auc:0.81, delta:+0.07}
  ]

family = "Meta-learning / Transfer"
results =
  [
    {paperId:"MSR2023-018", title:"MetaJIT: Few-shot Adaptation", year:2023,
     dataset:"JITGIT", auc:0.77, delta:+0.04}
  ]
```

**Stored evidence snippets** (linked by `:EVIDENCE`) might include short quotes like
* “Our CNN-RNN model achieves AUC 0.80 on SEOSS-JIT, improving over Random Forest by 0.08.”
* “CodeGraph-JIT raises AUC from 0.74 to 0.81 on JITGIT.”

---

#### Example LLM synthesis the GraphRAG would return

**Answer**
Since 2018, three families stand out for improving commit-level just-in-time defect prediction on JIT-style datasets. Deep learning models that learn commit embeddings report typical AUC gains of 0.06 to 0.08 over Random Forest on JITGIT and SEOSS-JIT, with representative values around 0.78 to 0.80. Graph neural network approaches that encode code and change structure show consistent improvements of 0.05 to 0.07, with AUC near 0.79 to 0.81 on JITGIT and CommitGuru-style corpora. Meta-learning methods provide smaller but useful gains of about 0.04, especially when target projects have limited labeled history. These results suggest structure-aware representations and temporal commit context help more than feature-engineered baselines, while transfer methods reduce cold-start gaps.

**Table of illustrative results**

| Technique family | Representative method  | Dataset          | AUC  | Δ vs RF |
| ---------------- | ---------------------- | ---------------- | ---- | ------- |
| Deep Learning    | Commit2Vec-JIT         | JITGIT           | 0.78 | +0.06   |
| Deep Learning    | Hybrid CNN-RNN for JIT | SEOSS-JIT        | 0.80 | +0.08   |
| GNN              | HeteroGNN-JIT          | CommitGuru-style | 0.79 | +0.05   |
| GNN              | CodeGraph-JIT          | JITGIT           | 0.81 | +0.07   |
| Meta-learning    | MetaJIT                | JITGIT           | 0.77 | +0.04   |

**Sources**
[MSR2020-045], [MSR2021-112], [MSR2019-073], [MSR2022-056], [MSR2023-018].
Each source links to evidence snippets and metric nodes in the graph, so results are auditable.

**Why GraphRAG returns this kind of answer**
1. The graph lets the system pivot on Task → Technique → Dataset → Metric without brittle keyword matching.
2. Precomputed `IMPROVES_OVER` edges store metric deltas and baselines, enabling fast aggregation.
3. The LLM sees a compact, structured context, so it can synthesize families, typical gains, and caveats rather than reciting abstracts.
4. Citations point to evidence snippets and paper nodes for verification and follow-up exploration.

---

#### Variant: exploratory question the graph also supports

> “Show me techniques that transfer across repositories and still beat Random Forest on cold-start settings.”

A GraphRAG plan would follow `:ADDRESSES(Task="JIT Defect Prediction")` → `:CLAIMS(Property="Cross-project Transfer")` → `:CONDITION(Setting="Cold-start")` → `:IMPROVES_OVER(Baseline="RF")`, then return candidate papers and their reported deltas, which the LLM summarizes with caveats about data scarcity and variance.

---
### RESTful Microservices for GraphRAG (MSR Corpus)

These services expose the GraphRAG built over the MSRCorpus papers. They provide synchronous and asynchronous query paths, evidence retrieval, graph neighborhood exploration, and admin utilities. All services are stateless behind an API gateway, backed by Neo4j for graph reads and a Flink JobService for any long-running expansions or LLM rescoring.


#### Overview
```
[Client]
   |
[API Gateway / AuthN-Z / Rate-limit]
   |
   +-- QueryService        --> Neo4j (read), Cache
   +-- EvidenceService     --> Neo4j (read), Object store (snippets)
   +-- ExploreService      --> Neo4j (read)
   +-- ExplainService      --> Neo4j (read), Prompt templates
   +-- JobsService         --> Flink JobService (async), Neo4j
   +-- MetadataService     --> Neo4j (read)
```

* **Auth**: OAuth 2.0 Bearer tokens or API keys.
* **Formats**: JSON requests and responses, UTF-8, application/json.
* **Versioning**: Path based, for example `/v1/...`.
* **Idempotency**: `Idempotency-Key` header for POST endpoints that create jobs.
* **Caching**: `ETag` and `Cache-Control` on GETs, server-side memoization keyed by normalized query.
* **Pagination**: `limit`, `offset`. For cursor mode use `nextPageToken`.

---

#### 1) QueryService: POST `/v1/query`

Submit a semantic query. If it can be answered from the graph immediately, a synchronous answer is returned. If the system needs heavy expansion, a 202 with a `jobId` is returned.

**Request**

```json
{
  "query": "Since 2018, which techniques improved just-in-time defect prediction on JITGIT compared to Random Forest?",
  "timeRange": {"from": 2018, "to": 2025},
  "constraints": {
    "datasets": ["JITGIT", "SEOSS-JIT", "CommitGuru-style"],
    "baselines": ["Random Forest"]
  },
  "output": {
    "groupBy": ["techFamily"],
    "metrics": ["AUC", "F1"],
    "topKPerGroup": 5,
    "includeCitations": true
  }
}
```

**Synchronous 200 Response**

```json
{
  "mode": "sync",
  "summary": "Three technique families show consistent gains since 2018...",
  "groups": [
    {
      "key": {"techFamily": "Deep Learning"},
      "items": [
        {
          "paperId": "MSR2020-045",
          "title": "Commit2Vec-JIT",
          "dataset": "JITGIT",
          "metric": {"name": "AUC", "value": 0.78, "deltaVsRF": 0.06},
          "citations": ["evid:chunk-7f2a", "evid:chunk-9c11"]
        }
      ]
    }
  ],
  "evidenceAvailable": true,
  "explainLink": "/v1/explain/trace/req-1a2b"
}
```

**Asynchronous 202 Response**

```json
{
  "mode": "async",
  "jobId": "job-8a1c3",
  "statusLink": "/v1/jobs/job-8a1c3",
  "pollAfterMs": 2000
}
```

**Query normalization and why**
The service canonicalizes the request into a graph pattern over Task, Dataset, Technique, and Metric. This allows cache hits for logically identical queries and stable aggregation.

---

#### 2) JobsService: GET `/v1/jobs/{jobId}`

Check status for an asynchronous query.

**Response**

```json
{
  "jobId": "job-8a1c3",
  "state": "SUCCEEDED",
  "startedAt": "2025-11-03T19:35:10Z",
  "finishedAt": "2025-11-03T19:35:38Z",
  "resultLink": "/v1/jobs/job-8a1c3/result"
}
```

GET `/v1/jobs/{jobId}/result`

Fetch the answer when ready.

**Response**

```json
{
  "summary": "Graph neural networks and deep embeddings improved AUC by 0.05–0.08 on JITGIT-like datasets.",
  "groups": [...],
  "citations": [
    {"paperId":"MSR2022-056","evidenceId":"evid:chunk-b3e0","span":"p.5 AUC 0.81 vs RF 0.74"}
  ]
}
```

Async exists because some queries require graph neighborhood expansions or LLM rescoring across many candidates. Offloading these to Flink keeps tail latency predictable for the API.

---

#### 3) EvidenceService: GET `/v1/evidence/{evidenceId}`

Return the stored snippet that supports a claim.

**Response**

```json
{
  "evidenceId": "evid:chunk-b3e0",
  "paperId": "MSR2022-056",
  "chunkId": "ch-1fd22",
  "text": "Our model achieves AUC 0.81 on JITGIT, exceeding the Random Forest baseline of 0.74.",
  "docRef": {
    "title": "CodeGraph-JIT",
    "year": 2022,
    "url": "https://example.org/papers/MSR2022-056.pdf"
  }
}
```

Users can audit the sources behind synthesized answers and export snippets for reproducible research.

---

#### 4) ExploreService

### GET `/v1/graph/concept/{conceptId}/neighbors`

Return a lightweight neighborhood for interactive exploration.

**Query params**
`direction=in|out|both`, `depth=1..3`, `limit=100`, `edgeTypes=RELATES_TO,CO_OCCURS,MENTIONS`

**Response**

```json
{
  "center": {"id":"concept:JIT_Defect_Prediction","label":"Concept"},
  "nodes": [
    {"id":"concept:Random_Forest","label":"Concept","props":{"family":"Baseline"}},
    {"id":"paper:MSR2022-056","label":"Paper","props":{"year":2022}}
  ],
  "edges": [
    {"from":"paper:MSR2022-056","to":"concept:JIT_Defect_Prediction","type":"ADDRESSES"},
    {"from":"concept:JIT_Defect_Prediction","to":"concept:Random_Forest","type":"HAS_BASELINE"}
  ],
  "page": {"limit": 50, "nextPageToken": null}
}
```

Exploration endpoints power UI graph viewers and programmatic analyses.

---

#### 5) ExplainService: GET `/v1/explain/trace/{requestId}`

Return an execution trace for a past query.

**Response**

```json
{
  "requestId": "req-1a2b",
  "plan": [
    {"step":"matchTask","cypher":"MATCH (t:Task {name:'JIT Defect Prediction'}) ..."},
    {"step":"filterTime","detail":"p.year >= 2018"},
    {"step":"baselineEdge","detail":"IMPROVES_OVER metric=AUC baseline=Random Forest"}
  ],
  "promptVersions": {"relationScoring":"v3.2"},
  "counters": {"nodesRead": 1489, "relsRead": 5503, "llmCalls": 0, "cacheHits": 1}
}
```

**Why**
Users can see how the system derived the answer and which constraints were applied.

---

## Error model

* `400 Bad Request` invalid JSON or unknown parameters.
* `401 Unauthorized` missing or invalid token.
* `403 Forbidden` insufficient scope.
* `404 Not Found` unknown `paperId` or `evidenceId`.
* `409 Conflict` duplicate request without `Idempotency-Key`.
* `422 Unprocessable Entity` unsupported constraint combination.
* `429 Too Many Requests` rate limit exceeded.
* `500 Internal Server Error` generic failure with trace id.

**Error payload**

```json
{
  "error": "Unprocessable Entity",
  "code": "UNSUPPORTED_BASELINE",
  "message": "Baseline 'SVM-RBF' is not indexed for metric AUC in this corpus.",
  "traceId": "tr-9f88"
}
```

---

#### Implementation notes

* **Framework** Akka HTTP or http4s for Scala 3, both support streaming and backpressure.
* **Neo4j** Use the official Java driver with read-only sessions and query templates. Parameterize all Cypher.
* **Flink** JobsService posts normalized query specs to a Kafka topic that triggers a Flink pipeline for large graph expansions or rescoring. Results are written back to Neo4j and cached.
* **Observability** Every response carries `X-Trace-Id`. Export metrics for P50, P95, token use, cache hit rate, and Neo4j query latency.

---

## Example usage

**cURL synchronous**

```bash
curl -s -X POST https://api.example.com/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Techniques improving JIT defect prediction on JITGIT since 2018 vs Random Forest",
    "timeRange": {"from": 2018, "to": 2025},
    "constraints": {"datasets": ["JITGIT"], "baselines": ["Random Forest"]},
    "output": {"groupBy": ["techFamily"], "metrics": ["AUC"], "topKPerGroup": 5, "includeCitations": true}
  }'
```

**cURL evidence fetch**

```bash
curl -s https://api.example.com/v1/evidence/evid:chunk-b3e0 \
  -H "Authorization: Bearer $TOKEN"
```

---

#### Why I chose this microservices design

* Separate services let you scale hot paths independently and apply tighter limits to heavy operations.
* Query normalization plus caching reduces duplicate load and improves latency.
* Evidence and explain endpoints build trust by making answers auditable.
* Async jobs ensure large expansions do not block the fast path while keeping the API simple for clients.

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your design and implemented components work to solve the problem, and the documentation that describe the build and runtime process, to be considered for grading. Grading rubrics will be posted on our Teams channel. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are.

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from [the big brother](https://www.cs.uic.edu/~drmark/) :-) who is watching your exchanges. However, *you must not describe your architecture or other specific details related to how you construct your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Saturday, November, 29, 2025 at 10PM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters you chose for your experiments, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.


## Evaluation criteria
- the maximum grade for this homework is 20%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 20%-2% => 18%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic Neo4J or Flink examples from some repos are given and nothing else is done: zero grade;
- using Python or Java or some other language instead of Scala: 10% penalty;
- having less than ten unit and/or integration scalatests: up to 10% lost;
- missing comments and explanations from your program with clarifications of your design rationale: up to 10% lost;
- logging is not used in your programs: up to 5% lost;
- hardcoding the input values in the source code instead of using the suggested configuration libraries: up to 5% lost;
- for each used *var* for heap-based shared variables or mutable collections without explicitly stated reasons: 0.3% lost;
- for each used *while* or *for* or other loops with induction variables to iterate over a collection: 0.5% lost;
- no instructions in README.md on how to install and run your program: up to 10% lost;
- the program crashes without completing the core functionality: up to 20% lost;
- the documentation exists but it is insufficient to understand your program design and models and how you assembled and deployed all components of your mappers and reducers: up to 20% lost;
- the minimum grade for this homework cannot be less than zero.

That's it, folks!