# Homework 2
### Congratulations on completing homework 1 where you created and applied [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) to a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from [Ollama](https://ollama.com/search) using [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus). The second homework requires students to implement an incremental delta indexer that detects and processes only changed documents and chunks, upserts versioned embeddings, and atomically publishes a refreshed retrieval index using Spark or UIMA.

Much of the background information is based on the books [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription and books on [A Simple Guide to Retrieval Augmented Generation by Abhinav Kimothi](https://www.manning.com/books/a-simple-guide-to-retrieval-augmented-generation) and [Essential GraphRAG by Tomaž Bratanič and Oskar Hane](https://www.manning.com/books/essential-graphrag). A majority of images in this and other homework descriptions are used from these books.

#### The goal of this homework is for students to gain experience with solving a distributed computational problem using cloud computing technologies. The main textbook group (option 1) will design and implement an instance of the [Spark](https://spark.apache.org/) computational model using AWS EMR whereas the alternative textbook group (option 2) will use the [UIMA](https://uima.apache.org/) model. You can check your textbook option in the corresponding column of the gradebook on the Blackboard.
#### Grade: 15%

## Preliminaries and Context
Before starting this homework, please make sure that you have completed all preliminary steps designated in [homework 1](https://github.com/0x1DOCD00D/CS441_Fall2025/blob/main/Homeworks/Homework1.md) where you built a batch index over a fixed MSRCorpus using Map/Reduce or CORBA. Homework 2 keeps that index fresh as documents and models change using Spark or UIMA. The first homework focuses on throughput for a one time job, the second focuses on correctness and cost when the corpus evolves daily. In Homework 1 you worked with a static snapshot, that is, these documents were available at once and the corpus stayed static during the index construction process. In Homework 2 the corpus is alive or you pretend that new documents arrive on a regular basis as it happens in the real-world setting.

In Homework 1 the pipeline was batch. Map/Reduce jobs or CORBA servants run to completion and then stop. In Homework 2 you design an incremental pipeline. With Spark you express anti joins and upserts, checkpoint progress, and publish snapshots atomically. With UIMA you gate unchanged documents, keep idempotent writes in your store, and can scale annotators with UIMA AS. Back pressure, batching, and partial reprocessing become first class concerns.

Homework 1 can get by with a single retrieval index plus logs. Provenance is minimal because the work happens once. Homework 2 adopts layered and versioned storage. You keep normalized documents, chunks, and embeddings as separate tables, then materialize a denormalized retrieval index for serving. With Spark this often means Delta or Iceberg with schema evolution and time travel. With UIMA you mimic merge semantics in a database or Parquet. Either way you can roll back to a previous version and audit what changed. Versioning is not required for this homework but it is a good practice.

Batch work in Homework 1 tolerates coarse retries. If a reducer or servant fails you rerun the step and move on. Incremental work in Homework 2 requires careful keys and idempotent upserts. Replays should never duplicate chunks or vectors. You rely on document id with content hash, and chunk id with embedder version, so that retried batches write the same records and only once.

In Homework 1 you measure quality after the final build and worried about the total cost for the single run. In Homework 2 you track freshness, throughput to the embedder service, deduplication ratio, and publish history. You test that a no change run does no work, that an edit in one file triggers rechunking and reembedding only for that file, and that a model version bump computes new vectors exactly once while keeping the previous version available.

Homework 1 asks for a correct chunking and embedding pipeline over the static MSRCorpus, an efficient MapReduce or CORBA client, and a final index artifact. Homework 2 asks for working delta logic that skips unchanged content, deterministic identifiers, versioned embeddings, atomic publish of the retrieval index, and a brief report with first run versus delta run timings and resource use. 

Thus, homework 1 proves you can build a robust batch indexer. Homework 2 proves you can keep that index fresh, cheap, and correct as both corpus and models change. This mirrors the difference between a one time data migration and a production knowledge system that must stay current every day.

## Overview and Motivation
All three homeworks are created under the general umbrella of a course project that allows students to create and train a RAG index for an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. All homework descriptions are written using a retroscripting technique, in which the homework outlines are generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3Adg7IGPGYyKODxJgBwRT2bRKS0ig_u-IFqzOBkeXbuPo1%40thread.tacv2/conversations?groupId=01f0341d-bb02-4af6-8e0c-ad0b2a320a32&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

In many industries, a RAG system that powers search, copilots, and customer support lives on a moving target: new docs arrive every minute, old ones change, models evolve, and stakeholders expect fresh, trustworthy answers right now. An incremental delta indexer turns that chaos into a controllable pipeline. Instead of burning hours and dollars re-embedding the entire corpus, it detects exactly what changed, re-chunks only the affected passages, and computes vectors just for those deltas. That keeps latency low and costs predictable, lets teams push daily or even hourly content updates, and enables safe model upgrades by versioning embeddings and swapping indexes with blue/green releases. With time-travel and provenance baked in, you also get auditability for compliance and easy rollbacks when an experiment underperforms.

As a cloud computing project, this is a perfect playground for serious engineering: streaming ingestion, idempotent upserts, batching to saturate embedder services, hybrid BM25+vector retrieval, and SLO-aware scheduling that balances freshness against spend. You’ll design schemas that survive change, wire services that scale elastically, and build an evaluation harness to prove quality and cost wins with real metrics. The result looks and feels like a production system a company would ship—fast, frugal, observable, and resilient—and it gives you portfolio-ready artifacts that showcase the exact skills modern AI platforms hire for. It is perfect for a course project because it combines theory and practice, covers a wide range of cloud computing concepts, and results in a tangible, impressive outcome. Showing this project on your LinkedIn profile or resume will definitely make you stand out in the competitive job market because students get to build something that mirrors real industry stacks—a pipeline that’s fast, cheap, observable, and correct, with clean separation of concerns between dataflow (Spark) and NLP/annotation (UIMA). It is résumé rocket fuel: the exact skills behind search, copilots, and knowledge platforms at scale.

For a commercial RAG platform, **Spark** is the force multiplier for freshness and frugality if you belong to book option 1. With structured streaming plus Delta/Iceberg, you express incrementality as simple anti-joins and MERGEs: only re-chunk and re-embed what actually changed, checkpoint progress, and atomically publish new snapshots for blue/green rollouts (blue environment is the current live production environment and green is a staging environment where the new version of the application is deployed and tested). Spark’s executors let you batch thousands of chunk embeddings per micro-batch, saturate GPU/CPU services, and keep strict SLOs while costs stay linear and predictable. You also inherit serious production hygiene—schema evolution, time-travel, compaction, lineage—which is exactly what security and compliance teams expect before anything touches customers.

**UIMA** makes the unstructured part elegant and testable, something that students from the book option 2 group can enjoy. Its CAS/type system gives you first-class annotations for sections, tables, code blocks, and entities, so chunking and enrichment are deterministic and auditable. With **UIMA-AS**, you can turn heavy annotators—OCR, PDF normalization, language detection, even specialized domain parsers—into scalable services. UIMA delivers high-quality document understanding. With Spark or UIMA, you can build a RAG indexer that’s not just functional but production-grade: fast, frugal, observable, and resilient. It’s a portfolio piece that showcases exactly the skills modern AI platforms hire for.

## Functionality
Your second homework assignment is to create a program for **continuous** parallel distributed processing of a large corpus of text. You will work with the same MSRCorpus dataset that consists of [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus), it is published under the directory MSRCorpus in this repository. A general goal is to create a program that processes these PDF files in parallel in a simulated continuous document arrival mode and produces a set of vector embeddings for the text in these files. The output of your program is a file with token embeddings and various statistics about the data.

To summarize the workflow for this homework includes the same large-granularity steps that we summarized for homework 1 as (1) choose a model, (2) RAG it, and (3) deploy on AWS. All conceptual computation steps to build indexes from homework 1 apply. To create the delta-incremental RAGged Ollama model you should do the following.
* Step 1: ingest new and changed documents from the source and assign a stable document id.
* Step 2: normalize each document with text extraction and language detection, then compute a content hash over the normalized text.
* Step 3: consult the delta store and skip any document whose stored content hash matches the current one.
* Step 4: chunk only the changed documents with deterministic boundaries and compute a stable chunk id that depends on document id, offsets, and content hash.
* Step 5: select chunks that do not yet have vectors for the current embedder and version, then batch and call the embedding service.
* Step 6: upsert (a write operation that inserts a new record when the key does not exist, and updates the existing record when the key already exists, which makes repeated writes idempotent and safe during retries) results into versioned tables for documents, chunks, and embeddings, keeping old versions for audit and rollback.
* Step 7: materialize a refreshed retrieval index by joining document fields, chunks, and embeddings, then publish atomically as the new snapshot.
* Step 8: verify delta behavior by rerunning on the same corpus to confirm no work is performed, and record metrics for freshness, throughput, and deduplication.
* Step 9: deploy on AWS EMR or as UIMA-AS on AWS, and document your design and implementation. For a simple and reliable UIMA-AS deployment on AWS, you can place a JMS broker in a private VPC subnet, run your UIMA-AS workers in the same VPC, and keep all ingress closed to the public internet. Start with one ActiveMQ broker on a small EC2 instance or a container in ECS, restrict a selected port to a worker security group, and expose the web console only through a bastion or VPN. Package each heavy annotator, such as normalize, chunk, embed, as its own container or EC2 process, point each to the broker URL, and keep annotators stateless and idempotent so retries are safe. Store delta state such as document hashes, chunk ids, and embedding versions in a durable database like RDS Postgres, put vectors and index snapshots in S3, and attach an IAM role that grants only the reads and writes you need. Send logs to CloudWatch, keep broker credentials in Secrets Manager, and tag every container with pipeline and model version so you can trace changes. You can use AWS CloudFormation or Terraform to script your infrastructure, but manual setup is fine for a course project. *In any case, document a setup of your own choice so it can be reproduced*.

When queues grow, scale by increasing worker count or batch size rather than vertical instance size, and track a small set of metrics to guide tuning, such as queue depth, dequeue rate, p95 service time, and error count. Use ECS Fargate for workers if you want managed autoscaling and rolling updates, keep the broker on EC2 for predictable state, and add an NLB only if you move to broker redundancy across 2 AZs. Organize S3 prefixes by environment and by model version to simplify cleanup and blue-green index releases. Validate the setup by replaying an unchanged corpus to confirm near-zero work, edit one file to verify targeted reprocessing, and record first-run versus delta-run timings so you can show cost and freshness gains.

### Assignment for the main textbook group
This workflow keeps a RAG corpus fresh in Spark by detecting deltas, then reprocessing only what changed. A DataFrame holds `docId`, `uri`, `title`, `language`, and `contentHash`, and another holds chunks with `chunkId`, `chunkIx`, `start`, `end`, `sectionPath`, `text`, and `contentHash`. You read files, normalize text, compute a stable `docId` from the URI, and a `contentHash` from normalized text. Consider the following example.

```scala
import org.apache.spark.sql.functions._
val raw = spark.read.format("binaryFile").load("s3://bucket/docs/")
val normalizeUdf = udf((bytes: Array[Byte]) => normalize(new String(bytes, "UTF-8")))
val docs = raw.select(
  col("path").as("uri"),
  normalizeUdf(col("content")).as("text")
).withColumn("language", detectLang(col("text")))
 .withColumn("title", firstLineOrName(col("text"), col("uri")))
 .withColumn("docId", sha2(col("uri"), 256))
 .withColumn("contentHash", sha2(col("text"), 256))
```

Delta detection uses anti joins so unchanged documents are skipped, and deterministic chunking turns each changed document into stable segments whose identifiers depend on document id, offsets, and content hash. Upserts keep writes idempotent. Consider the following example.

```scala
// Skip unchanged docs
val existingDocs = spark.table("rag.doc_normalized").select("docId","contentHash").distinct
val toProcess = docs.join(existingDocs, Seq("docId","contentHash"), "left_anti")

// Chunk deterministically
val chunkUdf = udf(chunkParagraphAware _)
val chunked = toProcess
  .withColumn("chunk", explode(chunkUdf(col("text")))) // chunk => struct(start,end,text,ix,sectionPath)
  .select(
    col("docId"), col("contentHash"),
    col("chunk.ix").as("chunkIx"),
    col("chunk.start").as("start"),
    col("chunk.end").as("end"),
    col("chunk.text").as("chunkText"),
    col("chunk.sectionPath").as("sectionPath")
  ).withColumn("chunkId", sha2(concat_ws(":", col("docId"), col("start"), col("end"), col("contentHash")), 256))

// Upsert docs and chunks
docs.createOrReplaceTempView("incoming_docs")
spark.sql("""
MERGE INTO rag.doc_normalized t
USING incoming_docs s
ON t.docId = s.docId
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")
chunked.createOrReplaceTempView("incoming_chunks")
spark.sql("""
MERGE INTO rag.chunks t
USING incoming_chunks s
ON t.chunkId = s.chunkId
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")
```

Embedding is delta aware, only chunks missing vectors for the current embedder and version are sent in batches using `foreachBatch`, then the retrieval index is materialized by joining documents, chunks, and embeddings into a versioned snapshot and published atomically. Consider the following example.

```scala
val embedder = lit("mxbai-embed-large")
val ver = lit("1.3.0")
val needEmb = spark.table("rag.chunks").select("chunkId","contentHash","chunkText")
  .join(spark.table("rag.embeddings")
        .where(col("embedder")===embedder && col("embedder_ver")===ver)
        .select("chunkId"), Seq("chunkId"), "left_anti")

def embedBatch(df: org.apache.spark.sql.DataFrame, id: Long): Unit = {
  import df.sparkSession.implicits._
  df.as[(String,String,String)].mapPartitions { it =>
    val batched = it.grouped(64).flatMap { group =>
      val ids = group.map(_._1).toList
      val texts = group.map(_._3).toList
      val vecs = callEmbedder(ids, texts, "mxbai-embed-large", "1.3.0") // Map[chunkId, Array[Float]]
      group.map { case (cid, ch, _) => (cid, ch, "mxbai-embed-large", "1.3.0", vecs(cid)) }
    }
    batched
  }.toDF("chunkId","contentHash","embedder","embedder_ver","embedding")
   .write.format("delta").mode("append").saveAsTable("rag.embeddings")
}
needEmb.writeStream.foreachBatch(embedBatch _)
  .option("checkpointLocation","s3://chk/embeddings").start()

spark.sql("""
CREATE OR REPLACE TABLE rag.retrieval_index AS
SELECT c.chunkId, c.docId, c.chunkText, c.sectionPath, d.title, d.language,
       e.embedding, e.embedder, e.embedder_ver, c.contentHash, current_timestamp() as version_ts
FROM rag.chunks c
JOIN rag.doc_normalized d USING(docId, contentHash)
JOIN rag.embeddings e USING(chunkId, contentHash)
""")
```

### Assignment for the alternative textbook group
The same goal is to create a workflow with UIMA that keeps a RAG corpus fresh by detecting deltas, then reprocessing only what changed. CAS in UIMA is the Common Analysis Structure, an in-memory data model that carries the subject of analysis together with all annotations produced by the pipeline. It stores the document text, one or more alternate views like plain text and extracted tables, and a graph of typed feature structures that follow your type system. Annotators read and write these feature structures, and the CAS maintains indexes so components can find annotations efficiently. In Java/Scala you often use the [JCas API](https://uima.apache.org/d/uimaj-current/api/org/apache/uima/jcas/JCas.html), which provides generated getters and setters for the UIMA programs' types while still backing them with the same CAS.

In the incremental indexer this matters because the CAS is the single container that flows through normalization, delta gating, chunking, and embedding. Your `Doc` annotation with `docId` and `contentHash` lives in the CAS, your `Chunk` annotations with deterministic offsets live there too, and components can decide to skip unchanged documents by throwing `SkipCasException`. The same CAS can be serialized to [XMI](https://librarytechnology.org/document/8790) for debugging, then reloaded to reproduce a result exactly, which makes testing and grading straightforward. A `Doc` carries `docId`, `uri`, `title`, `language`, and `contentHash`, while a `Chunk` carries `chunkId`, `chunkIx`, `start`, `end`, `sectionPath`, `text`, and `contentHash`. The reader yields one CAS per file, the normalizer extracts text with Tika, sets language, computes a stable `docId` from the URI, and a `contentHash` from normalized text. Consider the following example.

```java
public class NormalizeAnnotator extends JCasAnnotator_ImplBase {
  @Override public void process(JCas jCas) throws AnalysisEngineProcessException {
    Doc d = JCasUtil.selectSingle(jCas, Doc.class);
    Path p = Paths.get(URI.create(d.getUri()));
    String raw = new Tika().parseToString(Files.newInputStream(p));
    String txt = normalizeWhitespace(raw);
    jCas.setDocumentText(txt);
    d.setLanguage(detectLanguage(txt));
    d.setDocId(sha256(d.getUri()));
    d.setContentHash(sha256(txt));
    d.setTitle(deriveTitle(txt, p.getFileName().toString()));
  }
}
```

Chunking is deterministic so edits only change local `chunkId`s, and embedding is delta aware so only chunks missing vectors for the current `(embedder, embedder_ver)` are sent in batches. Upserts keep writes idempotent, so retries never duplicate data. Consider the following example.

```sql
MERGE INTO embeddings AS t
USING (SELECT :chunk_id AS chunk_id, :ver AS embedder_ver, :vec AS vector) AS s
ON t.chunk_id = s.chunk_id AND t.embedder_ver = s.embedder_ver
WHEN MATCHED THEN UPDATE SET t.vector = s.vector, t.updated_at = CURRENT_TIMESTAMP
WHEN NOT MATCHED THEN INSERT (chunk_id, embedder_ver, vector, updated_at)
VALUES (s.chunk_id, s.embedder_ver, s.vector, CURRENT_TIMESTAMP);
```

```java
public class EmbedAnnotator extends JCasAnnotator_ImplBase {
  @ConfigurationParameter(name="embedder")     private String embedder;
  @ConfigurationParameter(name="embedderVer")  private String embedderVer;
  private DeltaStore store; private EmbedClient client;
  @Override public void initialize(UimaContext c){ store=DeltaStore.connect((String)c.getConfigParameterValue("dsn"));
    client=new HttpEmbedClient((String)c.getConfigParameterValue("embedUrl")); }
  @Override public void process(JCas jCas){
    var todo = JCasUtil.select(jCas, Chunk.class).stream()
      .filter(ch -> !store.hasEmbedding(ch.getChunkId(), embedder, embedderVer)).toList();
    for (var batch : batches(todo, 64)) {
      var vecs = client.embed(batch.stream().map(Chunk::getChunkId).toList(),
                              batch.stream().map(Chunk::getText).toList(),
                              embedder, embedderVer);
      for (var ch : batch) store.upsertEmbedding(ch.getChunkId(), ch.getContentHash(),
                                                 embedder, embedderVer, vecs.get(ch.getChunkId()));
    }
  }
}
```

The retrieval index is materialized by joining documents, chunks, and embeddings into a versioned snapshot, then published atomically. The aggregate engine wires reader, normalize, delta gate, chunk, embed, and a consumer that persists tables and triggers index export. Run once to build everything, then run again to confirm that unchanged inputs perform near zero work. Consider the following example.

```java
public class DeltaIndexer {
  public static void main(String[] a) throws Exception {
    var reader = CollectionReaderFactory.createReaderDescription(FileCollectionReader.class, "inputDir","./data/raw");
    var norm   = AnalysisEngineFactory.createEngineDescription(NormalizeAnnotator.class);
    var gate   = AnalysisEngineFactory.createEngineDescription(DeltaGateAnnotator.class, "dsn","jdbc:sqlite:./var/delta.db");
    var chunk  = AnalysisEngineFactory.createEngineDescription(SemanticChunker.class, "maxChars", 1200);
    var embed  = AnalysisEngineFactory.createEngineDescription(EmbedAnnotator.class,
                  "dsn","jdbc:sqlite:./var/delta.db","embedUrl","http://127.0.0.1:11434/api/embed",
                  "embedder","mxbai-embed-large","embedderVer","1.3.0");
    SimplePipeline.runPipeline(reader, norm, gate, chunk, embed);
  }
}
```
.

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your design and implemented components work to solve the problem, and the documentation that describe the build and runtime process, to be considered for grading. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are.

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from [the big brother](https://www.cs.uic.edu/~drmark/) :-) who is watching your exchanges. However, *you must not describe your mappers/reducers or the CORBA architecture or other specific details related to how you construct your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Monday, November, 3, 2025 at 8AM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters you chose for your experiments, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.


## Evaluation criteria
- the maximum grade for this homework is 15%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 15%-2% => 13%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic Spark or UIMA examples from some repos are given and nothing else is done: zero grade;
- using Python or Java or some other language instead of Scala: 8% penalty;
- homework submissions for an incorrectly chosen textbook assignment option will be desk-rejected with the grade zero;
- having less than five unit and/or integration scalatests: up to 10% lost;
- missing comments and explanations from your program with clarifications of your design rationale: up to 10% lost;
- logging is not used in your programs: up to 5% lost;
- hardcoding the input values in the source code instead of using the suggested configuration libraries: up to 5% lost;
- for each used *var* for heap-based shared variables or mutable collections without explicitly stated reasons: 0.3% lost;
- for each used *while* or *for* or other loops with induction variables to iterate over a collection: 0.5% lost;
- no instructions in README.md on how to install and run your program: up to 10% lost;
- the program crashes without completing the core functionality: up to 15% lost;
- the documentation exists but it is insufficient to understand your program design and models and how you assembled and deployed all components of your mappers and reducers: up to 15% lost;
- the minimum grade for this homework cannot be less than zero.

That's it, folks!