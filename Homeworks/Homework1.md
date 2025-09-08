# Homework 1
### The hands-on project for CS441 is divided into three homeworks to create and apply [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) to a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from [Ollama](https://ollama.com/search) using [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus). The first homework requires students to implement an LLM RAG index builder using massively parallel distributed computations in the cloud, the goal of the second homework is to explore and label clusters, detect anomalies, and train a lightweight reranker that improves what goes into the RAGed LLM using a neural network library as part of a cloud-based computational platform called Spark and the third, final homework is to create an LLM RAG-based generative system with a small swarm of Lambda-based AI agents that retrieve, synthesize, find gaps, and then propose novel MSR research ideas. Much of the background information is based on the books [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription and books on [A Simple Guide to Retrieval Augmented Generation by Abhinav Kimothi](https://www.manning.com/books/a-simple-guide-to-retrieval-augmented-generation) and [Essential GraphRAG by Tomaž Bratanič and Oskar Hane](https://www.manning.com/books/essential-graphrag). All images in this homework description are used from these book.

#### The goal of this homework is for students to gain experience with solving a distributed computational problem using cloud computing technologies. The main textbook group (option 1) will design and implement an instance of the map/reduce computational model using AWS EMR whereas the alternative textbook group (option 2) will use the CORBA model. You can check your textbook option in the corresponding column of the gradebook on the Blackboard.
#### Grade: 15%

## Preliminaries
As part of this first homework assignment students are going to learn how to create and manage Git project repository, create an application in Scala, create tests using widely popular Scalatest framework, and expand on the provided SBT build and run script for their applications. As a student in this course, your job is to learn how to use a Map/Reduce framework in the cloud environment. To make this homework interesting I choose to implement various technologies by applying massively parallel computations in the cloud, and each year I choose a different technology and the criteria for choosing a technology is its high market value in the current economy. That is, not only students learn cloud computing frameworks and mechanisms but they also increase their market value by learning a new hot technology for the current job market.

First things first, if you haven't done so, you must create your account at [Github](https://github.com/), a Git repo management system. Please make sure that you write your name in your README.md in your repo as it is specified on the class roster. Since it is a large class, please use your UIC email address for communications and for signing your projects and you should avoid using emails from other accounts like funnybunny2005@gmail.com. As always, the homeworks class' Teams channel is the preferred way to exchange information and ask questions. If you don't receive a response within a few hours, please contact your TA or the professor by tagging our names. If you use emails it may be a case that your direct emails went to the spam folder.

Next, if you haven't done so, you will install [IntelliJ](https://www.jetbrains.com/student/) with your academic license, the JDK, the Scala runtime and the IntelliJ Scala plugin and the [Simple Build Toolkit (SBT)](https://www.scala-sbt.org/1.x/docs/index.html) and make sure that you can create, compile, and run Java and Scala programs. Please make sure that you can run [various Java tools from your chosen JDK between versions 8 and 24](https://docs.oracle.com/en/java/javase/index.html).

In this and all consecutive homeworks you will use logging and configuration management frameworks. You will comment your code extensively and supply logging statements at different logging levels (e.g., TRACE, INFO, WARN, ERROR) to record information at some salient points in the executions of your programs. All input configuration variables/parameters must be supplied through configuration files -- hardcoding these values in the source code is prohibited and will be punished by taking a large percentage of points from your total grade! You are expected to use [Logback](https://logback.qos.ch/) and [SLFL4J](https://www.slf4j.org/) for logging and [Typesafe Conguration Library](https://github.com/lightbend/config) for managing configuration files. These and other libraries should be imported into your project using your script [build.sbt](https://www.scala-sbt.org). These libraries and frameworks are widely used in the industry, so learning them is the time well spent to improve your resumes. Also, you should set up your account with [AWS](https://aws.amazon.com/). Using your UIC email address may enable you to receive free credits for running your jobs in the cloud, but it is not guaranteed. Preferably, you should create your developer account for a small fee of approximately $29 per month to enjoy the full range of AWS services. Some students I know created business accounts to receive better options from AWS and some of them even started companies while taking this course using their AWS account and applications they created and hosted there!

From many example projects on Github you can see how to use Scala to create a fully functional (not imperative) implementation with subprojects and tests. As you see from the StackOverflow survey, knowledge of Scala is highly paid and in great demand, and it is expected that you pick it relatively fast, especially since it is tightly integrated with Java. I recommend using the book on [Programming in Scala Fourth and Fifth Editions by Martin Odersky et al](https://www.amazon.com/Programming-Scala-Fourth-Updated-2-13-ebook/dp/B082T2ZNJG). You can obtain this book using the academic subscription on [Safari Books Online](https://learning.oreilly.com/home-new/). There are many other books and resources available on the Internet to learn Scala. Those who know more about functional programming can use the book on [Functional Programming in Scala published in 2023 by Michael Pilquist, Rúnar Bjarnason, and Paul Chiusano](https://www.manning.com/books/functional-programming-in-scala-second-edition?new=true&experiment=C).

When creating your Map/Reduce in Scala or CORBA program code in C++ or Python you should avoid using **var**s and **while/for** loops that iterate over collections using [induction variables](https://en.wikipedia.org/wiki/Induction_variable). Instead, you should learn to use collection methods **map**, **flatMap**, **foreach**, **filter** and many others with lambda functions, which make your code linear and easy to understand as we studied it in class. Also, avoid mutable variables that expose the internal states of your modules at all cost. Points will be deducted for having unreasonable **var**s and inductive variable loops without explanation why mutation is needed in your code unless it is confined to method scopes - you can always do without using mutable states.

## Overview
All three homeworks are created under the general umbrella of a course project that allows students to create and train a RAG index for an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. The first phase of the project is to build an LLM RAG pipeline by preparing and sampling the input data and learning vector embeddings that is a term designating the conversion of input categorical text data into a vector format of the continuous real values and implement the attention mechanism for LLMs whereas the second phase involves implementing the attention mechanism with a training loop using Spark and evaluating the resulting model. In this homework, you will create a distributed program for parallel processing of the large corpus of data starting with vector embedding.

This and all future homework scripts are written using a retroscripting technique, in which the homework outlines are generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3Adg7IGPGYyKODxJgBwRT2bRKS0ig_u-IFqzOBkeXbuPo1%40thread.tacv2/conversations?groupId=01f0341d-bb02-4af6-8e0c-ad0b2a320a32&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

In the context of constructing an LLM, which is not what you are supposed to do in this course, a [tensor](https://en.wikipedia.org/wiki/Tensor) is a fundamental data structure used to represent multi-dimensional arrays of data. Tensors generalize matrices (2D arrays) to higher dimensions, and are key components in the field of machine learning, particularly in the implementation of neural networks, including LLMs like GPT models. The term [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is not sufficient since we need to work with data that has more than two dimensions. Matrices are limited to 2D arrays, meaning they have rows and columns. However, in many machine learning and deep learning applications like the one we are creating in this course, especially in neural networks like those used in LLMs, the data is inherently multi-dimensional, requiring more than just rows and columns to represent it. This is where the term tensor becomes necessary.

Consider an input to an LLM where we are processing a batch of sequences of token embeddings. This involves multiple dimensions:
* Batch size: The number of sequences (e.g., sentences or documents) processed simultaneously;
* Sequence length: The number of tokens (words or subwords) in each sequence;
* Embedding size: The dimensionality of the embedding for each token, representing its features (e.g., 512 or 1024 dimensions).

To represent this, we would need a 3D tensor:
* Batch size: N (the number of sequences in a batch);
* Sequence length: L (the number of tokens in each sequence);
* Embedding size: E (the dimensionality of each token's vector).

So, we would represent this data as a tensor of shape (N,L,E). For instance, if the following parameters are given:
* N=322 (32 sequences in the batch);
* L=128 (each sequence has 128 tokens);
* E=512 (each token is represented by a 512-dimensional vector).

Hence, we need a tensor of shape (32,128,512), which is a 3D tensor. This goes beyond what a matrix (which can only represent two dimensions) can handle. The term tensor is necessary to describe this multi-dimensional structure.

Next, token embeddings are computed, a fancy term for converting token IDs into embedding vectors. An embedding vector for a token in LLMs represents a dense, continuous, high-dimensional vector that encodes the semantic meaning of that token. It is part of the model's internal representation of words or tokens in a way that allows them to capture the relationships between words, such as their meanings, context, and syntactic roles. At this point, text is divided into smaller units called tokens. These can be individual words, subwords, or even characters, depending on the tokenization strategy. Then each token is mapped to a numerical vector through a process called embedding. The embedding vector typically has hundreds or thousands of dimensions, each representing different aspects of the token's meaning or usage in context. These embeddings are learned during the training process of the model. As the model processes vast amounts of text, it adjusts the embedding vectors so that tokens with similar meanings are placed closer together in this high-dimensional space. The embedding vector thus allows the model to handle tokens in a more context-sensitive way. Rather than treating each token as an isolated entity, the embedding provides a rich, nuanced representation that reflects the token's meaning within the model's learned knowledge.

For example, the words "king" and "queen" would have similar but distinct embeddings, capturing both their semantic similarity (royalty) and their differences (biological sexes). Similarly, context can modify the embedding of a word, so "bank" in "river bank" will have a different embedding than "bank" in "financial institution." If represented in a multidimensional space as vectors the words "king" and "queen" will be almost collinear or the cos(angle(vector(king), vector(queen)) will be close to 1. The steps are schematically represented in the image below.

Initially, when constructing an LLM all embeddings are given random values/weights and these weights will be updated later as part of the learning/training process using neural networks with the backpropagation learning algorithm. It is assumed that the vocabulary size of your dataset could be at least a few thousand words and the number of the output embedding dimensions can be determined experimentally, e.g., three dimensions may be a very small and unrealistic number whereas choosing millions of dimensions may be computationally prohibitive.

## Functionality
Your homework assignment is to create a program for parallel distributed processing of a large corpus of text. First things first, you should examine a dataset that consists of [hundreds of PDF papers published at the conference on mining software repositories (MSR)](https://github.com/0x1DOCD00D/CS441_Fall2025/tree/main/Homeworks/MSRCorpus), it is published under the directory MSRCorpus in this repository. A general goal is to create a program that processes these PDF files in parallel and produces a set of vector embeddings for the text in these files. The output of your program is a file with token embeddings and various statistics about the data. Since the goal of this homework is to learn how to use the Map/Reduce and its Hadoop implementation or the CORBA model and its OmniORB implementation, we are not striving to optimize the RAG pipeline or to achieve a certain precision in the learned values, so highly imprecise results of learning embeddings are Ok.

RAG pairs a retriever over this private MSR corpus of data with a generator, i.e., an LLM so answers are grounded in the MSR corpus PDFs, not in the model’s memory. With Ollama, we typically use an embedding model to convert text into vectors suitable for similarity search and store them in a vector database, e.g., [Pinecone](https://www.pinecone.io/), [Weaviate](https://weaviate.io/), [Chroma](https://www.trychroma.com/), [Lucene HNSW](https://lucene.apache.org/) or [FAISS](https://en.wikipedia.org/wiki/FAISS) and a chat/generation model to write the final answer while staying inside the retrieved context. Key points that make RAG attractive with Ollama include no fine-tuning, since you keep the original model; privacy, since your PDFs never leave the users' protected space, and modularity, since we can swap either the retriever or the generator without retraining. A practical bundle is [mxbai-embed-large (embeddings)](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) + [llama3.1:8b (chat)](https://ollama.com/library/llama3.1:8b), though students can choose others that fit their latency and hardware budgets.

For this homework [Lucene]((https://lucene.apache.org) is recommended to store chunks of the texts and their embeddings. Then, for a new question submitted to an AI agent (that is essentially a front end to your RAGged model), it finds the most relevant chunks. It does not expand tokens or generate text, but instead your AI agent takes those retrieved chunks, stitches them into a context block, and send that as part of the prompt to your chosen Ollama chat model. Ollama then tokenizes that prompt and generates the answer. That is, the Ollama model stays untouched. RAG sits outside the model and augments the prompt with text we pulled from the created Lucene HNSW index. One can view this process of submitting a query to the Lucene index that returns chunks of text for the keywords in the query, using IR jargon as ***query expansion*** that often means adding keywords/synonyms to the search query. In RAG this process is called ***context augmentation***; that is, full text chunks are retrieved and put them into the model prompt. No weights are trained or changed in the Ollama model. The model is used as-is, and it generates answers using the context you provide. Key points are that Lucene doesn’t generate or expand tokens, it just returns the most relevant chunks and in your program those chunks are assembled into the prompt that are sent to Ollama. The model then generates answers using that supplied context. 

To summarize the workflow for this homework includes the following large-granularity steps that can be summarized as 1) choose a model, 2) RAG it, and 3) deploy on AWS. To use the RAGged Ollama model you should do the following.
* Step 1: pick models and constraints to embed query with the same embedding model you indexed with. Select the embedding model (vector dimension, speed, license) and the generator (context window, quality, VRAM/CPU). Decide cosine vs dot-product vs L2 for similarity, then be consistent.
* Step 2: build the corpus by extracting text from PDFs, chunking, embedding, and indexing. For chunking use 800–1,800 chars with 10–20% overlap, then batch-embed (vector, text, doc_id, chunk_id, hash, timestamp) with Ollama and store in a vector DB, L2-normalize vectors if using cosine/IP, and build an ANN index.
* Step 3: serve queries! At query time embed the question, search the index for top-k, pack a context block, then call the generator with a strict instruction to answer only from the provided context, [k-NN search in Lucene HNSW](https://docs.lucenia.io/search-plugins/knn/filter-search-knn/) → get top-k chunks (+ scores, doc_ids).
* Step 4: deploy on AWS. Put the index artifacts (Lucene directories or FAISS files) in S3 buckets, at boot, copy to local EBS/NVMe for speed, run Ollama + your small RAG API alongside it on the same instance. Choose one of the following architectures: EC2 for the simplest path; ECS with an Ollama sidecar and your API container behind an ALB; or (for advanced students), EKS if you already standardize on Kubernetes (Ollama on GPU nodes; API as a Deployment). You can use IAM roles for S3 read, keep Ollama private inside the VPC, front only your API with TLS and auth. 
* Step 5: (Optional) add quality switches: deduplicate near-identical chunks by hash, add a tiny reranker over the top-k, enforce “answer only from context” in the system message, and truncate context to fit the generator’s window.

First, you split the initial text corpus in shards for parallel processing. The size of the shard can be chosen experimentally. Next, for each shard you will convert the text into numerical tokens using a chosen Ollama model. Assume OLLAMA_HOST points to your server, for example export OLLAMA_HOST=http://127.0.0.1:11434. Consider the following SBT dependencies that you should include in your build.sbt file to use Ollama models and other required libraries.
```scala
ThisBuild / scalaVersion := "3.5.1"
libraryDependencies ++= Seq(
  // Retrieval index (pure JVM, cross-platform)
  "org.apache.lucene" % "lucene-core" % "9.10.0",
  "org.apache.lucene" % "lucene-analysis-common" % "9.10.0",
  // PDF extraction + HTTP + JSON
  "org.apache.pdfbox" % "pdfbox" % "2.0.31",
  "com.softwaremill.sttp.client3" %% "core"  % "3.9.5",
  "com.softwaremill.sttp.client3" %% "circe" % "3.9.5",
  "io.circe" %% "circe-generic" % "0.14.9",
  "io.circe" %% "circe-parser"  % "0.14.9",
  "ch.qos.logback" % "logback-classic" % "1.5.6"
)
// Optional FAISS CPU via JNI (Linux-only, CPU-only)
// libraryDependencies += "com.criteo.jfaiss" % "jfaiss-cpu" % "1.7.0-1"
```
You can use the following Scala-like pseudocode to call Ollama models from your Scala program to compute tokens. You can find more details about Ollama API at [Ollama API documentation](https://deepwiki.com/ollama/ollama). Some examples are available at DrMark's PLANE GitHub repo [Ollama Scala Client](https://github.com/0x1DOCD00D/PLANE/tree/master/src/main/scala/LLMs).
```scala
import sttp.client3.*
import sttp.client3.circe.*
import io.circe.*, io.circe.generic.semiauto.*

final case class EmbedReq(model: String, input: Vector[String])
final case class EmbedResp(embeddings: Vector[Vector[Float]])
object EmbedResp:
given Decoder[EmbedResp] =
  Decoder.instance { c =>
    c.downField("embeddings").as[Vector[Vector[Float]]].map(EmbedResp.apply)
      .orElse(c.downField("embedding").as[Vector[Float]].map(v => EmbedResp(Vector(v))))
  }
final case class ChatMessage(role: String, content: String)
final case class ChatReq(model: String, messages: Vector[ChatMessage], stream: Boolean = false)
final case class ChatMsg(role: String, content: String)
final case class ChatResp(message: ChatMsg)
object ChatResp:
given Decoder[ChatMsg]  = deriveDecoder
given Decoder[ChatResp] = deriveDecoder

class Ollama(base: String = sys.env.getOrElse("OLLAMA_HOST","http://127.0.0.1:11434")):
private val be   = HttpClientSyncBackend()
private val eurl = uri"$base/api/embeddings"
private val curl = uri"$base/api/chat"

def embed(texts: Vector[String], model: String): Vector[Array[Float]] =
val req = basicRequest.post(eurl).body(EmbedReq(model, texts)).response(asJson[EmbedResp])
req.send(be).body.fold(throw _, _.embeddings.map(_.toArray))

def chat(messages: Vector[ChatMessage], model: String): String =
val req = basicRequest.post(curl).body(ChatReq(model, messages)).response(asJson[ChatResp])
req.send(be).body.fold(throw _, _.message.content)
```
Next, you need to compute the sliding window data samples with the input shifted by some number bigger than one as shown in the example image below. For RAG chunking with a Lucene HNSW index, do not slide by one token/character; instead, use a fixed window with a moderate overlap, not a one-step shift. Keep in mind that the goal of this step is to create chunks that are big enough to carry meaning, yet small enough to index and recall accurately.
* Good practice: pick a window size W and an overlap O, then use stride S = W − O.
* Number of chunks N = ceil((L − W)/S) + 1 for a text of length L.
* Typical values: characters: W ≈ 800–1,800 with O ≈ 10–25% of W; tokens: W ≈ 256–768 with O ≈ 64–128.
* Why not shift by one or some other very small number: massive duplication and cost with almost no recall gain. For example: a 100k-char doc, W=1,000, O=200 → S=800 gives about N = ceil((100,000 − 1,000)/800) + 1 = 125 chunks. A shift-by-1 makes ~99,001 chunks. Index size, embedding calls, and latency explode.

The sliding window algorithm in the context of LLMs works to handle long texts that exceed the model's maximum token limit by processing the text in chunks. Here's how it typically functions:
1. Define the Window Size: The model has a maximum number of tokens it can process in one go, say 1024 tokens.
2. Initialize the Window: The first window covers the first 1024 tokens of the input text.
3. Process the Initial Window: The model processes this window to generate output or extract information.
4. Sliding the Window: After processing, the window "slides" forward by a certain number of tokens, typically with some overlap to maintain context. For example, it might move forward by 512 tokens, meaning the next window will cover tokens 513 to 1536.
5. Repeat Until End: This process continues until the window reaches the end of the text.

If the text length is shorter than the maximum token limit, or when the sliding window reaches the end of the text, there are no more tokens to fill the window. Here's how it works.
1. Incomplete Final Window: If there aren't enough tokens left to fill the window completely, the final window may contain fewer tokens than the initial window size. The model processes whatever data remains in this window.
2. Stop Condition: The algorithm detects that there are no more tokens to include in the next window and stops processing. This prevents any further sliding or unnecessary computations.
3. Handling Edge Cases: In cases where the last chunk of text is crucial for context (like summarization or continuation tasks), the overlap between windows ensures that the model retains context from the previous segment, even when the last window is smaller.
   
In essence, when there is no more data to slide, the algorithm concludes its processing as it has covered the entire input text.

Below is an example of Scala-like pseudocode with mutable variables to illustrate the point that specifies the steps of the process of reading text from PDF files and chunking the text into smaller pieces. Example Scala program for extracting text from PDF files is given at DrMark's PLANE GitHub repo [PDF text extraction](https://github.com/0x1DOCD00D/PLANE/blob/master/src/main/scala/pdfWorkshop/loadPdfExtractText.scala).
```scala
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import java.io.ByteArrayInputStream
import java.nio.file.{Files, Path}

object Pdfs:
  def readText(p: Path): String =
    val bytes = Files.readAllBytes(p)
    val doc   = PDDocument.load(new ByteArrayInputStream(bytes))
    try PDFTextStripper().getText(doc) finally doc.close()

object Chunker:
  def normalize(s: String): String = s.replaceAll("\\s+"," ").trim
  def split(s: String, maxChars: Int = 1800, overlap: Int = 250): Vector[String] =
    val clean = normalize(s)
    val out   = Vector.newBuilder[String]
    var i = 0
    while i < clean.length do
      val end   = (i + maxChars).min(clean.length)
      val slice = clean.substring(i, end)
      val cut   = slice.lastIndexWhere(ch => ch == '.' || ch == '\n')
      val piece = if cut >= (maxChars * 0.6).toInt then slice.substring(0, cut + 1) else slice
      out += piece
      i += (piece.length - overlap).max(1)
    out.result()
```

To experiment, you can start with the monolithic build that loops over PDFs on one machine. Converting to Map/Reduce makes the build scalable and reproducible with a partitioning rule that assigns each document to a shard. For R shards, you can use the following simple rule to assign a document with id doc_id to a shard with id shard_id in [0,R-1]: choose a stable shard id, for example abs(hash(doc_id)) % R. Mappers implement the following pipeline: extract → chunk → embed → emit where the mapper receives a PDF path, emits key = shardId, value = JSON or Yaml record with text and embedding. An example Scala-like pseudocode for the mapper is given below.
```scala
// value: line with absolute pdf path
class RagMapper extends Mapper[LongWritable, Text, IntWritable, Text]:
  private val client = Ollama()
  override def map(_: LongWritable, v: Text, ctx: Mapper[LongWritable,Text,IntWritable,Text]#Context): Unit =
    val path   = Paths.get(v.toString)
    val docId  = path.getFileName.toString
    val text   = Pdfs.readText(path)
    val chunks = Chunker.split(text)
    val vecs   = client.embed(chunks, "mxbai-embed-large").map(Vectors.l2)
    val shard  = math.abs(docId.hashCode) % ctx.getNumReduceTasks
    chunks.zip(vecs).zipWithIndex.foreach { case ((c, e), id) =>
      val rec = s"""{"doc_id":"$docId","chunk_id":$id,"text":${encode(c)},"vec":[${e.mkString(",")}]}"""
      ctx.write(new IntWritable(shard), new Text(rec))
    }
  private def encode(s: String) = "\"" + s.replace("\\","\\\\").replace("\"","\\\"") + "\""
```

To summarize, the input data is split across multiple shards and the mapper that implements the embedding process is applied to every shard in parallel to produce the resulting vector embeddings that are combined in reducers. Based on *your design* of mappers and reducers you partition the input text corpus into shards keeping their relative position in the text corpus, i.e., if the text document is split into two shards and then the mappers tokenize each shard and create a Lucene or FAISS index and then the reducers should produce combine separately produced indeces. The reducer therefor collects the output from mappers and combines the index vectors is given below as an example of Scala-like pseudocode.
```scala
class ShardReducer extends Reducer[IntWritable, Text, Text, Text]:
  override def reduce(key: IntWritable, values: java.lang.Iterable[Text],
                      ctx: Reducer[IntWritable,Text,Text,Text]#Context): Unit =
    val shard   = key.get
    val local   = Files.createTempDirectory(s"lucene-shard-$shard")
    val iw      = IndexWriter(FSDirectory.open(local), IndexWriterConfig(StandardAnalyzer()))
    import io.circe.parser.*
    values.forEach { t =>
      val json = t.toString
      val rec  = parse(json).toOption.get.hcursor
      val doc  = Document()
      doc.add(StringField("doc_id",   rec.get[String]("doc_id").toOption.get, Field.Store.YES))
      doc.add(StringField("chunk_id", rec.get[Int]("chunk_id").toOption.get.toString, Field.Store.YES))
      doc.add(TextField("text",       rec.get[String]("text").toOption.get, Field.Store.YES))
      val vec  = rec.get[Vector[Float]]("vec").toOption.get.toArray
      doc.add(new KnnFloatVectorField("vec", vec, VectorSimilarityFunction.COSINE))
      iw.addDocument(doc)
    }
    iw.commit(); iw.close()

    // copy local dir -> HDFS: <job-output>/index_shard_<shard>
    // (use Hadoop FileSystem API here)
 ```   
Your service loads all index_shard_*, runs KnnFloatVectorQuery on each shard in parallel, then merges results into global top-k via a min-heap. This is the standard “fan-out/fan-in” combine at query time and avoids an expensive global index merge. For a single index, as a final step, open a fresh IndexWriter and call addIndexes(dirA, dirB, …) on all shard directories; ensure the vector field name, dimension and similarity match. Note that Lucene doc IDs can change after merges, so rely on your stored doc_id/chunk_id for references.

Your RAGged Ollama model can be used locally and on AWS. For local usage you should do the following.
1.	Build the index with BuildLucene.run("pdfs","lucene-index","mxbai-embed-large") or its equivalents.
2.	Ask questions with AskLucene.answer("What does section 3 in the paper on inconsistency claim about consistency?").
3.	The loop is: embed query → search Lucene → pack context → ollama /api/chat.

Optional quality switches: deduplicate near-identical chunks by hash, add a tiny reranker over the top-k, enforce “answer only from context” in the system message, and truncate context to fit the generator’s window.

For AWS deployment, you can create lucene-index/ or shard directories index_shard_* in S3 and keep meta.jsonl only if you store extra metadata beyond what’s already stored in Lucene as fields. To host the model EC2 can be used with one VM with GPU (g5/g6) or CPU (c7i/c8g for small models).

Your other goal is to produce files with token embeddings and various statistics about the data. First, you will compute a Yaml or an CSV file that shows the vocabulary as the list of words, their numerical tokens and the frequency of occurences of these words in the text corpus. Second, for each token embedding you can output other tokens/words that are semantically close them it based on the computed vector embeddings. Finally, you will produce an estimate of how well your embeddings capture semantic relationships between words using the following two tasks. For these tasks, you can use a small set of predefined word pairs and triplets that are known to be semantically related.
* Word Analogy: Check how well your embeddings capture relationships like "king" - "man" + "woman" ≈ "queen".
* Word Similarity: Evaluate cosine similarity between known pairs of words (e.g., "cat" and "dog") and see if their embeddings capture the right degree of similarity.

### Assignment for the main textbook group
Your job is to create the mapper and the reducer for each task, explain how they work, and then to implement them and run on the text corpus using your predefined configuration parameters. The output of your map/reduce is a Yaml or an CSV file with token embeddings and the required statistics. The explanation of the map/reduce model is given in the main textbook and covered in class lectures.

You will create and run your software application using [Apache Hadoop](http://hadoop.apache.org/), a framework for distributed processing of large data sets across multiple computers (or even on a single node) using the map/reduce model. Next, after creating and testing your map/reduce program locally, you will deploy it and run it on the Amazon Elastic MapReduce (EMR) - you can find plenty of [documentation online](https://aws.amazon.com/emr). You will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](www.youtube.com) and you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or Zoom or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera. The display of your passwords and your credit card numbers should be avoided when possible :-).

### Assignment for the alternative textbook group
Your job is to create the distributed objects using [omniOrb CORBA framework](http://omniorb.sourceforge.net/omni42/omniORB/) for each task, explain how they work, and then to implement them and run on the generated log message dataset. The output of your distributed system is a Yaml or an CSV file with the required statistics. The explanation of the CORBA is given in the alternative textbook in Chapter 7 -Guide to Reliable Distributed Systems: Building High-Assurance Applications and Cloud-Hosted Services 2012th Edition by Kenneth P. Birman. You can complete your implementation using C++ or Python.

Next, after creating and testing your program locally, you will deploy it and run it on the AWS EC2 IaaS. You will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](www.youtube.com) and you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera.

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your mapper and reducer work to solve the problem for Option 1 group or how your CORBA distributed object work for Option 2 group, and the documentation that describe the build and runtime process, to be considered for grading. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are.

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from [the big brother](https://www.cs.uic.edu/~drmark/) :-) who is watching your exchanges. However, *you must not describe your mappers/reducers or the CORBA architecture or other specific details related to how you construct your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Thursday, October, 9, 2025 at 10PM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters you chose for your experiments, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.


## Evaluation criteria
- the maximum grade for this homework is 15%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 15%-2% => 13%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic map/reduce or CORBA examples from some repos are given and nothing else is done: zero grade;
- using Python or some other language instead of Scala for the M/R option: 8% penalty;
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