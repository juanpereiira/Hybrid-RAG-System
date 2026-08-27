#IMPORTS
#----------------------------------------------------------------
print("Importing libraries...")
from huggingface_hub import InferenceClient
import os
import warnings

warnings.filterwarnings("ignore")
from getpass import getpass

import numpy as np
import pandas as pd
import faiss
import re

from collections import Counter

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from smolagents import CodeAgent, tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Libraries imported successfully!")

#KNOWLEDGE BASE CREATION
#----------------------------------------------------------------
print("\n Creating knowledge base...")

documents = [
    {
        "id": "AI001",
        "title": "Artificial Intelligence",
        "content": """
        Artificial Intelligence is the field of building computer systems
        capable of performing tasks that normally require human intelligence.
        AI applications include prediction, classification, recommendation,
        natural language processing, computer vision and decision support.
        """
    },

    {
        "id": "LLM001",
        "title": "Large Language Models",
        "content": """
        Large Language Models are neural network models trained on large
        datasets. They process text using tokens and generate responses by
        predicting subsequent tokens. LLMs can perform summarisation,
        translation, question answering, code generation and information
        extraction.
        """
    },

    {
        "id": "RAG001",
        "title": "Retrieval Augmented Generation",
        "content": """
        Retrieval Augmented Generation, commonly called RAG, combines
        information retrieval with language generation. A user question is
        converted into a search query, relevant information is retrieved
        from a knowledge base, and the retrieved information is supplied to
        a language model as context.
        """
    },

    {
        "id": "EMB001",
        "title": "Embeddings",
        "content": """
        Embeddings are numerical representations of text. An embedding model
        converts sentences, paragraphs or documents into vectors. Texts with
        similar meanings tend to have similar vector representations.
        Embeddings are commonly used for semantic search, clustering,
        recommendations and retrieval systems.
        """
    },

    {
        "id": "CHUNK001",
        "title": "Document Chunking",
        "content": """
        Chunking divides large documents into smaller pieces before they are
        converted into embeddings. Chunking allows retrieval systems to find
        focused passages rather than retrieving an entire document.
        Chunk size and overlap are important parameters. Very small chunks
        can lose context, while very large chunks can reduce retrieval
        precision.
        """
    },

    {
        "id": "VECTOR001",
        "title": "Vector Search",
        "content": """
        Vector search retrieves documents by comparing the vector embedding
        of a query with the embeddings stored in a vector index. Similarity
        can be calculated using cosine similarity, dot product or other
        distance measures.
        """
    },

    {
        "id": "HYBRID001",
        "title": "Hybrid Search",
        "content": """
        Hybrid search combines keyword search and vector search. Keyword
        search is useful for exact terms, names, identifiers and technical
        expressions. Vector search is useful for finding semantically
        similar information even when the wording is different.
        """
    },

    {
        "id": "EVAL001",
        "title": "RAG Evaluation",
        "content": """
        RAG applications should be evaluated for retrieval relevance,
        correctness, groundedness, completeness, latency and cost.
        Retrieval quality is important because a language model can only
        reliably use information that is supplied to it correctly.
        """
    }
]

df = pd.DataFrame(documents)
print("\n",df,"\n")
print("Knowledge base created successfully!")

#CONVERT TO LANGCHAIN DOCS
#----------------------------------------------------------------
source_documents = []
print("Langchain Doc Conversion")
for item in documents:
    doc = Document(
        page_content=item["content"],
        metadata={
            "id": item["id"],
            "title": item["title"]
        }
    )
    source_documents.append(doc)

print("Number of documents created:", len(source_documents))

print("")


#CHUNKING
#----------------------------------------------------------------
print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(source_documents)

print("Original documents:", len(source_documents))
print("Generated chunks:", len(chunks))
print("")
for i, chunk in enumerate(chunks, start=1):

    print("=" * 80)

    print("Chunk:", i)

    print(
        "Source:",
        chunk.metadata["id"],
        "-",
        chunk.metadata["title"]
    )

    print()

    print(chunk.page_content)
print("\n"+"x-x-" * 40 + "x\n")

#LOAD OPEN SOURCE EMBEDDING MODEL
#----------------------------------------------------------------
print("Loading embedding model...")
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    embedding_model_name
)

print("Embedding model:", embedding_model_name)

print(
    "Embedding dimensions:",
    embedding_model.get_embedding_dimension()
)
print("Embedding model loaded successfully!\n")

#GENERATE EMBEDDINGS
#----------------------------------------------------------------
print("Generating embeddings for chunks...")
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = embedding_model.encode(
    chunk_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Embedding shape:", embeddings.shape)
print("Embeddings generated successfully!\n")

print("First chunk:")

print(chunks[0].page_content)

print("\nEmbedding:")
print(embeddings[0])

print("\nEmbedding dimension:",len(embeddings[0]))
print("")

#FAISS INDEX CREATION
#----------------------------------------------------------------

print("Creating FAISS index...")

dimension = embeddings.shape[1]

vector_index = faiss.IndexFlatIP(
    dimension
)

vector_index.add(
    embeddings
)

print(
    "Number of vectors:",
    vector_index.ntotal
)
print("FAISS index created successfully!\n")
#VECTOR RETRIEVAL FUNCTION and TESTING
#----------------------------------------------------------------

def vector_search(query, top_k=10):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = vector_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        chunk = chunks[int(index)]

        results.append({
            "score": float(score),
            "id": chunk.metadata["id"],
            "title": chunk.metadata["title"],
            "text": chunk.page_content
        })

    return results

#Testing
query = "How can an AI system find information with similar meaning?"

results = vector_search(
    query,
    top_k=10
)

for result in results:

    print("=" * 80)

    print(
        "Similarity:",
        round(result["score"], 4)
    )

    print(
        "Source:",
        result["title"]
    )

    print(
        result["text"]
    )

#KEYWORD SEARCH USING BM25
#----------------------------------------------------------------

#Index Creation
print("\nCreating BM25 index...")
tokenized_documents = [
    re.findall(
        r"\b[a-zA-Z]+\b",
        chunk.page_content.lower()
    )
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_documents
)

print("BM25 index created!\n")

#Keyword Retrieval Function and Testing
#func
def keyword_search(query, top_k=3):

    query_tokens = re.findall(
        r"\b[a-zA-Z]+\b",
        query.lower()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "score": float(scores[index]),
            "id": chunks[index].metadata["id"],
            "title": chunks[index].metadata["title"],
            "text": chunks[index].page_content
        })

    return results

#Testing
print("Testing keyword search with BM25...\n")
query = "embedding vector semantic search"

results = keyword_search(
    query,
    top_k=3
)

for result in results:

    print("=" * 80)

    print(
        "BM25 Score:",
        round(result["score"], 4)
    )

    print(
        "Source:",
        result["title"]
    )

    print(
        result["text"]
    )
print("")

#COMPARE VECTOR VS KEYWORD SEARCH
#----------------------------------------------------------------

print("Comparing vector search and keyword search...\n")
query = "How does semantic similarity help retrieval?"

print("\nVECTOR SEARCH")
print("=" * 80)

for r in vector_search(query):

    print(round(r["score"], 3),"→",r["title"])

print("\nKEYWORD SEARCH")
print("=" * 80)

for r in keyword_search(query):

    print(round(r["score"], 3),"→",r["title"])

print("\nComparison complete!\n")

#HYBRID SEARCH
#----------------------------------------------------------------

#Normalize Scores
def normalize_scores(scores):

    scores = np.array(
        scores,
        dtype=float
    )

    if np.max(scores) == np.min(scores):

        return np.ones(
            len(scores)
        )

    return (
        (scores - np.min(scores))
        /
        (np.max(scores) - np.min(scores))
    )

#Retrieval Function 

def hybrid_search(
    query,
    top_k=3,
    vector_weight=0.7,
    keyword_weight=0.3
):

    vector_results = vector_search(
        query,
        top_k=6
    )

    keyword_results = keyword_search(
        query,
        top_k=6
    )

    vector_scores = normalize_scores(
        [r["score"] for r in vector_results]
    )

    keyword_scores = normalize_scores(
        [r["score"] for r in keyword_results]
    )

    combined = {}

    # Add vector results

    for i, result in enumerate(
        vector_results
    ):

        combined[result["id"]] = {

            **result,

            "vector_score":
                vector_scores[i],

            "keyword_score":
                0
        }

    # Add keyword results

    for i, result in enumerate(
        keyword_results
    ):

        if result["id"] not in combined:

            combined[result["id"]] = {

                **result,

                "vector_score":
                    0,

                "keyword_score":
                    keyword_scores[i]
            }

        else:

            combined[
                result["id"]
            ]["keyword_score"] = keyword_scores[i]

    # Calculate final score

    for result in combined.values():

        result["hybrid_score"] = (

            vector_weight *
            result["vector_score"]

            +

            keyword_weight *
            result["keyword_score"]
        )

    results = sorted(
        combined.values(),
        key=lambda x:
            x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]

#Testing

query = "How does a system find information based on meaning?"

results = hybrid_search(
    query,
    top_k=3
)
print("Final Comparison with Hybrid Testing Score")
for result in results:

    print("=" * 80)

    print(
        "Title:",
        result["title"]
    )

    print(
        "Vector score:",
        round(
            result["vector_score"],
            3
        )
    )

    print(
        "Keyword score:",
        round(
            result["keyword_score"],
            3
        )
    )

    print(
        "Hybrid score:",
        round(
            result["hybrid_score"],
            3
        )
    )
print("")

#CREATING and TESTING RETRIEVER TOOL

def retrieve_context(query):

    results = hybrid_search(
        query,
        top_k=4
    )

    context = ""

    # Keep track of documents already added
    added_documents = set()

    for i, result in enumerate(results, start=1):

        document_id = result["id"]

        # Skip duplicate documents
        if document_id in added_documents:
            continue

        added_documents.add(document_id)

        # Find the original document
        original_document = next(
            doc for doc in documents
            if doc["id"] == document_id
        )

        context += f"""
SOURCE {i}
Document ID: {original_document["id"]}
Title: {original_document["title"]}
Hybrid Score: {result["hybrid_score"]:.3f}

Content:
{original_document["content"]}

-------------------------
"""

    return context

#ADDING HUGGING FACE API KEY

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file.")

print("HF_TOKEN loaded successfully.")

#CREATE RETRIEVAL TOOL

@tool #decorator tells the agent: Make this python function available as a tool that the agent can decide to call.
def knowledge_base_search(query: str) -> str:
    """
    Search the enterprise knowledge base.

    Args:
        query: A natural-language question or search query.

    Returns:
        Relevant passages from the knowledge base.
    """

    #triple quotes create a multiline string.
    #Agent uses the tool desc to understand what the tool does and when it should use it.
    return retrieve_context(query)

#Connect an OPEN WEIGHT LLM

from smolagents.agents import ChatMessage
from transformers import pipeline, AutoTokenizer, logging

logging.set_verbosity_error()
import os


class HFPipelineWrapper:

    def __init__(self, hf_pipeline):

        self.pipeline = hf_pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_pipeline.model.name_or_path,
            trust_remote_code=True
        )

    def generate(self, messages, **kwargs):

        formatted_messages = []

        # System instruction
        system_message = """
        You are a knowledge base assistant.

        Answer questions ONLY using the provided knowledge base.

        STRICT RULES:
        1. Use only the information provided in the knowledge base.
        2. Do not use web search.
        3. Do not use Wikipedia or external sources.
        4. Do not use your general knowledge.
        5. Do not invent information.
        6. If the knowledge base does not contain enough information,
        say:
        "The knowledge base does not contain enough information
        to answer this question."
        7. Give a concise, natural-language answer.
        """

        formatted_messages.append({
            "role": "system",
            "content": system_message
        })

        # Convert smolagents messages
        for msg in messages:

            if isinstance(msg, ChatMessage):

                content = msg.content

                # IMPORTANT:
                # smolagents may provide content as a list
                if isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if isinstance(item, dict):

                            if "text" in item:
                                text_parts.append(
                                    str(item["text"])
                                )

                            elif "content" in item:
                                text_parts.append(
                                    str(item["content"])
                                )

                        else:
                            text_parts.append(
                                str(item)
                            )

                    content = "\n".join(text_parts)

                else:
                    content = str(content)

                formatted_messages.append({
                    "role": msg.role,
                    "content": content
                })

            elif isinstance(msg, dict):

                content = msg.get("content", "")

                # Handle list content
                if isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if isinstance(item, dict):

                            if "text" in item:
                                text_parts.append(
                                    str(item["text"])
                                )

                            elif "content" in item:
                                text_parts.append(
                                    str(item["content"])
                                )

                        else:
                            text_parts.append(
                                str(item)
                            )

                    content = "\n".join(text_parts)

                else:
                    content = str(content)

                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })

        # Convert messages to Qwen chat format
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generation parameters
        generation_params = {
            "max_new_tokens": kwargs.get(
                "max_tokens",
                128 #max number of new tokens qwen should generate
            ),
            "temperature": kwargs.get(
                "temperature",
                0.2 #controls randomness
            ),
            "do_sample": True,
            "return_full_text": False
        }

        # Generate response
        result = self.pipeline(
            prompt,
            **generation_params
        )

        generated_text = result[0]["generated_text"]

        # Make absolutely sure the output is a string
        if isinstance(generated_text, list):

            text_parts = []

            for item in generated_text:

                if isinstance(item, dict):
                    text_parts.append(
                        str(item.get("content", item.get("text", "")))
                    )
                else:
                    text_parts.append(
                        str(item)
                    )

            generated_text = "\n".join(text_parts)

        generated_text = str(generated_text).strip()

        # Remove Qwen end token
        generated_text = generated_text.replace(
            "<|im_end|>",
            ""
        ).strip()

        return ChatMessage(
            role="assistant",
            content=generated_text
        )


# Create Hugging Face pipeline

hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    token=os.environ["HF_TOKEN"],
    trust_remote_code=True
)

# Wrap pipeline for smolagents

model = HFPipelineWrapper(
    hf_pipeline
)

print("Open-weight LLM connected successfully!")

# ============================================================
# RAG ASSISTANT
# ============================================================

def ask_assistant(question):

    # Retrieve information from the knowledge base
    context = retrieve_context(question)

    messages = [

        {
            "role": "system",
            "content": """
You are a knowledge base assistant.

Answer questions ONLY using the provided knowledge base.

STRICT RULES:
1. Use only the information provided in the knowledge base.
2. Do not use web search.
3. Do not use Wikipedia or external sources.
4. Do not use your general knowledge.
5. Do not invent information.
6. If the knowledge base does not contain enough information,
   say:
   "The knowledge base does not contain enough information
   to answer this question."
7. Give a concise, natural-language answer.
"""
        },

        {
            "role": "user",
            "content": f"""
Knowledge Base Context:

{context}

Question:

{question}

Answer using ONLY the knowledge base context.
"""
        }

    ]

    # Convert messages to Qwen format
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate answer
    result = hf_pipeline(
        prompt,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=True,
        return_full_text=False
    )

    answer = result[0]["generated_text"]

    # Clean output
    answer = str(answer).strip()

    answer = answer.replace(
        "<|im_end|>",
        ""
    ).strip()

    return answer

def evaluate_session(results):

    prompt = f"""
Evaluate this RAG session using ONLY the provided context and answers.

Give scores from 1 to 5 for:

- Retrieval Relevance
- Answer Correctness
- Groundedness
- Completeness
- Overall

Return ONLY:
Retrieval Relevance: X/5
Answer Correctness: X/5
Groundedness: X/5
Completeness: X/5
Overall: X/5

Session:
{results}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict RAG evaluator. Use only the provided session information. Do not use external knowledge."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    result = hf_pipeline(
        prompt,
        max_new_tokens=64,
        temperature=0.1,
        do_sample=False,
        return_full_text=False
    )

    return str(result[0]["generated_text"]).strip()
# ============================================================
# INTERACTIVE TESTING
# ============================================================

session_results = []

print("\n" + "=" * 70)
print("SYSTEM READY")
print("Ask questions about the knowledge base.")
print("Type 'exit' or 'quit' to stop.")
print("=" * 70)

while True:

    question = input("\nYou: ")

    if question.strip().lower() in ["exit", "quit"]:
        break

    if not question.strip():
        continue

    context = retrieve_context(question)
    answer = ask_assistant(question)

    print("\nAssistant:")
    print(answer)

    session_results.append({
        "Question": question,
        "Context": context,
        "Answer": answer
    })


# ============================================================
# SESSION EVALUATION
# ============================================================

if session_results:

    print("\n" + "=" * 70)
    print("SESSION EVALUATION")
    print("=" * 70)

    evaluation = evaluate_session(session_results)

    print(evaluation)

else:
    print("\nNo questions were asked.")