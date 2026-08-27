

A Retrieval-Augmented Generation (RAG) system for knowledge-base question answering using hybrid retrieval, combining dense vector search and sparse BM25 search with an open-weight language model.

## Overview

This project implements a knowledge-base question answering system that retrieves relevant information before generating an answer.

The system combines:

- Dense semantic retrieval using Sentence Transformers
- Vector search using FAISS
- Sparse keyword retrieval using BM25
- Hybrid score-based retrieval
- Document chunking using LangChain
- Agentic retrieval using Smolagents
- Open-weight LLM inference using Qwen2.5-3B-Instruct
- LLM-based self-evaluation

The assistant is restricted to the provided knowledge base and is designed not to use web search, Wikipedia, or other external sources.

## System Architecture

```text
                                                  User Question
                                                        │
                                                        ▼
                                                  Knowledge Base
                                                        │
                                                        ▼
                                                  Document Chunking
                                                        │
                                                ├───────────────|
                                                ▼               ▼
                                          Dense Retrieval    BM25 Retrieval
                                                │               │
                                                └───────┬───────┘
                                                        ▼
                                                  Hybrid Ranking
                                                        │
                                                        ▼
                                                Retrieved Context
                                                        │
                                                        ▼
                                                 Qwen2.5-3B-Instruct
                                                        │
                                                        ▼
                                                     Answer
                                                        │
                                                        ▼
                                                 Self-Evaluation
```
## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FAISS | Vector search |
| BM25 | Keyword retrieval |
| Sentence Transformers | Text embeddings |
| LangChain | Document chunking |
| Smolagents | Agentic retrieval |
| Hugging Face Transformers | LLM inference |
| Qwen2.5-3B-Instruct | Answer generation |
| NumPy | Numerical operations |
| Pandas | Data Handling |

## Knowledge Base

The knowledge base contains information about:

- Artificial Intelligence
- Large Language Models
- Retrieval-Augmented Generation
- Embeddings
- Document Chunking
- Vector Search
- Hybrid Search
- RAG Evaluation

## Retrieval Pipeline

### Document Chunking :-
Documents are divided into smaller chunks using LangChain's RecursiveCharacterTextSplitter.

Current configuration:

- Chunk size: 300
- Chunk overlap: 50

### Dense Retrieval :-

The system uses: sentence-transformers/all-MiniLM-L6-v2
The generated embeddings are stored in a FAISS index and used for semantic similarity search.

### Sparse Retrieval :-

BM25 is used for keyword-based retrieval.
This helps retrieve documents containing exact terms, technical expressions, and identifiers.

### Hybrid Retrieval :- 

Dense retrieval and BM25 scores are normalized and combined to produce a hybrid retrieval score.

This combines:
Semantic similarity
Keyword matching

## Agentic Retrieval 

The retrieval system is exposed through a single tool:
knowledge_base_search(query)

The agent is restricted to the provided knowledge base.

It is not allowed to use:

- Web search
- Wikipedia
- External sources
- URLs
- External databases
- Unapproved tools

If the knowledge base does not contain enough information, the system returns:
The knowledge base does not contain enough information to answer this question.

## Language Model

The system uses the open-weight model:
Qwen/Qwen2.5-3B-Instruct

The model is accessed through the Hugging Face Transformers pipeline.
The model generates answers using the retrieved knowledge-base context.

## Interactive Testing

The system provides an interactive question-answering interface.

Example:

<img width="1029" height="359" alt="image" src="https://github.com/user-attachments/assets/65fc80c0-6ea3-4060-b6ea-28ceeb24eb92" />
<img width="1032" height="325" alt="image" src="https://github.com/user-attachments/assets/0a977923-cd68-4577-a679-98608eb2edae" />

## Self-Evaluation

After the interactive testing session ends, the system performs LLM-based self-evaluation.

The evaluation measures:

Metric Scores :-
- Retrieval Relevance	- /5
- Answer Correctness - /5
- Groundedness - /5
- Completeness -	/5
- Overall - /5








