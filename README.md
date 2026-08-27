# Hybrid RAG System

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

## Technologies
Python | Hugging Face Transformers | Sentence Transformers | FAISS | BM25 | LangChain Text Splitters | Smolagents | Qwen2.5-3B-Instruct | NumPy | Pandas
