# Course Navigator using Retrieval-Augmented Generation (RAG)
## Overview
This project implements a Retrieval-Augmented Generation (RAG) based system that allows users to ask natural language questions about a video-based course and receive precise answers pointing to which video and which timestamp the relevant content is taught.

The system processes course videos end-to-end:
- Converts videos to audio
- Transcribes speech to text
- Chunks and embeds transcripts
- Retrieves relevant segments using semantic similarity
- Uses an LLM to generate guided, context-aware answers

This project demonstrates practical understanding of LLMs, embeddings, vector similarity search, and RAG pipelines.

## Problem Statement
Video courses are hard to navigate. Learners often want answers to questions like:
- “Where is decorators explained?”
- “Which video covers list comprehensions?”
- “How much of OOP is taught and where?”

Manually searching through long videos is inefficient.
This project solves that by enabling semantic search + LLM-based guidance over course content.

## System Architecture

```mermaid
flowchart TD
    A[Video Files] --> B[Audio Extraction\n(ffmpeg)];
    B --> C[Speech-to-Text\n(Whisper)];
    C --> D[Chunking & Metadata];
    D --> E[Embedding Generation\n(bge-m3)];
    E --> F[Vector Similarity Search];
    F --> G[LLM Answer Generation\n(LLaMA)];
```
## Tech Stack
- Python
- Whisper (large-v2) – Speech-to-text & translation
- Ollama API
    - bge-m3 for embeddings
    - llama3.2:1b for text generation
- scikit-learn – Cosine similarity
- pandas / NumPy – Data handling
- joblib – Embedding persistence
- ffmpeg – Audio extraction

## Key Concepts Demonstrated
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- Vector similarity search (cosine similarity)
- Prompt engineering with contextual grounding
- Speech-to-text pipelines
- Handling unstructured video data