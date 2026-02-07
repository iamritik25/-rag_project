👻 Agentic RAG PDF Question Answering System

Local-First • Hallucination-Safe • Session-Scoped • Agent-Driven UI

1. Project Overview
=====================

This project is a fully local Retrieval-Augmented Generation (RAG) system that allows users to upload PDFs and ask questions strictly grounded in document content, with zero hallucinations, no cloud APIs, and automatic cleanup of embeddings on session end.

The system behaves like an AI agent, not just a chatbot:

It reacts to failures

Changes UI behavior

Prevents hallucinations

Manages its own memory lifecycle

This README documents everything that was built, in the exact order it was engineered.

2. Initial Goal
===================
Original Objective:
===================

“Build a local PDF chatbot using Mistral and Ollama.”

Problems Identified Early

Hallucinated answers

No document isolation

No session cleanup

Embeddings persisted forever

UI gave false confidence

Typical RAG demo ≠ production system

👉 I decided to build a production-thinking RAG system instead of a demo.

3. Step-by-Step Development Journey
======================================
    STEP 1️⃣ — Local LLM Setup (Ollama + Mistral)
    =============================================
    What I Did:
    =============

        i)Installed Ollama

        ii)Pulled Mistral

        iii)Verified fully local inference

        iv)Run: ollama run mistral

    Why?

        i)Zero API cost

        ii)No data leakage

        iii)Enterprise-safe

        iv)Works offline

    STEP 2️⃣ — Streamlit UI Foundation
    ==================================
    What I Did?

        i)Created app.py

        ii)Built Streamlit UI

        iii)Added file uploader

        iv)Added question input box

    Why?

        i)Rapid prototyping

        ii)Local UI

        iii)Easy debugging

        iv)Perfect for internal tools

    STEP 3️⃣ — PDF Text Extraction
    ==============================
    What I Did?

        i)Used pypdf

        ii)Extracted text page-by-page

        iii)Ignored empty pages

    Why?

        i)PDFs are messy

        ii)Clean input = better embeddings

        iii)Avoids junk vectors

    STEP 4️⃣ — Text Chunking Strategy
    =================================
    What I Did?

        i)Implemented manual chunking

        ii)Used overlap

            chunk_size = 500
            overlap = 100

    Why?

        i)Prevents context loss

        ii)Improves semantic retrieval

        iii)Standard in production RAG

    STEP 5️⃣ — Embedding Model Selection
    ====================================
    What I Did?

        i)Chose all-MiniLM-L6-v2

        ii)Loaded via SentenceTransformers

    Why?

        i)Fast

        ii)Lightweight

        iii)Strong semantic performance

        iv)Local-friendly

    STEP 6️⃣ — FAISS Vector Store (Manual Control)
    ==============================================
    What I Did?

        i)Used FAISS directly

        ii)No LangChain abstractions

        iii)Created indexes per PDF

            indexes/
                ├── pdf1.index
                ├── pdf1.txt

    Why?

        i)Full control

        ii)Debuggable

        iii)No hidden memory


    STEP 7️⃣ — Multi-PDF Support
    ============================
    What I Did?

        i)Allowed uploading multiple PDFs

        ii)Created separate FAISS index per PDF

        iii)Added dropdown to select which PDF to query

    Why?

        i)Prevents cross-document contamination

        ii)Mirrors real document systems

        iii)Enables focused retrieval

    STEP 8️⃣ — Retrieval Logic (Top-K Search)
    ==========================================
    What I Did?

        i)Embedded user query

        ii)Retrieved top-3 chunks

        iii)Assembled deterministic context

    Why?

        i)Predictable behavior

        ii)Easy to explain failures

        iii)Avoids hallucination amplification

    STEP 9️⃣ — Hallucination Prevention Layer
    ===========================================
    What I Did?

        i)Added hard rule:

        ii)If answer not in retrieved context → say
            “Not found in the document.”

    Why?

        i)Correct AI > confident AI

        ii)Prevents legal & factual risks

        iii)This is how real AI systems behave

    STEP 🔟 — Agentic UI (Ghost System)
    ====================================
    What I Did?

        i)Added ghost character to UI

        ii)Ghost reacts to system state

        iii)Condition	Ghost:
            Normal	😄
            Thinking	🤔
            Repeated failures	😡
            Session ended	☠️
    Why?

        i)Makes AI behavior interpretable

        ii)Shows uncertainty

        iii)Prevents blind trust

    STEP 1️⃣1️⃣ — Failure Tracking
    ===============================
    What I Did?

        i)Tracked repeated hallucination attempts

        ii)Increased ghost anger on failures

        iii)Triggered red neon pulse effect

    Why?

        i)AI should push back

        ii)Users must see limits

        iii)Encourages correct usage

    STEP 1️⃣2️⃣ — Typing Animation
    =============================
    What I Did?

        i)Streamed answer word-by-word

        ii)Synced ghost behavior with typing

    Why?

        i)Better UX

        ii)Feels alive

        iii)Professional polish

    STEP 1️⃣3️⃣ — Session Lifecycle Management
    ==========================================
    What I Did?

        i)Added 🛑 End Session button

        ii)On click:

            Delete PDFs

            Delete FAISS indexes

            Delete chunk files

            Clear session state

            Stop execution

    Why?

        i)Prevents data leakage

        ii)Required for compliance

        iii)Rarely done in demos (but critical)

    STEP 1️⃣4️⃣ — Auto-Delete Embeddings on Session End
    ===================================================
    What I Did?

        i)Ensured no vectors persist after session

        ii)No background memory

        iii)Clean shutdown

    Why?

        i)GDPR-safe design

        ii)Prevents accidental reuse

        iii)Production-grade behavior

    STEP 1️⃣5️⃣ — Local-Only Enforcement
===================================
    What I Did?

        i)Avoided network exposure

        ii)Focused on localhost

        iii)Ignored network URL display

    Why?

        i)Privacy

        ii)Security

        iii)Internal tool mindset

4. Final Architecture Diagram
==============================

User
 │
 ▼
Streamlit UI (Agent Controller)
 │
 ├── PDF Upload
 ├── PDF Selector
 ├── Ghost Agent
 ├── Session Control
 │
 ▼
PDF Parsing (PyPDF)
 │
 ▼
Chunking
 │
 ▼
Embeddings (MiniLM)
 │
 ▼
FAISS (Per-PDF Index)
 │
 ▼
Retriever (Top-K)
 │
 ▼
Mistral (Ollama)
 │
 ▼
Hallucination Guard
 │
 ▼
Typed Answer + Agent Reaction

5. Tech Stack
===============
    i)Layer:    Technology
    ii)UI:      Streamlit
    iii)LLM:    Mistral (Ollama)
    iv)Embeddings: SentenceTransformers
    v)Vector DB:  FAISS
    vi)PDF Parsing: PyPDF
    vii)Language:	Python

6. How to Run
================
pip install -r requirement.txt
ollama run mistral
streamlit run app.py


Open:

http://localhost:8501

7. What This Project Proves
=============================

    i)Deep RAG understanding

    ii)Agentic system thinking

    iii)AI safety awareness

    iv)Memory lifecycle control

    v)Production-ready design

    vi)No framework dependency illusion