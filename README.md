# SignalTrace

### Visual analytics dashboard for VAST Challenge 2025 — Mini-Challenge 3

**Team:** AmanDeep Saini Singh · Dominic van den Bungelaar · Kim Wilmink

Utrecht University · Visual Analytics for Big Data · 2025–2026

---

## Overview

SignalTrace is an interactive visual analytics dashboard for investigating radio communications in the fictional coastal community of Oceanus. Built with [Marimo](https://marimo.io/) reactive notebooks and D3.js, the dashboard combines LLM-based message classification, topic modelling, and entity-relationship visualisation to support analysts working through the VAST Challenge 2025 Mini-Challenge 3 questions.

This repository is forked from the [original team repository](https://github.com/dominicvdb/visualanalytics-mini-challenge3) and maintained under my account for portfolio purposes.

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
git clone https://github.com/SainiAmandeepSingh/SignalTrace.git
cd SignalTrace
pip install -r requirements.txt
```

All pre-computed data files (LLM classifications, topic model cache) are already included in `data/`, so no additional generation steps are needed.

## Run

```bash
marimo run combined_app_final.py
```

The dashboard opens in your browser. Use the top-level tabs to navigate between questions.

> **Note:** The provided `pyproject.toml` sets Marimo's output size limit to 20 MB. Make sure it sits next to `combined_app_final.py`.

## LLM classification notebook

`intent_modeling.py` is a separate Marimo notebook that classifies all 584 messages into 10 categories using the OpenAI API (`gpt-4o-mini`). The output (`data/categories_v2.csv`) is already included in the repository, so you do **not** need to re-run it or have an OpenAI API key.

To re-run it yourself:

```bash
pip install openai
# Edit intent_modeling.py and replace YOUR_API_KEY_HERE with your OpenAI key
marimo run intent_modeling.py
```

## Regenerating topic model cache (optional)

The topic model outputs are also pre-provided in `data/`. If you want to regenerate them:

```bash
pip install bertopic sentence-transformers umap-learn
python save_topic_cache.py
```

This creates `topic_plot_df.csv`, `topics_df.csv`, and `topic_keywords.csv` in `data/`.

## Project structure

```
├── combined_app_final.py   # Main Marimo notebook (all questions)
├── intent_modeling.py      # LLM classification notebook (OpenAI API)
├── save_topic_cache.py     # One-time script for topic model caching
├── pyproject.toml          # Marimo runtime config (output size limit)
├── data/
│   ├── MC3_graph.json      # VAST Challenge knowledge graph
│   ├── MC3_schema.json     # Graph schema
│   ├── categories_v2.csv   # LLM-classified message categories (pre-generated)
│   ├── topic_plot_df.csv   # UMAP document embeddings (pre-generated)
│   ├── topics_df.csv       # Entity × topic matrix (pre-generated)
│   └── topic_keywords.csv  # Topic keyword labels (pre-generated)
└── public/                 # Static images
```
