# VAST Challenge 2025 — Mini-Challenge 3

Interactive visual analytics dashboard for investigating radio communications in the fictional community of Oceanus, built with [marimo](https://marimo.io/) and D3.js.

**Team:** AmanDeep Singh · Dominic van den Bungelaar · Kim Wilmink

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
# Clone the repository
git clone https://github.com/dominicvdb/visualanalytics-mini-challenge3.git
cd visualanalytics-mini-challenge3

# Install dependencies
pip install marimo pandas numpy networkx altair scipy plotly

# (One-time) Generate topic model cache
# This runs BERTopic + UMAP and saves results to CSV so the dashboard loads quickly.
pip install bertopic sentence-transformers umap-learn
python save_topic_cache.py
```

The cache script creates three files in `data/`:
- `topic_plot_df.csv` — UMAP document embeddings
- `topics_df.csv` — entity × topic matrix
- `topic_keywords.csv` — topic keywords for readable labels

## Run

```bash
marimo run combined_app.py
```

The dashboard opens in your browser. Use the top-level tabs to navigate between questions.

> **Note:** If the output exceeds marimo's default size limit, place the provided `pyproject.toml` next to `combined_app.py` — it sets the limit to 20 MB.

## Project structure

```
├── combined_app.py        # Main marimo notebook (all questions)
├── save_topic_cache.py    # One-time script for topic model caching
├── pyproject.toml         # Marimo runtime config (output size limit)
├── data/
│   ├── MC3_graph.json     # VAST Challenge knowledge graph
│   ├── MC3_schema.json    # Graph schema
│   ├── categories_v2.csv  # LLM-classified message categories
│   ├── topic_plot_df.csv  # (generated) UMAP embeddings
│   ├── topics_df.csv      # (generated) Topic matrix
│   └── topic_keywords.csv # (generated) Topic labels
└── public/                # Static images for Q4
```
