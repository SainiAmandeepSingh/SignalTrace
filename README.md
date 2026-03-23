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
pip install -r requirements.txt
```

All pre-computed data files (LLM classifications, topic model cache) are already included in `data/`, so no additional generation steps are needed.

## Run

```bash
marimo run combined_app_final.py
```

The dashboard opens in your browser. Use the top-level tabs to navigate between questions.

> **Note:** The provided `pyproject.toml` sets marimo's output size limit to 20 MB. Make sure it sits next to `combined_app.py`.

## LLM classification notebook

`intent_modeling.py` is a separate marimo notebook that was used to classify all 584 messages into 10 categories using the OpenAI API (`gpt-4o-mini`). The output (`data/categories_v2.csv`) is already included in the repository, so you do **not** need to re-run it or have an OpenAI API key.

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
├── combined_app.py        # Main marimo notebook (all questions)
├── intent_modeling.py     # LLM classification notebook (OpenAI API)
├── save_topic_cache.py    # One-time script for topic model caching
├── pyproject.toml         # Marimo runtime config (output size limit)
├── data/
│   ├── MC3_graph.json     # VAST Challenge knowledge graph
│   ├── MC3_schema.json    # Graph schema
│   ├── categories_v2.csv  # LLM-classified message categories (pre-generated)
│   ├── topic_plot_df.csv  # UMAP document embeddings (pre-generated)
│   ├── topics_df.csv      # Entity × topic matrix (pre-generated)
│   └── topic_keywords.csv # Topic keyword labels (pre-generated)
└── public/                # Static images for Q4
```
