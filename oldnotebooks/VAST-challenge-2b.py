import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import json
    import networkx as nx
    from collections import Counter
    from pyvis.network import Network
    import matplotlib.pyplot as plt
    import altair as alt

    return alt, json, nx, pd


@app.cell
def _(json):
    def load_json(file_path):
        """
        Load a JSON file and return its content as a Python dictionary.

        :param file_path: Path to the JSON file.
        :return: Dictionary containing the JSON data.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data


    return (load_json,)


@app.cell
def _(load_json):
    data = load_json('Desktop/ADS UU/Visual Analytics for Big Data/VAST-challenge/MC3-data/MC3_graph.json')
    schema = load_json('Desktop/ADS UU/Visual Analytics for Big Data/VAST-challenge/MC3-data/MC3_schema.json')
    nodes_type = schema['schema']['nodes'].keys()
    # Process the data to differentes kinds of events
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    return data, edges, nodes


@app.cell
def _(nodes, pd):
    entities = []
    for entity in nodes:
        if entity.get("type") == "Entity":
            entities.append(entity)

    entities_df = pd.DataFrame(entities)
    entities_df
    return (entities_df,)


@app.cell
def _(entities_df):
    persons = entities_df[entities_df['sub_type'] == 'Person']
    persons
    return


@app.function
def get_participants_communications(event_id: str, nodes, edges):
    sent_edge = next((e for e in edges if e.get('type') == 'sent' and e.get('target') == event_id), None)
    # Find the 'received' edge where source is the event
    received_edge = next((e for e in edges if e.get('type') == 'received' and e.get('source') == event_id), None)

    if not sent_edge or not received_edge:
        return None, None

    source_entity_id = sent_edge['source']
    target_entity_id = received_edge['target']

    source_node = next((n for n in nodes if n.get('id') == source_entity_id), None)
    target_node = next((n for n in nodes if n.get('id') == target_entity_id), None)

    return source_node, target_node


@app.cell
def _(edges, nodes, pd):
    messages = []
    for event in nodes:
        if event.get("type") == "Event":
            if event.get("sub_type") == "Communication":
                event_id = event.get("id")
                time_slot = event.get("timestamp")
                content = event.get("content", "")
                participants = get_participants_communications(event_id, nodes, edges)
                messages.append(
                    {
                        "event_id": event_id,
                        "datetime": time_slot,
                        "content": content,
                        "source": participants[0]['name'],
                        "target": participants[1]['name'] 
                }
            )
    messages_df = pd.DataFrame(messages)
    messages_df
    return (messages_df,)


@app.cell
def _(messages_df):
    unique_event_ids = messages_df['event_id'].unique()
    print(f"Number of unique event IDs: {len(unique_event_ids)}")
    # I want to verify if i have duplicates in the event IDs
    duplicates = messages_df[messages_df.duplicated(subset=['event_id'], keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate event IDs:")
        print(duplicates[['event_id', 'content']])
    else:
        print("No duplicate event IDs found.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Community Detection
    """)
    return


@app.cell
def _(data, nx):
    G = nx.node_link_graph(data, edges="edges")
    keep_subtypes = {"Person", "Organization", "Group", "Vessel"}

    entity_nodes = {
        n for n, a in G.nodes(data=True)
        if a.get("type") == "Entity" and a.get("sub_type") in keep_subtypes
    }
    event_nodes = {n for n, a in G.nodes(data=True) if a.get("type") == "Event"}
    return G, entity_nodes, event_nodes


@app.cell
def _(G, entity_nodes, event_nodes, nx):
    from networkx.algorithms import bipartite
    B = nx.Graph()
    B.add_nodes_from((n, G.nodes[n]) for n in (entity_nodes | event_nodes))

    for u, v, attr in G.to_undirected().edges(data=True):
        u_is_entity = u in entity_nodes
        v_is_entity = v in entity_nodes
        u_is_event  = u in event_nodes
        v_is_event  = v in event_nodes

        # keep only Entity<->Event edges
        if (u_is_entity and v_is_event) or (v_is_entity and u_is_event):
            B.add_edge(u, v, **attr)

    E = bipartite.weighted_projected_graph(B, entity_nodes)
    return (E,)


@app.cell
def _(E, nx):
    communities = list(nx.community.greedy_modularity_communities(E, weight="weight"))
    communities
    return (communities,)


@app.cell
def _(communities):
    len(communities)
    return


@app.cell
def _(communities):
    for x, c in enumerate(communities, 1):
        print(x, len(c), sorted(list(c)))
    return


@app.cell
def _(E):
    # strongest entity-entity links (most shared events)
    top_edges = sorted(E.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True)[:10]
    top_edges
    return


@app.cell
def _(communities, mo):
    community_list = [sorted(list(c)) for c in communities]  # stable ordering
    community_sizes = [len(c) for c in community_list]

    options = [
        (f"Community {i} ({community_sizes[i]} nodes)", i)
        for i in range(len(community_list))
    ]

    community_labels = [f"Community {i} ({community_sizes[i]} nodes)" for i in range(len(community_list))]

    community_dd = mo.ui.dropdown(
        options=community_labels,
        value=community_labels[0],
        label="Select a community",
    )

    community_dd
    return community_dd, community_labels, community_list


@app.cell
def _(G, community_dd, community_labels, community_list, mo, pd):
    idx = community_labels.index(community_dd.value)
    members = community_list[idx]

    rows = []
    for n in members:
        a = G.nodes[n]  # attributes from original graph
        rows.append({
            "node": n,
            "sub_type": a.get("sub_type"),
            "label": a.get("label"),
            "type": a.get("type"),
        })

    df_members = pd.DataFrame(rows).sort_values(["sub_type", "node"])
    mo.ui.table(df_members)
    return


@app.cell
def _(E, G, alt, community_dd, community_labels, community_list, nx, pd):
    def _():
        # pick community
        idx = community_labels.index(community_dd.value)
        members = community_list[idx]

        # subgraph of the projected entity graph
        H = E.subgraph(members).copy()

        # compute layout coordinates (force-directed)
        pos = nx.spring_layout(H, seed=42)

        # nodes df
        nodes_df = pd.DataFrame(
            [
                {
                    "node": n,
                    "x": float(pos[n][0]),
                    "y": float(pos[n][1]),
                    "sub_type": G.nodes[n].get("sub_type"),
                }
                for n in H.nodes()
            ]
        )

        # edges df (as line segments)
        edges_df = pd.DataFrame(
            [
                {
                    "source": u,
                    "target": v,
                    "x": float(pos[u][0]),
                    "y": float(pos[u][1]),
                    "x2": float(pos[v][0]),
                    "y2": float(pos[v][1]),
                    "weight": float(d.get("weight", 1)),
                }
                for u, v, d in H.edges(data=True)
            ]
        )

        # ---- Altair layers ----
        edges_layer = (
            alt.Chart(edges_df)
            .mark_rule(opacity=0.25)
            .encode(
                x="x:Q", y="y:Q",
                x2="x2:Q", y2="y2:Q",
                strokeWidth=alt.StrokeWidth("weight:Q", legend=None),
                tooltip=["source:N", "target:N", "weight:Q"],
            )
        )

        nodes_layer = (
            alt.Chart(nodes_df)
            .mark_circle(size=120)
            .encode(
                x="x:Q", y="y:Q",
                tooltip=["node:N", "sub_type:N"],
                shape="sub_type:N",
            )
        )

        labels_layer = (
            alt.Chart(nodes_df)
            .mark_text(align="left", dx=7, dy=-7, fontSize=11)
            .encode(
                x="x:Q", y="y:Q",
                text="node:N",
            )
        )

        chart = (
            (edges_layer + nodes_layer + labels_layer)
            .properties(
                width=800,
                height=600,
                title=f"{community_dd.value} — {H.number_of_edges()} edges",
            )
            .configure_axis(grid=False, domain=False, ticks=False, labels=False)
        )
        return chart


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Topic Modeling
    """)
    return


@app.cell
def _(np):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    def extract_topics_tfidf(texts, num_topics=15):
        """Extract topics using TF-IDF keywords"""
        if len(texts) < 2:
            return [["insufficient", "data"]], [[1.0] for _ in texts]

        try:
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                max_df=0.8,
            )
            tfidf_matrix = tfidf.fit_transform(texts)

            # Get overall document scores
            doc_scores = tfidf_matrix.toarray()

            # Get top keywords across all documents
            feature_names = tfidf.get_feature_names_out()
            total_scores = np.sum(doc_scores, axis=0)
            top_indices = total_scores.argsort()[-num_topics:][::-1]

            # Create topics - each topic is a single keyword
            topics = [[feature_names[i]] for i in top_indices]

            # Create document-topic matrix (each doc has scores for each keyword)
            doc_topics = doc_scores[:, top_indices].tolist()

            return topics, doc_topics

        except Exception as e:
            print(f"TF-IDF extraction failed: {str(e)}")
            return [["error", "processing"]], [[1.0] for _ in texts]

    def extract_topics_bertopic(texts, min_topic_size=5, top_n=10):
        """Extract topics using BERTopic"""
        if len(texts) < min_topic_size * 2:
            min_topic_size = max(2, len(texts) // 4)

        try:
            # Use KeyBERT for better keyword extraction
            representation_model = KeyBERTInspired()

            topic_model = BERTopic(
                min_topic_size=min_topic_size,
                nr_topics="auto",
                language="english",
                calculate_probabilities=True,
                representation_model=representation_model,
                verbose=False,
            )
            topics, probs = topic_model.fit_transform(texts)

            # Get topic keywords
            topic_keywords = {}
            unique_topics = set(topics)
            if -1 in unique_topics:
                unique_topics.remove(-1)  # Remove outlier topic

            for topic_id in unique_topics:
                try:
                    keywords = topic_model.get_topic(topic_id)
                    topic_keywords[topic_id] = [word for word, _ in keywords[:top_n]]
                except:
                    continue

            if not topic_keywords:
                # Fallback if no topics found
                return extract_topics_tfidf(texts) + (None,)

            # Prepare document topic weights
            doc_topics = []
            max_topic_id = max(topic_keywords.keys()) if topic_keywords else 0

            for i, doc_topic in enumerate(topics):
                weights = [0.0] * (max_topic_id + 1)
                if doc_topic in topic_keywords:
                    weights[doc_topic] = 1.0
                doc_topics.append(weights)

            # Convert to ordered list of topic keywords
            topics_list = []
            for i in range(max_topic_id + 1):
                if i in topic_keywords:
                    topics_list.append(topic_keywords[i])
                else:
                    topics_list.append([])

            return topics_list, doc_topics, topic_model
        except Exception as e:
            print(f"BERTopic failed: {str(e)}")
            # Fallback to TF-IDF
            topics, doc_topics = extract_topics_tfidf(texts)
            return topics, doc_topics, None

    return extract_topics_bertopic, extract_topics_tfidf


@app.cell
def _(extract_topics_bertopic, extract_topics_tfidf, messages_df):
    topics_listm, doc_topics, topic_model = extract_topics_bertopic(messages_df['content'].tolist())
    if topic_model: 
        print(f"Extracted {len(topics_listm)} topics using BERTopic.")
    else:
        print("BERTopic failed, using TF-IDF instead.")
        topics_listm, doc_topics = extract_topics_tfidf(messages_df['content'].tolist())
    return doc_topics, topic_model, topics_listm


@app.cell
def _(topics_listm):
    topics_listm
    return


@app.cell
def _(topic_model):
    topic_model.get_topic_freq()
    return


@app.cell
def _(topic_model):
    topic_model.visualize_topics()
    return


@app.cell
def _(messages_df, topic_model):
    topic_model.visualize_documents(messages_df['content'].tolist())
    return


@app.cell
def _(doc_topics, messages_df, pd, topics_listm):
    # Create a dataframe with the topics X messages
    topics_df = pd.DataFrame(doc_topics, columns=[f"Topic_{i}" for i in range(len(topics_listm))])
    topics_df['source'] = messages_df['source']
    # Use source as the index
    topics_df.set_index('source', inplace=True)
    topics_df = topics_df.groupby(topics_df.index).sum()
    topics_df
    return (topics_df,)


@app.cell
def _(messages_df, topic_model):
    import plotly.express as px
    import umap
    from sentence_transformers import SentenceTransformer

    # Original docs
    docs = messages_df["content"].tolist()

    # Embed docs
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Reduce to 2D
    reduced = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    # Topic assignments
    topics, probs = topic_model.transform(docs)

    # Topic names
    topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
    topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    plot_df = messages_df.copy()
    plot_df["x"] = reduced[:, 0]
    plot_df["y"] = reduced[:, 1]
    plot_df["topic"] = topics
    plot_df["topic_name"] = plot_df["topic"].map(topic_name_map)


    def truncate_text(text, n=100):
        text = str(text)
        return text if len(text) <= n else text[:n] + "..."


    plot_df["content_short"] = plot_df["content"].apply(truncate_text)
    return plot_df, px


@app.cell
def _(plot_df, px):
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic_name",
        hover_data={
            "source": True,
            "target": True,
            "content": True,
            "topic_name": True,
            "x": False,
            "y": False,
        },
        width=1200,
        height=800,
    )

    fig.update_traces(
        customdata=plot_df[
            ["source", "target", "content_short", "topic_name"]
        ].to_numpy(),
        hovertemplate=(
            "<b>Topic:</b> %{customdata[3]}<br>"
            "<b>From:</b> %{customdata[0]}<br>"
            "<b>To:</b> %{customdata[1]}<br><br>"
            "<b>Message preview:</b><br>%{customdata[2]}<br><br>"
        ),
    )

    fig.update_layout(
        title="Topic Clusters",
        legend_title="Clusters",
    )

    fig.show()
    return


@app.cell
def _(messages_df):
    messages_df
    return


@app.cell
def _(G, community_list):
    node_to_community = {}
    for i, comm in enumerate(community_list):
        for node_id in comm:
            name = G.nodes[node_id].get("name")
            if name:
                node_to_community[name] = i
    return (node_to_community,)


@app.cell
def _(node_to_community, topics_df):
    community_topics_df = topics_df.copy()
    community_topics_df["source"] = community_topics_df.index
    community_topics_df["community"] = community_topics_df["source"].map(node_to_community)
    return (community_topics_df,)


@app.cell
def _(community_topics_df):
    community_topic_matrix = (
        community_topics_df
        .drop(columns="source")
        .groupby("community")
        .sum()
    )

    community_topic_matrix
    return (community_topic_matrix,)


@app.cell
def _(community_topic_matrix):
    # converting to long format for visualization
    matrix_plot_df = (
        community_topic_matrix
        .reset_index()
        .melt(
            id_vars="community",
            var_name="topic",
            value_name="score"
        )
    )

    matrix_plot_df
    return (matrix_plot_df,)


@app.cell
def _(alt, matrix_plot_df):
    topic_heatmap = (
        alt.Chart(matrix_plot_df)
        .mark_rect()
        .encode(
            x=alt.X("topic:N", title="Topic"),
            y=alt.Y("community:N", title="Community"),
            color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["community", "topic", "score"]
        )
        .properties(
            width=500,
            height=300,
            title="Topic Distribution per Community"
        )
    )

    topic_heatmap
    return


@app.cell
def _(alt, community_topic_matrix):
    community_topic_matrix_norm = community_topic_matrix.div(
        community_topic_matrix.sum(axis=1), axis=0
    )

    norm_matrix_plot_df = (
        community_topic_matrix_norm
        .reset_index()
        .melt(id_vars="community", var_name="topic", value_name="score")
    )


    norm_topic_heatmap = (
        alt.Chart(norm_matrix_plot_df)
        .mark_rect()
        .encode(
            x=alt.X("topic:N", title="Topic"),
            y=alt.Y("community:N", title="Community"),
            color=alt.Color("score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["community", "topic", "score"]
        )
        .properties(
            width=500,
            height=300,
            title="Topic Distribution per Community"
        )
    )

    norm_topic_heatmap
    return


@app.cell
def _(node_to_community, pd, plot_df):

    plot_df["sender_community"] = plot_df["source"].map(node_to_community)
    plot_df["recipient_community"] = plot_df["target"].map(node_to_community)

    sender_view = plot_df[["topic_name", "sender_community"]].rename(
        columns={"sender_community": "community"}
    )
    recipient_view = plot_df[["topic_name", "recipient_community"]].rename(
        columns={"recipient_community": "community"}
    )

    community_topics_df2 = pd.concat([sender_view, recipient_view], ignore_index=True)

    community_topic_counts = (
        community_topics_df2
        .groupby(["community", "topic_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["community", "count"], ascending=[True, False])
    )
    return (community_topic_counts,)


@app.cell
def _(community_topic_counts):
    community_topic_counts
    return


@app.cell
def _(community_topic_counts):
    community_topic_summary = (
        community_topic_counts
        .groupby("community")
        .apply(lambda x: x[["topic_name", "count"]].head(5).to_dict("records"))
        .to_dict()
    )
    community_topic_summary
    return


if __name__ == "__main__":
    app.run()
