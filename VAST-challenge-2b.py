import marimo

__generated_with = "0.20.2"
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

    return alt, json, nx, pd, plt


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
    G = nx.node_link_graph(data, link="edges")
    keep_subtypes = {"Person", "Organization", "Group"}

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
    return idx, members


@app.cell
def _(E, idx, members, nx, plt):
    H = E.subgraph(members).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(H, seed=42, k=None)

    # edge widths reflect shared-event count
    weights = [H[u][v].get("weight", 1) for u, v in H.edges()]
    # scale widths so it’s not too thick
    edge_widths = [0.5 + (w ** 0.6) for w in weights]

    # you can also size nodes by degree
    deg = dict(H.degree())
    node_sizes = [200 + 30 * deg.get(n, 0) for n in H.nodes()]

    nx.draw_networkx_edges(H, pos, ax=ax, width=edge_widths, alpha=0.35)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=9)

    ax.set_title(f"Community {idx} — {len(members)} nodes, {H.number_of_edges()} edges")
    ax.axis("off")
    fig
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


if __name__ == "__main__":
    app.run()
