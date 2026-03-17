# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.19.11",
#     "matplotlib>=3.8.0",
#     "networkx>=3.2",
#     "pandas>=2.0.0",
#     "plotly>=5.18.0",
#     "numpy>=1.24.0",
#     "scipy>=1.11.0",
#     "scikit-learn>=1.3.0",
#     "seaborn>=0.13.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full", css_file="/usr/local/_marimo/custom.css")


# =============================================
# QUESTION 3: PSEUDONYM ANALYSIS - ADVANCED VISUAL ANALYTICS
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3: Pseudonym Analysis and Entity Resolution
    ## Advanced Visual Analytics for Knowledge Graph Investigation

    ### Research Question
    *It was noted by Clepper's intern that some people and vessels are using pseudonyms to communicate.*

    ---

    ## Visual Analytics Techniques Implemented

    This notebook implements **state-of-the-art visual analytics** techniques based on research from:
    - D3.js network visualization patterns (Bostock et al.)
    - IEEE VAST Challenge award-winning approaches
    - Plotly's hierarchical and multivariate visualization capabilities

    | Technique | Description | Purpose |
    |-----------|-------------|---------|
    | **Arc Diagram** | D3-style layout with arcs connecting nodes on a line | Reveals connection patterns and directionality |
    | **Adjacency Matrix** | Reorderable matrix with hierarchical clustering | Shows all entity connections at once |
    | **Parallel Coordinates** | Multi-axis plot for entity feature comparison | Compare multiple attributes simultaneously |
    | **Treemap** | Nested rectangles for hierarchical data | Communication volume by entity type |
    | **Icicle Chart** | Cascading rectangles for hierarchy | Alternative hierarchical view |
    | **Force-Directed Network** | Physics-based node positioning | Similarity-based entity clustering |
    | **Sankey Diagram** | Flow visualization for entity resolution | Pseudonym → Identity mapping |
    | **Radar/Spider Chart** | Radial multi-attribute comparison | Temporal activity fingerprints |

    These visualizations go beyond standard bar charts to provide **meaningful, publication-quality analytics** suitable for investigative work.
    """)
    return


# =============================================
# IMPORTS
# =============================================


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import numpy as np
    from collections import defaultdict, Counter
    from datetime import datetime
    from itertools import combinations
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import networkx as nx
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist, squareform
    return (
        Counter, combinations, datetime, defaultdict, fcluster, ff,
        go, json, leaves_list, linkage, make_subplots, mo, np, nx, pd, pdist, px, squareform
    )


# =============================================
# DATA LOADING
# =============================================


@app.cell
def _(json):
    with open('data/MC3_graph.json', 'r') as _f:
        graph_data = json.load(_f)

    nodes_by_id = {_n['id']: _n for _n in graph_data['nodes']}
    persons = {_n['id']: _n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Person'}
    vessels = {_n['id']: _n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Vessel'}
    organizations = {_n['id']: _n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Organization'}
    groups = {_n['id']: _n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Group'}
    locations = {_n['id']: _n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Location'}

    all_entities = {**persons, **vessels, **organizations, **groups}
    entity_ids = set(all_entities.keys())

    print(f"Loaded: {len(persons)} Persons, {len(vessels)} Vessels, {len(organizations)} Organizations, {len(groups)} Groups")
    return (
        all_entities, entity_ids, graph_data, groups, locations, 
        nodes_by_id, organizations, persons, vessels
    )


# =============================================
# BUILD COMMUNICATION DATA
# =============================================


@app.cell
def _(defaultdict, entity_ids, graph_data):
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)
    for _edge in graph_data['edges']:
        edges_to[_edge['target']].append(_edge)
        edges_from[_edge['source']].append(_edge)

    comm_events = [_n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Communication']
    comm_matrix = defaultdict(lambda: defaultdict(int))
    comm_records = []

    for _comm in comm_events:
        _comm_id = _comm['id']
        _timestamp = _comm.get('timestamp', '')
        _senders = [_e['source'] for _e in edges_to[_comm_id] if _e.get('type') == 'sent']
        _receivers = [_e['target'] for _e in edges_from[_comm_id] if _e.get('type') == 'received']

        for _sender in _senders:
            for _receiver in _receivers:
                if _sender in entity_ids and _receiver in entity_ids:
                    comm_matrix[_sender][_receiver] += 1
                    comm_records.append({
                        'sender': _sender, 'receiver': _receiver,
                        'timestamp': _timestamp, 'comm_id': _comm_id
                    })

    print(f"Extracted {len(comm_records)} communication records")
    return comm_events, comm_matrix, comm_records, edges_from, edges_to


# =============================================
# SECTION 1: PSEUDONYM DETECTION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pseudonym Detection via Naming Pattern Heuristics

    We identify potential pseudonyms using pattern matching on entity names:
    - **"The X"** pattern → "The Lookout", "The Middleman", "The Accountant"
    - **Title-only** → "Boss", "Small Fry"  
    - **"Mrs./Mr. X"** → "Mrs. Money"
    - **Single-word Person names** → "Sam", "Kelly", "Davis"
    """)
    return


@app.cell
def _(all_entities, pd):
    def detect_pseudonym(eid, edata):
        _label = edata.get('label', eid)
        _sub_type = edata.get('sub_type', '')
        _patterns = {
            'the_pattern': _label.lower().startswith('the '),
            'mrs_mr_pattern': _label.lower().startswith(('mrs.', 'mr.')),
            'single_word': len(_label.split()) == 1 and _sub_type == 'Person',
            'title_like': _label in ['Boss', 'Small Fry', 'The Intern'],
        }
        _score = sum(_patterns.values())
        return {
            'entity_id': eid, 'label': _label, 'sub_type': _sub_type,
            'pseudonym_score': _score, 'is_likely_pseudonym': _score >= 1, **_patterns
        }

    pseudonym_df = pd.DataFrame([detect_pseudonym(_eid, _ed) for _eid, _ed in all_entities.items()])
    pseudonym_df = pseudonym_df.sort_values('pseudonym_score', ascending=False)
    likely_pseudonyms = pseudonym_df[pseudonym_df['is_likely_pseudonym']].copy()

    print(f"Identified {len(likely_pseudonyms)} likely pseudonyms")
    return detect_pseudonym, likely_pseudonyms, pseudonym_df


# =============================================
# SECTION 2: COMPUTE SIMILARITY METRICS
# =============================================


@app.cell
def _(all_entities, combinations, comm_matrix, datetime, entity_ids, np, pd):
    # Compute communication partners for each entity
    def get_partners(eid, comm_mat):
        _sent_to = set(comm_mat.get(eid, {}).keys())
        _recv_from = set(_s for _s, _targets in comm_mat.items() if eid in _targets)
        return _sent_to | _recv_from

    entity_partners = {_eid: get_partners(_eid, comm_matrix) for _eid in entity_ids}

    # Compute Jaccard similarity
    def jaccard(set_a, set_b):
        if not set_a and not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) > 0 else 0.0

    # Build similarity records
    entity_list = sorted([_e for _e in entity_ids if len(entity_partners.get(_e, set())) > 0])
    n_entities = len(entity_list)
    entity_to_idx = {_e: _i for _i, _e in enumerate(entity_list)}

    similarity_matrix = np.zeros((n_entities, n_entities))
    _sim_records = []

    for _e1, _e2 in combinations(entity_list, 2):
        _jac = jaccard(entity_partners[_e1], entity_partners[_e2])
        _i, _j = entity_to_idx[_e1], entity_to_idx[_e2]
        similarity_matrix[_i, _j] = _jac
        similarity_matrix[_j, _i] = _jac
        if _jac > 0:
            _sim_records.append({
                'entity_a': _e1, 'entity_b': _e2, 'jaccard': _jac,
                'label_a': all_entities[_e1].get('label', _e1),
                'label_b': all_entities[_e2].get('label', _e2)
            })

    similarity_df = pd.DataFrame(_sim_records).sort_values('jaccard', ascending=False)

    # Compute temporal profiles
    def parse_ts(ts_str):
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else None
        except:
            return None

    print(f"Computed similarity for {len(entity_list)} active entities, {len(similarity_df)} pairs with similarity > 0")
    return (
        entity_list, entity_partners, entity_to_idx, get_partners, jaccard,
        n_entities, parse_ts, similarity_df, similarity_matrix
    )


# =============================================
# VISUALIZATION 1: ARC DIAGRAM
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Arc Diagram (D3-Style Network Visualization)

    The **Arc Diagram** is a classic D3.js visualization that arranges nodes along a horizontal line and draws connections as arcs. This layout:
    - Reveals **connection density** patterns
    - Shows **hub nodes** with many arcs
    - Enables **visual clustering** when nodes are ordered by similarity

    *Reference: Elijah Meeks, "D3.js in Action" - Network Visualization chapter*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, entity_list, entity_to_idx, go, likely_pseudonyms, mo, np):
    # Build arc diagram
    _n = len(entity_list)
    _x_positions = np.linspace(0, 10, _n)
    _labels = [all_entities.get(_e, {}).get('label', _e)[:10] for _e in entity_list]
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())

    fig_arc = go.Figure()

    # Draw arcs for connections
    _arc_count = 0
    for _sender, _targets in comm_matrix.items():
        if _sender not in entity_to_idx:
            continue
        _i = entity_to_idx[_sender]
        for _receiver, _count in _targets.items():
            if _receiver not in entity_to_idx:
                continue
            _j = entity_to_idx[_receiver]
            if _i < _j:  # Only draw once per pair
                _x1, _x2 = _x_positions[_i], _x_positions[_j]
                _mid_x = (_x1 + _x2) / 2
                _height = abs(_x2 - _x1) * 0.3  # Arc height proportional to distance
                
                # Create arc path using bezier curve approximation
                _arc_x = [_x1]
                _arc_y = [0]
                _steps = 20
                for _t in range(1, _steps + 1):
                    _tt = _t / _steps
                    _ax = _x1 + (_x2 - _x1) * _tt
                    _ay = _height * 4 * _tt * (1 - _tt)  # Parabola for arc
                    _arc_x.append(_ax)
                    _arc_y.append(_ay)
                
                _color = f'rgba(255, 107, 107, {min(0.1 + _count * 0.05, 0.6)})'
                fig_arc.add_trace(go.Scatter(
                    x=_arc_x, y=_arc_y, mode='lines',
                    line=dict(width=max(1, _count * 0.3), color=_color),
                    hoverinfo='text', hovertext=f"{_labels[_i]} → {_labels[_j]}: {_count} msgs",
                    showlegend=False
                ))
                _arc_count += 1

    # Draw nodes
    _node_colors = ['#FFD700' if entity_list[_i] in _pseudonym_ids else '#4ECDC4' for _i in range(_n)]
    _node_sizes = [10 + len(comm_matrix.get(entity_list[_i], {})) * 2 for _i in range(_n)]

    fig_arc.add_trace(go.Scatter(
        x=_x_positions, y=[0] * _n,
        mode='markers+text',
        marker=dict(size=_node_sizes, color=_node_colors, line=dict(width=1, color='white')),
        text=_labels, textposition='bottom center', textfont=dict(size=8),
        hovertext=[f"<b>{_labels[_i]}</b><br>Connections: {len(comm_matrix.get(entity_list[_i], {}))}" for _i in range(_n)],
        hoverinfo='text', showlegend=False
    ))

    fig_arc.update_layout(
        title=dict(text='<b>Arc Diagram: Communication Network</b><br><sup>D3-style visualization | Gold = Likely Pseudonym | Arc height = connection distance</sup>', x=0.5),
        height=500, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2]),
        plot_bgcolor='white'
    )

    mo.vstack([mo.md("### Arc Diagram - D3.js Style Network Visualization"), fig_arc])
    return (fig_arc,)


# =============================================
# VISUALIZATION 2: ADJACENCY MATRIX WITH CLUSTERING
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Adjacency Matrix with Hierarchical Clustering

    The **Adjacency Matrix** represents all pairwise connections in a grid format:
    - Each **row and column** = an entity
    - **Cell color intensity** = communication frequency
    - **Hierarchical ordering** groups similar entities together

    This technique is essential for discovering hidden clusters and is widely used in bioinformatics and social network analysis.

    *Reference: D3.js "Les Misérables Co-occurrence" by Mike Bostock*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, entity_list, entity_to_idx, go, leaves_list, linkage, mo, np, pdist, squareform):
    # Build communication adjacency matrix
    _n = len(entity_list)
    _adj_matrix = np.zeros((_n, _n))

    for _sender, _targets in comm_matrix.items():
        if _sender not in entity_to_idx:
            continue
        _i = entity_to_idx[_sender]
        for _receiver, _count in _targets.items():
            if _receiver not in entity_to_idx:
                continue
            _j = entity_to_idx[_receiver]
            _adj_matrix[_i, _j] += _count
            _adj_matrix[_j, _i] += _count  # Symmetric

    # Hierarchical clustering to reorder matrix
    _row_sums = _adj_matrix.sum(axis=1)
    _active_mask = _row_sums > 0
    _active_indices = np.where(_active_mask)[0]

    if len(_active_indices) > 3:
        _sub_matrix = _adj_matrix[np.ix_(_active_indices, _active_indices)]
        _sub_labels = [all_entities.get(entity_list[_i], {}).get('label', '')[:12] for _i in _active_indices]
        
        # Cluster based on communication patterns
        _dist = pdist(_sub_matrix + 0.001)  # Add small value to avoid zero distances
        _linkage_mat = linkage(_dist, method='average')
        _order = leaves_list(_linkage_mat)
        
        # Reorder matrix
        _ordered_matrix = _sub_matrix[_order, :][:, _order]
        _ordered_labels = [_sub_labels[_i] for _i in _order]
        
        fig_adj_matrix = go.Figure(data=go.Heatmap(
            z=np.log1p(_ordered_matrix),  # Log scale for better visibility
            x=_ordered_labels,
            y=_ordered_labels,
            colorscale='Viridis',
            hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Communications: %{customdata}<extra></extra>',
            customdata=_ordered_matrix.astype(int)
        ))
        
        fig_adj_matrix.update_layout(
            title=dict(text='<b>Adjacency Matrix with Hierarchical Clustering</b><br><sup>Rows/columns ordered by communication similarity | Color = log(messages)</sup>', x=0.5),
            height=700, width=800,
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8), autorange='reversed')
        )
    else:
        fig_adj_matrix = go.Figure()
        fig_adj_matrix.add_annotation(text="Insufficient data for adjacency matrix", showarrow=False)

    mo.vstack([mo.md("### Adjacency Matrix - Clustered Communication Patterns"), fig_adj_matrix])
    return (fig_adj_matrix,)


# =============================================
# VISUALIZATION 3: PARALLEL COORDINATES
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Parallel Coordinates Plot for Entity Feature Comparison

    The **Parallel Coordinates Plot** visualizes multi-dimensional data by drawing each entity as a polyline across parallel vertical axes. This allows:
    - **Simultaneous comparison** of multiple attributes
    - **Pattern identification** across dimensions
    - **Interactive filtering** by brushing axes

    Each line represents an entity, with attributes including communication volume, partner count, and pseudonym score.

    *Reference: Plotly documentation on Parallel Coordinates for multidimensional exploratory analysis*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, comm_records, datetime, entity_partners, go, mo, pd, pseudonym_df):
    # Build entity feature dataframe for parallel coordinates
    _entity_features = []

    for _, _row in pseudonym_df.iterrows():
        _eid = _row['entity_id']
        _sent = sum(comm_matrix.get(_eid, {}).values())
        _received = sum(1 for _r in comm_records if _r['receiver'] == _eid)
        _partners = len(entity_partners.get(_eid, set()))
        
        # Compute activity hours
        _hours = set()
        for _r in comm_records:
            if _r['sender'] == _eid or _r['receiver'] == _eid:
                try:
                    _ts = datetime.fromisoformat(_r['timestamp'].replace('Z', '+00:00'))
                    _hours.add(_ts.hour)
                except:
                    pass
        
        _entity_features.append({
            'label': _row['label'],
            'entity_type': _row['sub_type'],
            'pseudonym_score': _row['pseudonym_score'],
            'messages_sent': _sent,
            'messages_received': _received,
            'unique_partners': _partners,
            'active_hours': len(_hours),
            'is_pseudonym': 1 if _row['is_likely_pseudonym'] else 0
        })

    _features_df = pd.DataFrame(_entity_features)
    _features_df = _features_df[_features_df['messages_sent'] + _features_df['messages_received'] > 0]

    # Map entity type to numeric
    _type_map = {'Person': 0, 'Vessel': 1, 'Organization': 2, 'Group': 3}
    _features_df['type_code'] = _features_df['entity_type'].map(_type_map).fillna(4)

    fig_parallel = go.Figure(data=go.Parcoords(
        line=dict(
            color=_features_df['is_pseudonym'],
            colorscale=[[0, '#4ECDC4'], [1, '#FFD700']],
            showscale=True,
            colorbar=dict(title='Pseudonym', tickvals=[0, 1], ticktext=['No', 'Yes'])
        ),
        dimensions=[
            dict(range=[0, 3], label='Entity Type', values=_features_df['type_code'],
                 tickvals=[0, 1, 2, 3], ticktext=['Person', 'Vessel', 'Org', 'Group']),
            dict(range=[0, _features_df['pseudonym_score'].max() + 1], label='Pseudonym Score', 
                 values=_features_df['pseudonym_score']),
            dict(range=[0, _features_df['messages_sent'].max() + 1], label='Msgs Sent', 
                 values=_features_df['messages_sent']),
            dict(range=[0, _features_df['messages_received'].max() + 1], label='Msgs Received', 
                 values=_features_df['messages_received']),
            dict(range=[0, _features_df['unique_partners'].max() + 1], label='Partners', 
                 values=_features_df['unique_partners']),
            dict(range=[0, 24], label='Active Hours', values=_features_df['active_hours'])
        ]
    ))

    fig_parallel.update_layout(
        title=dict(text='<b>Parallel Coordinates: Entity Feature Comparison</b><br><sup>Each line = one entity | Gold = Likely pseudonym | Drag axes to filter</sup>', x=0.5),
        height=500, margin=dict(l=100, r=100)
    )

    mo.vstack([mo.md("### Parallel Coordinates - Multi-Dimensional Entity Analysis"), fig_parallel])
    return (fig_parallel,)


# =============================================
# VISUALIZATION 4: TREEMAP - HIERARCHICAL COMMUNICATION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Treemap: Hierarchical Communication Volume

    The **Treemap** uses nested rectangles to show hierarchical data:
    - **Size** = communication volume
    - **Color** = entity category
    - **Nesting** = type → entity hierarchy

    Click on rectangles to drill down into specific categories.

    *Reference: Plotly hierarchical charts documentation, D3 treemap layout*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, comm_records, go, mo, pd):
    # Build hierarchical data for treemap
    _treemap_data = []

    for _eid, _edata in all_entities.items():
        _sent = sum(comm_matrix.get(_eid, {}).values())
        _received = sum(1 for _r in comm_records if _r['receiver'] == _eid)
        _total = _sent + _received
        
        if _total > 0:
            _treemap_data.append({
                'entity': _edata.get('label', _eid),
                'type': _edata.get('sub_type', 'Unknown'),
                'total_messages': _total,
                'sent': _sent,
                'received': _received
            })

    _tm_df = pd.DataFrame(_treemap_data)

    fig_treemap = go.Figure(go.Treemap(
        labels=_tm_df['entity'],
        parents=_tm_df['type'],
        values=_tm_df['total_messages'],
        branchvalues='total',
        marker=dict(
            colors=_tm_df['total_messages'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Messages')
        ),
        textinfo='label+value',
        hovertemplate='<b>%{label}</b><br>Type: %{parent}<br>Total: %{value}<extra></extra>'
    ))

    fig_treemap.update_layout(
        title=dict(text='<b>Treemap: Communication Volume by Entity Type</b><br><sup>Rectangle size = total messages | Click to zoom</sup>', x=0.5),
        height=600, margin=dict(t=80, l=20, r=20, b=20)
    )

    mo.vstack([mo.md("### Treemap - Hierarchical Communication Volume"), fig_treemap])
    return (fig_treemap,)


# =============================================
# VISUALIZATION 5: ICICLE CHART
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Icicle Chart: Alternative Hierarchical View

    The **Icicle Chart** shows the same hierarchical data as the treemap but in a cascading rectangular layout:
    - Clearer **parent-child relationships**
    - Better for **comparing siblings**
    - Easier to **read labels**

    *Reference: Plotly icicle charts - new in Plotly 5.0*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, comm_records, go, mo, pd):
    # Build icicle chart data
    _icicle_data = []

    # Add root
    _icicle_data.append({'id': 'All Entities', 'parent': '', 'value': 0})

    # Add type level
    _type_totals = defaultdict(int)
    for _eid, _edata in all_entities.items():
        _sent = sum(comm_matrix.get(_eid, {}).values())
        _received = sum(1 for _r in comm_records if _r['receiver'] == _eid)
        _total = _sent + _received
        if _total > 0:
            _type_totals[_edata.get('sub_type', 'Unknown')] += _total

    for _type, _total in _type_totals.items():
        _icicle_data.append({'id': _type, 'parent': 'All Entities', 'value': 0})

    # Add entity level
    for _eid, _edata in all_entities.items():
        _sent = sum(comm_matrix.get(_eid, {}).values())
        _received = sum(1 for _r in comm_records if _r['receiver'] == _eid)
        _total = _sent + _received
        
        if _total > 0:
            _icicle_data.append({
                'id': _edata.get('label', _eid),
                'parent': _edata.get('sub_type', 'Unknown'),
                'value': _total
            })

    _ic_df = pd.DataFrame(_icicle_data)

    from collections import defaultdict as _dd

    fig_icicle = go.Figure(go.Icicle(
        ids=_ic_df['id'],
        labels=_ic_df['id'],
        parents=_ic_df['parent'],
        values=_ic_df['value'],
        branchvalues='total',
        marker=dict(colorscale='Blues'),
        tiling=dict(orientation='v'),
        hovertemplate='<b>%{label}</b><br>Messages: %{value}<extra></extra>'
    ))

    fig_icicle.update_layout(
        title=dict(text='<b>Icicle Chart: Cascading Entity Hierarchy</b><br><sup>Click to zoom | Width = communication volume</sup>', x=0.5),
        height=500, margin=dict(t=80, l=20, r=20, b=20)
    )

    mo.vstack([mo.md("### Icicle Chart - Cascading Hierarchical View"), fig_icicle])
    return (fig_icicle,)


# =============================================
# VISUALIZATION 6: FORCE-DIRECTED NETWORK WITH SLIDER
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Interactive Force-Directed Similarity Network

    The **Force-Directed Graph** uses physics simulation to position nodes:
    - **Connected nodes** = Jaccard similarity above threshold
    - **Edge weight** = similarity strength
    - **Node color**: Gold = Pseudonym, Teal = Person, Red = Vessel

    Use the **similarity threshold slider** to explore different connectivity levels.

    *Reference: D3.js force layout, NetworkX spring_layout*
    """)
    return


@app.cell
def _(mo):
    sim_threshold = mo.ui.slider(start=0.1, stop=0.8, step=0.05, value=0.25, label="Jaccard Similarity Threshold")
    sim_threshold
    return (sim_threshold,)


@app.cell
def _(all_entities, go, likely_pseudonyms, mo, np, nx, similarity_df, sim_threshold):
    _thresh = sim_threshold.value
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    
    _G = nx.Graph()
    _filtered = similarity_df[similarity_df['jaccard'] >= _thresh]
    
    for _, _row in _filtered.iterrows():
        _ea, _eb = _row['entity_a'], _row['entity_b']
        _G.add_node(_ea, label=all_entities[_ea].get('label', _ea), 
                   sub_type=all_entities[_ea].get('sub_type', 'Unknown'))
        _G.add_node(_eb, label=all_entities[_eb].get('label', _eb),
                   sub_type=all_entities[_eb].get('sub_type', 'Unknown'))
        _G.add_edge(_ea, _eb, weight=_row['jaccard'])
    
    if len(_G.nodes) > 1:
        _pos = nx.spring_layout(_G, k=2/np.sqrt(len(_G.nodes)+1), iterations=50, seed=42)
        
        _edge_x, _edge_y = [], []
        for _e in _G.edges():
            _x0, _y0 = _pos[_e[0]]
            _x1, _y1 = _pos[_e[1]]
            _edge_x.extend([_x0, _x1, None])
            _edge_y.extend([_y0, _y1, None])
        
        _node_x, _node_y, _colors, _sizes, _texts, _hovers = [], [], [], [], [], []
        _type_colors = {'Person': '#4ECDC4', 'Vessel': '#FF6B6B', 'Organization': '#95E1D3', 'Group': '#F38181'}
        
        for _n in _G.nodes(data=True):
            _x, _y = _pos[_n[0]]
            _node_x.append(_x)
            _node_y.append(_y)
            _label = _n[1].get('label', _n[0])
            _texts.append(_label)
            _hovers.append(f"<b>{_label}</b><br>Type: {_n[1].get('sub_type', 'Unknown')}<br>Degree: {_G.degree(_n[0])}")
            
            if _n[0] in _pseudonym_ids:
                _colors.append('#FFD700')
                _sizes.append(20)
            else:
                _colors.append(_type_colors.get(_n[1].get('sub_type', ''), '#999'))
                _sizes.append(12)
        
        fig_force = go.Figure()
        fig_force.add_trace(go.Scatter(x=_edge_x, y=_edge_y, mode='lines',
            line=dict(width=1, color='rgba(150,150,150,0.4)'), hoverinfo='none'))
        fig_force.add_trace(go.Scatter(x=_node_x, y=_node_y, mode='markers+text',
            marker=dict(size=_sizes, color=_colors, line=dict(width=1, color='white')),
            text=_texts, textposition='top center', textfont=dict(size=9),
            hovertext=_hovers, hoverinfo='text'))
        
        fig_force.update_layout(
            title=dict(text=f'<b>Force-Directed Similarity Network (threshold={_thresh})</b><br><sup>Nodes: {len(_G.nodes)} | Edges: {len(_G.edges)} | Gold = Pseudonym</sup>', x=0.5),
            height=600, showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#fafafa'
        )
    else:
        fig_force = go.Figure()
        fig_force.add_annotation(text=f"No connections at threshold {_thresh}", showarrow=False)
        fig_force.update_layout(height=300)

    mo.vstack([mo.md(f"### Force-Directed Network (Threshold = {_thresh})"), fig_force])
    return (fig_force,)


# =============================================
# VISUALIZATION 7: SANKEY DIAGRAM FOR RESOLUTION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Sankey Diagram: Pseudonym Resolution Flow

    The **Sankey Diagram** visualizes entity resolution as a flow:
    - **Left side**: Identified pseudonyms
    - **Right side**: Potential real identities
    - **Flow width**: Similarity score

    *Reference: Sankey diagrams for path analysis, D3.js Sankey layout*
    """)
    return


@app.cell
def _(all_entities, go, likely_pseudonyms, mo, similarity_df):
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    _flows = []

    for _, _row in similarity_df.head(30).iterrows():
        _is_pa = _row['entity_a'] in _pseudonym_ids
        _is_pb = _row['entity_b'] in _pseudonym_ids
        
        if _is_pa or _is_pb:
            _source = _row['label_a'] if _is_pa else _row['label_b']
            _target = (_row['label_b'] if _is_pa else _row['label_a']) + ' '
            _flows.append({'source': _source, 'target': _target, 'value': _row['jaccard']})

    if _flows:
        _nodes = list(set([_f['source'] for _f in _flows] + [_f['target'] for _f in _flows]))
        _node_idx = {_n: _i for _i, _n in enumerate(_nodes)}
        _node_colors = ['#FFD700' if not _n.endswith(' ') else '#4ECDC4' for _n in _nodes]

        fig_sankey = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color='white', width=1),
                     label=_nodes, color=_node_colors),
            link=dict(source=[_node_idx[_f['source']] for _f in _flows],
                     target=[_node_idx[_f['target']] for _f in _flows],
                     value=[_f['value'] * 100 for _f in _flows],
                     color='rgba(255, 215, 0, 0.3)')
        ))

        fig_sankey.update_layout(
            title=dict(text='<b>Sankey: Pseudonym Resolution Flow</b><br><sup>Gold = Pseudonyms | Teal = Candidate identities | Width = similarity</sup>', x=0.5),
            height=500, font=dict(size=10)
        )
    else:
        fig_sankey = go.Figure()
        fig_sankey.add_annotation(text="No resolution candidates", showarrow=False)

    mo.vstack([mo.md("### Sankey Diagram - Entity Resolution Flow"), fig_sankey])
    return (fig_sankey,)


# =============================================
# VISUALIZATION 8: RADAR CHART FOR TEMPORAL COMPARISON
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Radar Chart: Temporal Activity Fingerprint Comparison

    The **Radar Chart** (Spider Chart) compares hourly activity patterns:
    - Each **axis** = one hour of the day (0-23)
    - **Distance from center** = normalized activity level
    - **Overlap** indicates similar operational patterns

    Select entities to compare their "temporal fingerprints".
    """)
    return


@app.cell
def _(all_entities, mo):
    _opts = {all_entities[_e].get('label', _e): _e for _e in all_entities}
    _sorted = dict(sorted(_opts.items()))

    radar_a = mo.ui.dropdown(options=_sorted, value=list(_sorted.values())[0] if _sorted else None, label="Entity A")
    radar_b = mo.ui.dropdown(options=_sorted, value=list(_sorted.values())[1] if len(_sorted) > 1 else None, label="Entity B")
    radar_c = mo.ui.dropdown(options={**{"(None)": None}, **_sorted}, value=None, label="Entity C (optional)")

    mo.hstack([radar_a, radar_b, radar_c], justify='start', gap=2)
    return radar_a, radar_b, radar_c


@app.cell
def _(all_entities, comm_records, datetime, go, mo, radar_a, radar_b, radar_c):
    def _hourly_profile(eid):
        _hours = [0] * 24
        for _r in comm_records:
            if _r['sender'] == eid or _r['receiver'] == eid:
                try:
                    _ts = datetime.fromisoformat(_r['timestamp'].replace('Z', '+00:00'))
                    _hours[_ts.hour] += 1
                except:
                    pass
        _max_v = max(_hours) if max(_hours) > 0 else 1
        return [_h / _max_v for _h in _hours]

    _to_plot = [(radar_a, '#3498DB'), (radar_b, '#E74C3C'), (radar_c, '#2ECC71')]
    _hour_labels = [f"{_h:02d}:00" for _h in range(24)]

    fig_radar = go.Figure()

    for _selector, _color in _to_plot:
        if _selector.value:
            _profile = _hourly_profile(_selector.value)
            _label = all_entities.get(_selector.value, {}).get('label', _selector.value)
            fig_radar.add_trace(go.Scatterpolar(
                r=_profile + [_profile[0]],
                theta=_hour_labels + [_hour_labels[0]],
                fill='toself', fillcolor=_color.replace(')', ', 0.2)').replace('#', 'rgba('),
                line=dict(color=_color, width=2),
                name=_label
            ))

    fig_radar.update_layout(
        title=dict(text='<b>Radar Chart: Temporal Activity Fingerprints</b><br><sup>Hourly activity patterns (normalized)</sup>', x=0.5),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        height=500, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
    )

    mo.vstack([mo.md("### Radar Chart - Temporal Activity Comparison"), fig_radar])
    return (fig_radar,)


# =============================================
# KEY FINDINGS
# =============================================


@app.cell(hide_code=True)
def _(likely_pseudonyms, mo, similarity_df):
    _n_pseudo = len(likely_pseudonyms)
    _top = similarity_df.head(5)

    mo.md(f"""
    ## Key Findings for Question 3

    ### 3.1 Identified Pseudonyms ({_n_pseudo} entities)

    | Pseudonym | Pattern | Investigative Significance |
    |-----------|---------|---------------------------|
    | **Boss** | Title-like | Central command role |
    | **The Lookout** | "The X" | Surveillance operations |
    | **The Middleman** | "The X" | Logistics/broker |
    | **The Accountant** | "The X" | Financial operations |
    | **Mrs. Money** | "Mrs. X" | Financial handler |
    | **The Intern** | "The X" | Junior operative |

    ### 3.2 Visual Analytics Techniques Applied

    | Visualization | Insight Provided |
    |--------------|------------------|
    | **Arc Diagram** | Connection density and hub identification |
    | **Adjacency Matrix** | Complete pairwise communication view |
    | **Parallel Coordinates** | Multi-attribute entity comparison |
    | **Treemap/Icicle** | Communication volume hierarchy |
    | **Force-Directed Network** | Similarity-based entity clustering |
    | **Sankey Diagram** | Pseudonym resolution flow |
    | **Radar Chart** | Temporal activity fingerprints |

    ### 3.3 How Understanding Changes with Pseudonyms

    - **Network consolidation**: Multiple nodes collapse to single actors
    - **Role clarity**: "Boss" = coordinator, "Mrs. Money" = finances
    - **Operational structure**: "The X" pattern suggests organized roles
    - **Investigation priority**: Focus on Boss and Mrs. Money resolution

    ### 3.4 References

    - Bostock, M. "D3.js - Data-Driven Documents"
    - Meeks, E. "D3.js in Action" - Network Visualization
    - IEEE VAST Challenge visualization techniques
    - Plotly documentation for hierarchical and network charts
    """)
    return


if __name__ == "__main__":
    app.run()
