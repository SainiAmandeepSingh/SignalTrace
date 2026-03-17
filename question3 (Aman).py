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
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


# =============================================
# QUESTION 3: PSEUDONYM DETECTION & ENTITY RESOLUTION
# Advanced Visual Analytics for Knowledge Graph Investigation
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3: Pseudonym Detection and Entity Resolution

    ## Research Questions
    
    *It was noted by Clepper's intern that some people and vessels are using pseudonyms to communicate.*

    1. **Who is using pseudonyms to communicate, and what are these pseudonyms?**
       - Known pseudonyms include "Boss" and "The Lookout", but there appear to be many more.
       - Pseudonyms may be used by multiple people or vessels.
    
    2. **How do visualizations help Clepper identify common entities?**
    
    3. **How does understanding of activities change with pseudonym knowledge?**

    ---

    ## Visual Analytics Approach

    This notebook implements **research-backed visual analytics techniques** for entity resolution:

    | Technique | Purpose | Reference |
    |-----------|---------|-----------|
    | **Pseudonym Detection Heuristics** | Identify naming patterns suggesting aliases | Pattern matching on "The X", "Mrs. X", title-only names |
    | **Jaccard Similarity Analysis** | Find entities with overlapping communication partners | Bilgic et al., "D-Dupe: Entity Resolution in Social Networks" (IEEE VAST 2006) |
    | **Bipartite Communication Network** | Visualize pseudonym-to-partner connections | NetworkX bipartite layouts |
    | **Hierarchical Clustering Heatmap** | Group entities by communication similarity | Scipy hierarchical clustering with dendrogram |
    | **Temporal Activity Fingerprinting** | Compare hourly patterns to identify same-person usage | Heatmap with entity×hour matrix |
    | **Force-Directed Similarity Network** | Interactive exploration of entity clusters | D3.js force layout principles |
    | **Sankey Diagram** | Map pseudonyms to candidate real identities | Flow visualization for entity resolution |
    | **Parallel Coordinates** | Multi-dimensional entity comparison | Plotly parcoords with interactive brushing |

    *References: Bilgic et al. (2006), Bostock D3.js, IEEE VAST Challenge methodologies*
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
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
    from scipy.spatial.distance import pdist, squareform
    return (
        Counter, combinations, datetime, defaultdict, dendrogram, fcluster, ff,
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
        _content = _comm.get('content', '')
        _senders = [_e['source'] for _e in edges_to[_comm_id] if _e.get('type') == 'sent']
        _receivers = [_e['target'] for _e in edges_from[_comm_id] if _e.get('type') == 'received']

        for _sender in _senders:
            for _receiver in _receivers:
                if _sender in entity_ids and _receiver in entity_ids:
                    comm_matrix[_sender][_receiver] += 1
                    comm_records.append({
                        'sender': _sender, 'receiver': _receiver,
                        'timestamp': _timestamp, 'comm_id': _comm_id,
                        'content': _content
                    })

    print(f"Extracted {len(comm_records)} communication records from {len(comm_events)} events")
    return comm_events, comm_matrix, comm_records, edges_from, edges_to


# =============================================
# SECTION 1: PSEUDONYM DETECTION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pseudonym Detection via Naming Pattern Analysis

    We identify potential pseudonyms using heuristic pattern matching on entity names. The patterns detected include:

    | Pattern | Examples | Rationale |
    |---------|----------|-----------|
    | **"The X"** | The Lookout, The Middleman, The Accountant | Role-based aliases commonly used in covert operations |
    | **"Mrs./Mr. X"** | Mrs. Money | Formal title with role-based surname |
    | **Title-only** | Boss, Small Fry | Single descriptive words suggesting rank/role |
    | **Single-word Person** | Sam, Kelly, Davis | Potentially first-name-only aliases |

    This approach is inspired by entity resolution research showing that **naming conventions often reveal organizational structure** (Bilgic et al., IEEE VAST 2006).
    """)
    return


@app.cell
def _(all_entities, pd):
    def detect_pseudonym(eid, edata):
        _label = edata.get('label', eid)
        _sub_type = edata.get('sub_type', '')
        
        _patterns = {
            'the_pattern': _label.lower().startswith('the '),
            'mrs_mr_pattern': _label.lower().startswith(('mrs.', 'mr.', 'mrs ', 'mr ')),
            'single_word_person': len(_label.split()) == 1 and _sub_type == 'Person',
            'title_like': _label in ['Boss', 'Small Fry', 'The Intern'],
        }
        
        _detected_patterns = [_k for _k, _v in _patterns.items() if _v]
        _score = sum(_patterns.values())
        
        return {
            'entity_id': eid,
            'label': _label,
            'sub_type': _sub_type,
            'pseudonym_score': _score,
            'is_likely_pseudonym': _score >= 1,
            'detected_patterns': ', '.join(_detected_patterns) if _detected_patterns else 'none',
            **_patterns
        }

    pseudonym_df = pd.DataFrame([detect_pseudonym(_eid, _ed) for _eid, _ed in all_entities.items()])
    pseudonym_df = pseudonym_df.sort_values('pseudonym_score', ascending=False)
    likely_pseudonyms = pseudonym_df[pseudonym_df['is_likely_pseudonym']].copy()

    print(f"Identified {len(likely_pseudonyms)} likely pseudonyms out of {len(all_entities)} entities")
    return detect_pseudonym, likely_pseudonyms, pseudonym_df


@app.cell
def _(go, likely_pseudonyms, mo):
    # Create a summary visualization of detected pseudonyms
    _pseudo_summary = likely_pseudonyms[['label', 'sub_type', 'detected_patterns', 'pseudonym_score']].copy()
    _pseudo_summary = _pseudo_summary.sort_values('pseudonym_score', ascending=True)
    
    fig_pseudo_bar = go.Figure()
    
    _colors = {'Person': '#4ECDC4', 'Vessel': '#FF6B6B', 'Organization': '#95E1D3', 'Group': '#F38181'}
    
    for _idx, _row in _pseudo_summary.iterrows():
        fig_pseudo_bar.add_trace(go.Bar(
            y=[_row['label']],
            x=[_row['pseudonym_score']],
            orientation='h',
            marker_color=_colors.get(_row['sub_type'], '#999'),
            text=f"{_row['detected_patterns']}",
            textposition='outside',
            hovertemplate=f"<b>{_row['label']}</b><br>Type: {_row['sub_type']}<br>Pattern: {_row['detected_patterns']}<br>Score: {_row['pseudonym_score']}<extra></extra>",
            showlegend=False
        ))
    
    fig_pseudo_bar.update_layout(
        title=dict(text='<b>Detected Pseudonyms by Pattern Score</b><br><sup>Higher score = more pseudonym indicators | Colors indicate entity type</sup>', x=0.5),
        xaxis_title='Pseudonym Score',
        yaxis_title='Entity',
        height=max(400, len(_pseudo_summary) * 35),
        showlegend=False,
        bargap=0.3
    )
    
    mo.vstack([
        mo.md("### Pseudonym Detection Results"),
        fig_pseudo_bar
    ])
    return (fig_pseudo_bar,)


# =============================================
# SECTION 2: JACCARD SIMILARITY COMPUTATION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Entity Similarity Analysis via Communication Partners

    To identify entities that might be the **same person using different pseudonyms**, we compute **Jaccard similarity** based on shared communication partners:

    $$J(A, B) = \frac{|Partners(A) \cap Partners(B)|}{|Partners(A) \cup Partners(B)|}$$

    High Jaccard similarity between two entities suggests they:
    - Communicate with the same people/vessels
    - May be the same person operating under different aliases
    - Share operational roles within the network

    This technique is fundamental to **entity resolution in social networks** (Bilgic et al., 2006).
    """)
    return


@app.cell
def _(all_entities, combinations, comm_matrix, datetime, entity_ids, np, pd):
    def get_partners(eid, comm_mat):
        _sent_to = set(comm_mat.get(eid, {}).keys())
        _recv_from = set(_s for _s, _targets in comm_mat.items() if eid in _targets)
        return _sent_to | _recv_from

    entity_partners = {_eid: get_partners(_eid, comm_matrix) for _eid in entity_ids}

    def jaccard(set_a, set_b):
        if not set_a and not set_b:
            return 0.0
        _union = set_a | set_b
        return len(set_a & set_b) / len(_union) if len(_union) > 0 else 0.0

    # Build list of entities with at least one communication
    entity_list = sorted([_e for _e in entity_ids if len(entity_partners.get(_e, set())) > 0])
    n_entities = len(entity_list)
    entity_to_idx = {_e: _i for _i, _e in enumerate(entity_list)}

    # Compute similarity matrix
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
                'label_b': all_entities[_e2].get('label', _e2),
                'type_a': all_entities[_e1].get('sub_type', 'Unknown'),
                'type_b': all_entities[_e2].get('sub_type', 'Unknown'),
                'shared_partners': len(entity_partners[_e1] & entity_partners[_e2]),
                'total_partners': len(entity_partners[_e1] | entity_partners[_e2])
            })

    similarity_df = pd.DataFrame(_sim_records).sort_values('jaccard', ascending=False)

    def parse_ts(ts_str):
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else None
        except:
            return None

    print(f"Computed similarity for {len(entity_list)} active entities")
    print(f"Found {len(similarity_df)} entity pairs with similarity > 0")
    print(f"Top similarity: {similarity_df['jaccard'].max():.3f}" if len(similarity_df) > 0 else "No pairs found")
    return (
        entity_list, entity_partners, entity_to_idx, get_partners, jaccard,
        n_entities, parse_ts, similarity_df, similarity_matrix
    )


# =============================================
# VISUALIZATION 1: BIPARTITE COMMUNICATION NETWORK
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Bipartite Communication Network

    This visualization shows **pseudonyms on the left** and their **communication partners on the right**. 
    
    - **Node size** = total message volume
    - **Edge thickness** = number of messages between entities
    - **Gold nodes** = identified pseudonyms
    - **Teal nodes** = other entities

    Bipartite layouts are particularly effective for analyzing relationships between two distinct classes of entities (Tom Sawyer Software, 2024).
    """)
    return


@app.cell
def _(all_entities, comm_matrix, comm_records, go, likely_pseudonyms, mo, np):
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    
    # Get communication partners of pseudonyms
    _partner_ids = set()
    _edges = []
    
    for _pid in _pseudonym_ids:
        for _target, _count in comm_matrix.get(_pid, {}).items():
            if _target not in _pseudonym_ids:
                _partner_ids.add(_target)
                _edges.append((_pid, _target, _count))
        for _source, _targets in comm_matrix.items():
            if _pid in _targets and _source not in _pseudonym_ids:
                _partner_ids.add(_source)
                _edges.append((_source, _pid, _targets[_pid]))
    
    # Position pseudonyms on left, partners on right
    _pseudo_list = sorted(list(_pseudonym_ids))
    _partner_list = sorted(list(_partner_ids))
    
    if _pseudo_list and _partner_list:
        _pseudo_y = np.linspace(0, 1, len(_pseudo_list)) if len(_pseudo_list) > 1 else [0.5]
        _partner_y = np.linspace(0, 1, len(_partner_list)) if len(_partner_list) > 1 else [0.5]
        
        _pos = {}
        for _i, _p in enumerate(_pseudo_list):
            _pos[_p] = (0, _pseudo_y[_i])
        for _i, _p in enumerate(_partner_list):
            _pos[_p] = (1, _partner_y[_i])
        
        # Compute node sizes based on message volume
        _node_volumes = {}
        for _eid in _pseudo_list + _partner_list:
            _sent = sum(comm_matrix.get(_eid, {}).values())
            _recv = sum(1 for _r in comm_records if _r['receiver'] == _eid)
            _node_volumes[_eid] = _sent + _recv
        
        _max_vol = max(_node_volumes.values()) if _node_volumes else 1
        
        # Create figure
        fig_bipartite = go.Figure()
        
        # Add edges
        for _src, _tgt, _cnt in _edges:
            if _src in _pos and _tgt in _pos:
                _x0, _y0 = _pos[_src]
                _x1, _y1 = _pos[_tgt]
                fig_bipartite.add_trace(go.Scatter(
                    x=[_x0, _x1], y=[_y0, _y1],
                    mode='lines',
                    line=dict(width=max(0.5, _cnt * 0.3), color='rgba(150,150,150,0.3)'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Add pseudonym nodes (left)
        _px = [_pos[_p][0] for _p in _pseudo_list]
        _py = [_pos[_p][1] for _p in _pseudo_list]
        _plabels = [all_entities[_p].get('label', _p) for _p in _pseudo_list]
        _psizes = [10 + 30 * (_node_volumes.get(_p, 0) / _max_vol) for _p in _pseudo_list]
        
        fig_bipartite.add_trace(go.Scatter(
            x=_px, y=_py, mode='markers+text',
            marker=dict(size=_psizes, color='#FFD700', line=dict(width=2, color='#B8860B')),
            text=_plabels, textposition='middle left', textfont=dict(size=10),
            hovertemplate=[f"<b>{_plabels[_i]}</b><br>Messages: {_node_volumes.get(_pseudo_list[_i], 0)}<extra></extra>" for _i in range(len(_pseudo_list))],
            showlegend=False
        ))
        
        # Add partner nodes (right)
        _rx = [_pos[_p][0] for _p in _partner_list]
        _ry = [_pos[_p][1] for _p in _partner_list]
        _rlabels = [all_entities.get(_p, {}).get('label', _p) for _p in _partner_list]
        _rsizes = [10 + 30 * (_node_volumes.get(_p, 0) / _max_vol) for _p in _partner_list]
        
        fig_bipartite.add_trace(go.Scatter(
            x=_rx, y=_ry, mode='markers+text',
            marker=dict(size=_rsizes, color='#4ECDC4', line=dict(width=1, color='#2E8B8B')),
            text=_rlabels, textposition='middle right', textfont=dict(size=9),
            hovertemplate=[f"<b>{_rlabels[_i]}</b><br>Messages: {_node_volumes.get(_partner_list[_i], 0)}<extra></extra>" for _i in range(len(_partner_list))],
            showlegend=False
        ))
        
        fig_bipartite.update_layout(
            title=dict(text='<b>Bipartite Network: Pseudonyms ↔ Communication Partners</b><br><sup>Gold = Pseudonyms (left) | Teal = Partners (right) | Edge width = message count</sup>', x=0.5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 1.3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=max(500, len(_pseudo_list + _partner_list) * 25),
            plot_bgcolor='white',
            showlegend=False
        )
    else:
        fig_bipartite = go.Figure()
        fig_bipartite.add_annotation(text="No bipartite data available", showarrow=False)
        fig_bipartite.update_layout(height=300)
    
    mo.vstack([mo.md("### Bipartite Communication Network"), fig_bipartite])
    return (fig_bipartite,)


# =============================================
# VISUALIZATION 2: HIERARCHICAL CLUSTERING HEATMAP
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hierarchical Clustering of Entity Similarity

    This heatmap shows **pairwise Jaccard similarity** between all communicating entities, with rows and columns reordered by **hierarchical clustering**.

    - **Bright cells** = high similarity (potential same person)
    - **Clusters along diagonal** = groups of related entities
    - **Dendrogram** shows hierarchical relationships

    This technique reveals hidden clusters of entities that communicate with similar partners, suggesting they may be operated by the same person or coordinated group.

    *Reference: Scipy hierarchical clustering, Mike Bostock "Les Misérables Co-occurrence"*
    """)
    return


@app.cell
def _(all_entities, entity_list, go, leaves_list, likely_pseudonyms, linkage, mo, np, pdist, similarity_matrix):
    if len(entity_list) > 3:
        # Filter to entities with some similarity
        _row_sums = similarity_matrix.sum(axis=1)
        _active_mask = _row_sums > 0
        _active_indices = np.where(_active_mask)[0]
        
        if len(_active_indices) > 3:
            _sub_matrix = similarity_matrix[np.ix_(_active_indices, _active_indices)]
            _sub_entities = [entity_list[_i] for _i in _active_indices]
            _sub_labels = [all_entities.get(_e, {}).get('label', _e)[:15] for _e in _sub_entities]
            
            # Mark pseudonyms
            _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
            _label_colors = ['#FFD700' if _sub_entities[_i] in _pseudonym_ids else '#333333' for _i in range(len(_sub_entities))]
            
            # Hierarchical clustering
            _dist = pdist(_sub_matrix + 0.001)
            _linkage_mat = linkage(_dist, method='average')
            _order = leaves_list(_linkage_mat)
            
            # Reorder matrix and labels
            _ordered_matrix = _sub_matrix[_order, :][:, _order]
            _ordered_labels = [_sub_labels[_i] for _i in _order]
            _ordered_colors = [_label_colors[_i] for _i in _order]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=_ordered_matrix,
                x=_ordered_labels,
                y=_ordered_labels,
                colorscale='Viridis',
                hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Jaccard Similarity: %{z:.3f}<extra></extra>',
                colorbar=dict(title='Jaccard<br>Similarity')
            ))
            
            fig_heatmap.update_layout(
                title=dict(text='<b>Entity Similarity Heatmap with Hierarchical Clustering</b><br><sup>Bright = high similarity | Ordered by communication pattern similarity</sup>', x=0.5),
                height=700, width=800,
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8), autorange='reversed')
            )
        else:
            fig_heatmap = go.Figure()
            fig_heatmap.add_annotation(text="Insufficient similar entities for clustering", showarrow=False)
            fig_heatmap.update_layout(height=300)
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.add_annotation(text="Insufficient data for heatmap", showarrow=False)
        fig_heatmap.update_layout(height=300)
    
    mo.vstack([mo.md("### Similarity Heatmap - Clustered by Communication Patterns"), fig_heatmap])
    return (fig_heatmap,)


# =============================================
# VISUALIZATION 3: TEMPORAL ACTIVITY MATRIX
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Temporal Activity Fingerprint Matrix

    This heatmap shows **hourly activity patterns** for each entity:
    
    - **Rows** = entities (sorted by total activity)
    - **Columns** = hours of the day (0-23)
    - **Color intensity** = normalized message count per hour
    
    Entities with **non-overlapping temporal patterns** could be the same person (one person can't communicate as two aliases simultaneously). Conversely, entities active at the **same hours** are likely different people.

    *This temporal fingerprinting technique is commonly used in social network analysis for identity disambiguation.*
    """)
    return


@app.cell
def _(all_entities, comm_records, datetime, go, likely_pseudonyms, mo, np, pd):
    # Build hourly activity matrix for all active entities
    _activity_data = []
    _entity_totals = {}
    
    for _rec in comm_records:
        try:
            _ts = datetime.fromisoformat(_rec['timestamp'].replace('Z', '+00:00'))
            _hour = _ts.hour
            _activity_data.append({'entity': _rec['sender'], 'hour': _hour})
            _activity_data.append({'entity': _rec['receiver'], 'hour': _hour})
        except:
            pass
    
    _activity_df = pd.DataFrame(_activity_data)
    
    if len(_activity_df) > 0:
        _pivot = _activity_df.groupby(['entity', 'hour']).size().unstack(fill_value=0)
        
        # Normalize each row
        _row_sums = _pivot.sum(axis=1)
        _active_entities = _row_sums[_row_sums > 5].index.tolist()  # Filter low-activity
        
        if len(_active_entities) > 3:
            _pivot_filtered = _pivot.loc[_active_entities]
            _pivot_norm = _pivot_filtered.div(_pivot_filtered.max(axis=1), axis=0).fillna(0)
            
            # Sort by total activity
            _pivot_norm = _pivot_norm.loc[_pivot_filtered.sum(axis=1).sort_values(ascending=False).index]
            
            # Get labels
            _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
            _labels = []
            for _eid in _pivot_norm.index:
                _label = all_entities.get(_eid, {}).get('label', _eid)[:15]
                if _eid in _pseudonym_ids:
                    _label = f"★ {_label}"
                _labels.append(_label)
            
            # Ensure all 24 hours
            _hours = list(range(24))
            _matrix = np.zeros((len(_pivot_norm), 24))
            for _i, _eid in enumerate(_pivot_norm.index):
                for _h in _pivot_norm.columns:
                    if 0 <= _h < 24:
                        _matrix[_i, _h] = _pivot_norm.loc[_eid, _h]
            
            fig_temporal = go.Figure(data=go.Heatmap(
                z=_matrix,
                x=[f"{_h:02d}:00" for _h in _hours],
                y=_labels,
                colorscale='YlOrRd',
                hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Activity: %{z:.2f}<extra></extra>',
                colorbar=dict(title='Normalized<br>Activity')
            ))
            
            fig_temporal.update_layout(
                title=dict(text='<b>Temporal Activity Fingerprints</b><br><sup>★ = Pseudonym | Non-overlapping patterns may indicate same person</sup>', x=0.5),
                xaxis_title='Hour of Day',
                yaxis_title='Entity',
                height=max(400, len(_labels) * 20),
                yaxis=dict(tickfont=dict(size=9))
            )
        else:
            fig_temporal = go.Figure()
            fig_temporal.add_annotation(text="Insufficient active entities", showarrow=False)
            fig_temporal.update_layout(height=300)
    else:
        fig_temporal = go.Figure()
        fig_temporal.add_annotation(text="No temporal data available", showarrow=False)
        fig_temporal.update_layout(height=300)
    
    mo.vstack([mo.md("### Temporal Activity Matrix"), fig_temporal])
    return (fig_temporal,)


# =============================================
# VISUALIZATION 4: INTERACTIVE FORCE-DIRECTED NETWORK
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Interactive Similarity Network

    This **force-directed graph** positions entities based on their communication similarity:
    
    - **Edges** connect entities with Jaccard similarity ≥ threshold
    - **Node proximity** indicates similar communication patterns
    - **Gold nodes** = identified pseudonyms
    - **Interactive slider** controls similarity threshold
    
    Use the slider to explore different levels of entity clustering. Higher thresholds reveal only the strongest potential identity matches.

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
        
        # Draw edges
        _edge_x, _edge_y = [], []
        for _e in _G.edges():
            _x0, _y0 = _pos[_e[0]]
            _x1, _y1 = _pos[_e[1]]
            _edge_x.extend([_x0, _x1, None])
            _edge_y.extend([_y0, _y1, None])
        
        # Draw nodes
        _node_x, _node_y, _colors, _sizes, _texts, _hovers = [], [], [], [], [], []
        _type_colors = {'Person': '#4ECDC4', 'Vessel': '#FF6B6B', 'Organization': '#95E1D3', 'Group': '#F38181'}
        
        for _n in _G.nodes(data=True):
            _x, _y = _pos[_n[0]]
            _node_x.append(_x)
            _node_y.append(_y)
            _label = _n[1].get('label', _n[0])
            _texts.append(_label)
            _hovers.append(f"<b>{_label}</b><br>Type: {_n[1].get('sub_type', 'Unknown')}<br>Connections: {_G.degree(_n[0])}")
            
            if _n[0] in _pseudonym_ids:
                _colors.append('#FFD700')
                _sizes.append(20)
            else:
                _colors.append(_type_colors.get(_n[1].get('sub_type', ''), '#999'))
                _sizes.append(12)
        
        fig_force = go.Figure()
        fig_force.add_trace(go.Scatter(
            x=_edge_x, y=_edge_y, mode='lines',
            line=dict(width=1, color='rgba(150,150,150,0.4)'),
            hoverinfo='none'
        ))
        fig_force.add_trace(go.Scatter(
            x=_node_x, y=_node_y, mode='markers+text',
            marker=dict(size=_sizes, color=_colors, line=dict(width=1, color='white')),
            text=_texts, textposition='top center', textfont=dict(size=9),
            hovertext=_hovers, hoverinfo='text'
        ))
        
        fig_force.update_layout(
            title=dict(text=f'<b>Similarity Network (threshold ≥ {_thresh:.2f})</b><br><sup>Nodes: {len(_G.nodes)} | Edges: {len(_G.edges)} | Gold = Pseudonym</sup>', x=0.5),
            height=600, showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#fafafa'
        )
    else:
        fig_force = go.Figure()
        fig_force.add_annotation(text=f"No entity pairs with similarity ≥ {_thresh:.2f}", showarrow=False)
        fig_force.update_layout(height=300)
    
    mo.vstack([mo.md(f"### Force-Directed Network (Threshold = {_thresh:.2f})"), fig_force])
    return (fig_force,)


# =============================================
# VISUALIZATION 5: SANKEY DIAGRAM FOR RESOLUTION
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Sankey Diagram: Pseudonym Resolution Flow

    This **Sankey diagram** visualizes potential entity resolution mappings:
    
    - **Left side (Gold)**: Identified pseudonyms
    - **Right side (Teal)**: Candidate real identities
    - **Flow width**: Proportional to Jaccard similarity score
    
    Wider flows indicate stronger evidence that the pseudonym and candidate may be the same entity.

    *Reference: D3.js Sankey layout for path analysis and flow visualization*
    """)
    return


@app.cell
def _(go, likely_pseudonyms, mo, similarity_df):
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    _flows = []
    
    for _, _row in similarity_df.head(40).iterrows():
        _is_pa = _row['entity_a'] in _pseudonym_ids
        _is_pb = _row['entity_b'] in _pseudonym_ids
        
        # We want flows FROM pseudonym TO candidate
        if _is_pa and not _is_pb:
            _flows.append({'source': _row['label_a'], 'target': _row['label_b'] + ' ', 'value': _row['jaccard']})
        elif _is_pb and not _is_pa:
            _flows.append({'source': _row['label_b'], 'target': _row['label_a'] + ' ', 'value': _row['jaccard']})
        elif _is_pa and _is_pb:
            # Both are pseudonyms - show connection
            _flows.append({'source': _row['label_a'], 'target': _row['label_b'] + ' ', 'value': _row['jaccard']})
    
    if _flows:
        _nodes = list(set([_f['source'] for _f in _flows] + [_f['target'] for _f in _flows]))
        _node_idx = {_n: _i for _i, _n in enumerate(_nodes)}
        _node_colors = ['#FFD700' if not _n.endswith(' ') else '#4ECDC4' for _n in _nodes]
        
        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color='white', width=1),
                label=_nodes,
                color=_node_colors
            ),
            link=dict(
                source=[_node_idx[_f['source']] for _f in _flows],
                target=[_node_idx[_f['target']] for _f in _flows],
                value=[_f['value'] * 100 for _f in _flows],
                color='rgba(255, 215, 0, 0.3)',
                hovertemplate='%{source.label} → %{target.label}<br>Similarity: %{value:.1f}%<extra></extra>'
            )
        ))
        
        fig_sankey.update_layout(
            title=dict(text='<b>Sankey: Pseudonym Resolution Candidates</b><br><sup>Gold = Pseudonyms | Teal = Candidate identities | Flow width = similarity</sup>', x=0.5),
            height=500, font=dict(size=10)
        )
    else:
        fig_sankey = go.Figure()
        fig_sankey.add_annotation(text="No resolution candidates found", showarrow=False)
        fig_sankey.update_layout(height=300)
    
    mo.vstack([mo.md("### Sankey Diagram - Entity Resolution Flow"), fig_sankey])
    return (fig_sankey,)


# =============================================
# VISUALIZATION 6: PARALLEL COORDINATES
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Parallel Coordinates: Multi-Dimensional Entity Comparison

    This visualization displays each entity as a **polyline** crossing multiple parallel axes:
    
    - **Entity Type**: Person, Vessel, Organization, Group
    - **Pseudonym Score**: Higher = more alias indicators
    - **Messages Sent/Received**: Communication volume
    - **Unique Partners**: Network breadth
    - **Active Hours**: Temporal spread
    
    **Interactive brushing**: Click and drag on any axis to filter entities. Gold lines indicate pseudonyms.

    *Reference: Plotly Parallel Coordinates for multidimensional exploratory analysis*
    """)
    return


@app.cell
def _(all_entities, comm_matrix, comm_records, datetime, entity_partners, go, mo, pd, pseudonym_df):
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
        
        if _sent + _received > 0:  # Only include active entities
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
    
    if len(_features_df) > 0:
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
                dict(range=[0, max(1, _features_df['pseudonym_score'].max())], label='Pseudonym<br>Score',
                     values=_features_df['pseudonym_score']),
                dict(range=[0, max(1, _features_df['messages_sent'].max())], label='Msgs<br>Sent',
                     values=_features_df['messages_sent']),
                dict(range=[0, max(1, _features_df['messages_received'].max())], label='Msgs<br>Received',
                     values=_features_df['messages_received']),
                dict(range=[0, max(1, _features_df['unique_partners'].max())], label='Unique<br>Partners',
                     values=_features_df['unique_partners']),
                dict(range=[0, 24], label='Active<br>Hours', values=_features_df['active_hours'])
            ]
        ))
        
        fig_parallel.update_layout(
            title=dict(text='<b>Parallel Coordinates: Entity Feature Comparison</b><br><sup>Each line = one entity | Gold = Pseudonym | Drag on axes to filter</sup>', x=0.5),
            height=500, margin=dict(l=100, r=100)
        )
    else:
        fig_parallel = go.Figure()
        fig_parallel.add_annotation(text="No entity data available", showarrow=False)
        fig_parallel.update_layout(height=300)
    
    mo.vstack([mo.md("### Parallel Coordinates - Multi-Dimensional Entity Analysis"), fig_parallel])
    return (fig_parallel,)


# =============================================
# VISUALIZATION 7: TOP SIMILARITY PAIRS TABLE
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Top Entity Resolution Candidates

    This table shows the **highest-similarity entity pairs**, ranked by Jaccard similarity. These are the strongest candidates for being the same person/entity using different names.
    """)
    return


@app.cell
def _(likely_pseudonyms, mo, similarity_df):
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    
    # Filter to pairs involving at least one pseudonym
    _relevant = similarity_df[
        (similarity_df['entity_a'].isin(_pseudonym_ids)) |
        (similarity_df['entity_b'].isin(_pseudonym_ids))
    ].head(15).copy()
    
    if len(_relevant) > 0:
        _display_df = _relevant[['label_a', 'label_b', 'jaccard', 'shared_partners', 'total_partners']].copy()
        _display_df.columns = ['Entity A', 'Entity B', 'Jaccard Similarity', 'Shared Partners', 'Total Partners']
        _display_df['Jaccard Similarity'] = _display_df['Jaccard Similarity'].round(3)
        
        _table = mo.ui.table(_display_df, selection=None)
        mo.vstack([
            mo.md("### Top Resolution Candidates (Involving Pseudonyms)"),
            _table
        ])
    else:
        mo.md("No resolution candidates found involving identified pseudonyms.")
    return


# =============================================
# KEY FINDINGS
# =============================================


@app.cell(hide_code=True)
def _(likely_pseudonyms, mo, similarity_df):
    _n_pseudo = len(likely_pseudonyms)
    _top_pairs = similarity_df.head(5) if len(similarity_df) > 0 else None
    
    mo.md(f"""
    ## Key Findings for Question 3

    ### 3.1 Who is Using Pseudonyms? ({_n_pseudo} entities identified)

    Through naming pattern analysis, we identified the following likely pseudonyms:

    | Pseudonym | Pattern | Investigative Significance |
    |-----------|---------|---------------------------|
    | **Boss** | Title-like | Central command/coordination role |
    | **The Lookout** | "The X" | Surveillance operations |
    | **The Middleman** | "The X" | Logistics/brokerage role |
    | **The Accountant** | "The X" | Financial operations |
    | **Mrs. Money** | "Mrs. X" | Financial handler |
    | **The Intern** | "The X" | Junior operative |
    | **Small Fry** | Title-like | Minor player/low rank |
    | **Sam, Kelly, Davis, Elise** | Single-word Person | First-name-only aliases |

    ### 3.2 How Do Visualizations Help Clepper?

    | Visualization | Insight Provided |
    |--------------|------------------|
    | **Bipartite Network** | Shows which entities pseudonyms communicate with directly |
    | **Similarity Heatmap** | Reveals clusters of entities with overlapping communication patterns |
    | **Temporal Matrix** | Identifies entities with non-overlapping schedules (could be same person) |
    | **Force-Directed Network** | Interactive exploration of entity similarity clusters |
    | **Sankey Diagram** | Maps pseudonyms to candidate real identities |
    | **Parallel Coordinates** | Multi-dimensional comparison of entity attributes |

    ### 3.3 How Does Understanding Change with Pseudonyms?

    With pseudonym awareness, Clepper can:

    1. **Consolidate the network**: Multiple pseudonyms may collapse to fewer actual people
    2. **Identify roles**: "The X" pattern suggests organized operational structure
    3. **Prioritize investigation**: Focus on resolving "Boss" and "Mrs. Money" as likely key figures
    4. **Detect coordination**: Entities with high Jaccard similarity but non-overlapping temporal patterns are strong candidates for being the same person
    5. **Map organizational hierarchy**: Title-like names (Boss, The Intern, Small Fry) suggest rank structure

    ---

    ## References

    1. Bilgic, M., Licamele, L., Getoor, L., & Shneiderman, B. (2006). "D-Dupe: An Interactive Tool for Entity Resolution in Social Networks." *IEEE Symposium on Visual Analytics Science and Technology*, pp. 43-50.
    
    2. Bostock, M., Ogievetsky, V., & Heer, J. (2011). "D3: Data-Driven Documents." *IEEE Trans. Visualization & Computer Graphics*.
    
    3. IEEE VAST Challenge (2025). Visual Analytics Science and Technology Challenge. https://vast-challenge.github.io/2025/
    
    4. Plotly Technologies Inc. (2024). Plotly Python Graphing Library. https://plotly.com/python/
    
    5. NetworkX Developers. (2024). NetworkX: Network Analysis in Python. https://networkx.org/
    """)
    return


if __name__ == "__main__":
    app.run()
