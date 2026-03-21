import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # <center> Mini-Challenge 3 </center>

    ## **Team Members**

    1. AmanDeep Singh
    2. Dominic van den Bungelaar
    3. Kim Wilmink

    ## **Background**
    Over the past decade, the community of Oceanus has faced numerous transformations and challenges evolving from its fishing-centric origins. Following major crackdowns on illegal fishing activities, suspects have shifted investments into more regulated sectors such as the ocean tourism industry, resulting in growing tensions. This increased tourism has recently attracted the likes of international pop star Sailor Shift, who announced plans to film a music video on the island.

    Clepper Jessen, a former analyst at FishEye and now a seasoned journalist for the Hacklee Herald, has been keenly observing these rising tensions. Recently, he turned his attention towards the temporary closure of Nemo Reef. By listening to radio communications and utilizing his investigative tools, Clepper uncovered a complex web of expedited approvals and secretive logistics. These efforts revealed a story involving high-level Oceanus officials, Sailor Shift's team, local influential families, and local conservationist group The Green Guardians, pointing towards a story of corruption and manipulation.

    Our task is to develop new and novel visualizations and visual analytics approaches to help Clepper get to the bottom of this story.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Tasks & Questions**

    Clepper diligently recorded all intercepted radio communications over the last two weeks. With the help of his intern, they have analyzed their content to identify important events and relationships between key players. The result is a knowledge graph describing the last two weeks on Oceanus. Clepper and his intern have spent a large amount of time generating this knowledge graph, and they would now like some assistance using it to answer the following questions.

    1. Clepper found that messages frequently came in at around the same time each day.
        - Develop a graph-based visual analytics approach to identify any daily temporal patterns in communications.
        - How do these patterns shift over the two weeks of observations?
        - Focus on a specific entity and use this information to determine who has influence over them.

    2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.
        - Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.
        - Are there groups that are more closely associated? If so, what are the topic areas that are predominant for each group?
              - For example, these groupings could be related to: Environmentalism (known associates of Green Guardians), Sailor Shift, and fishing/leisure vessels.

    3. It was noted by Clepper's intern that some people and vessels are using pseudonyms to communicate.
        - Expanding upon your prior visual analytics, determine who is using pseudonyms to communicate, and what these pseudonyms are.
              - Some that Clepper has already identified include: "Boss", and "The Lookout", but there appear to be many more.
              - To complicate the matter, pseudonyms may be used by multiple people or vessels.
        - Describe how your visualizations make it easier for Clepper to identify common entities in the knowledge graph.
        - How does your understanding of activities change given your understanding of pseudonyms?

    4. Clepper suspects that Nadia Conti, who was formerly entangled in an illegal fishing scheme, may have continued illicit activity within Oceanus.

        - Through visual analytics, provide evidence that Nadia is, or is not, doing something illegal.
        - Summarize Nadia's actions visually. Are Clepper's suspicions justified?
    """)
    return


@app.cell
def _():
    import json as json_lib
    import marimo as mo
    from networkx.readwrite import json_graph
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import altair as alt
    import networkx as nx
    from collections import Counter, defaultdict
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import time

    return (
        alt,
        datetime,
        defaultdict,
        go,
        json_graph,
        json_lib,
        make_subplots,
        mo,
        np,
        nx,
        pd,
    )


@app.cell
def _(json_graph, json_lib):
    with open("data/MC3_graph.json", "r") as _f:
        graph_data = json_lib.load(_f)
    G = json_graph.node_link_graph(graph_data, edges="edges")
    return G, graph_data


@app.cell
def _(graph_data):
    nodes_by_id = {n['id']: n for n in graph_data['nodes']}

    persons = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Person'}
    vessels = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Vessel'}
    organizations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Organization'}
    groups = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Group'}
    locations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Location'}

    # all_entities excludes locations (for communication network — focus on actors)
    all_entities = {**persons, **vessels, **organizations, **groups}
    entity_ids = set(all_entities.keys())

    # all_entities_full includes locations (for type lookups and relationship extraction)
    all_entities_full = {**persons, **vessels, **organizations, **groups, **locations}
    entity_ids_with_locations = set(all_entities_full.keys())

    print(f"Loaded: {len(persons)} persons, {len(vessels)} vessels, {len(organizations)} organizations, {len(groups)} groups, {len(locations)} locations")
    print(f"Total entities of interest (excl. locations): {len(all_entities)}")
    print(f"Total entities including locations: {len(all_entities_full)}")
    return (
        all_entities,
        all_entities_full,
        entity_ids,
        entity_ids_with_locations,
        nodes_by_id,
    )


@app.cell
def _(defaultdict, entity_ids, entity_ids_with_locations, graph_data):
    # Build edge lookup structures
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)
    for edge in graph_data['edges']:
        edges_to[edge['target']].append(edge)
        edges_from[edge['source']].append(edge)

    # Extract Communication events
    comm_events = [n for n in graph_data['nodes'] if n.get('sub_type') == 'Communication']

    # Build communication matrix: comm_matrix[sender][receiver] = list of {timestamp, content, comm_id}
    comm_matrix = defaultdict(lambda: defaultdict(list))

    for _comm in comm_events:
        _comm_id = _comm['id']
        _timestamp = _comm.get('timestamp', '')
        _content = _comm.get('content', '')

        # Senders: edges TO communication with type 'sent'
        _senders = [e['source'] for e in edges_to[_comm_id] if e.get('type') == 'sent']
        # Receivers: edges FROM communication with type 'received'
        _receivers = [e['target'] for e in edges_from[_comm_id] if e.get('type') == 'received']

        for _sender in _senders:
            for _receiver in _receivers:
                # AND ensures both endpoints are actors (Person/Vessel/Org/Group).
                # OR would admit Location nodes as senders/receivers, inflating counts
                # by ~87 phantom entries (e.g. "Nemo Reef → Mako").
                if _sender in entity_ids and _receiver in entity_ids:
                    comm_matrix[_sender][_receiver].append({
                        'timestamp': _timestamp,
                        'content': _content,
                        'comm_id': _comm_id
                    })

    # Extract Relationship nodes
    _relationships_raw = [n for n in graph_data['nodes'] if n['type'] == 'Relationship']

    relationship_data = []
    for _rel in _relationships_raw:
        _rel_id = _rel['id']
        _rel_type = _rel['sub_type']

        _sources = [e['source'] for e in edges_to[_rel_id] if e['source'] in entity_ids_with_locations]
        _targets = [e['target'] for e in edges_from[_rel_id] if e['target'] in entity_ids_with_locations]

        if _rel_type in ['Colleagues', 'Friends']:
            if len(_sources) >= 2:
                relationship_data.append({
                    'type': _rel_type,
                    'entity1': _sources[0],
                    'entity2': _sources[1],
                    'bidirectional': True,
                    'rel_id': _rel_id
                })
        else:
            for _s in _sources:
                for _t in _targets:
                    relationship_data.append({
                        'type': _rel_type,
                        'entity1': _s,
                        'entity2': _t,
                        'bidirectional': False,
                        'rel_id': _rel_id
                    })

    print(f"Extracted {len(comm_events)} communications and {len(relationship_data)} formal relationship edges")
    print(f"  (relationship edges = entity-pair expansions of {len(_relationships_raw)} relationship nodes in the graph)")
    return comm_events, comm_matrix, edges_from, edges_to, relationship_data


@app.cell
def _(comm_events, edges_from, edges_to, nodes_by_id, pd):
    _messages = []
    for _comm in comm_events:
        _comm_id = _comm['id']
        _senders = [e['source'] for e in edges_to[_comm_id] if e.get('type') == 'sent']
        _receivers = [e['target'] for e in edges_from[_comm_id] if e.get('type') == 'received']
        if _senders and _receivers:
            _source_node = nodes_by_id.get(_senders[0], {})
            _target_node = nodes_by_id.get(_receivers[0], {})
            _messages.append({
                'event_id': _comm_id,
                'datetime': _comm.get('timestamp', ''),
                'content': _comm.get('content', ''),
                'source': _source_node.get('name', _senders[0]),
                'target': _target_node.get('name', _receivers[0]),
            })
    messages_df = pd.DataFrame(_messages)
    return (messages_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1: Daily Temporal Patterns in Communications
    """)
    return


@app.cell
def _(pd):
    df_intents = pd.read_csv("data/categories_v2.csv")
    print(f"Loaded {len(df_intents)} classified messages")
    df_intents.head()
    return (df_intents,)


@app.cell
def _(alt, df_intents, mo):
    _intent_bar = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("count()", title="Number of Messages"),
        y=alt.Y("category:N", title="Category", sort="-x"),
        color=alt.Color("category:N", legend=None),
    ).properties(
        title="Overall Category Distribution",
        width=600,
        height=300
    )
    mo.vstack([mo.md("### Category Distribution"), _intent_bar])
    return


@app.cell
def _(alt, df_intents, mo):
    _intent_entity = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("count()", title="Count"),
        y=alt.Y("sender_name:N", title="Entity", sort="-x"),
        color=alt.Color("category:N", title="Category"),
    ).properties(
        title="Message Categories by Sender",
        width=700,
        height=500
    )
    mo.vstack([mo.md("### Category by Entity"), _intent_entity])
    return


@app.cell
def _(G, alt, mo, pd):
    import re as _re

    _self_msgs = []
    for _n, _a in G.nodes(data=True):
        if _a.get("sub_type") != "Communication":
            continue

        _sender = _receiver = None
        for _pred in G.predecessors(_n):
            if G.nodes[_pred].get("type") == "Entity" and G.edges[_pred, _n].get("type") == "sent":
                _sender = G.nodes[_pred].get("name", _pred)
        for _succ in G.successors(_n):
            if G.nodes[_succ].get("type") == "Entity" and G.edges[_n, _succ].get("type") == "received":
                _receiver = G.nodes[_succ].get("name", _succ)

        if _sender and _sender == _receiver:
            _content = _a.get("content", "")
            _match = _re.match(r"^([\w\s.]+),\s+(?:it's\s+)?([\w\s.]+?)\s+(?:here|reporting|responding|checking|this is)", _content)
            _actual = _match.group(2).strip() if _match else "???"
            _addressee = _match.group(1).strip() if _match else _sender

            _self_msgs.append({
                "node_id": _n,
                "graph_sender": _sender,
                "graph_receiver": _receiver,
                "actual_sender": _actual,
                "actual_receiver": _sender,
                "timestamp": _a.get("timestamp", ""),
                "content": _content[:200],
                "mislabeled": _actual != _sender,
            })

    _df_self = pd.DataFrame(_self_msgs)

    # Summary chart: how often each actual sender is hidden
    _chart_actual = alt.Chart(_df_self).mark_bar().encode(
        x=alt.X("count()", title="Messages where they are the hidden sender"),
        y=alt.Y("actual_sender:N", title="Actual Sender (from content)", sort="-x"),
        color=alt.Color("graph_sender:N", title="Graph labels it as"),
        tooltip=["actual_sender", "graph_sender", "count()"]
    ).properties(title="Hidden Senders: who actually sent the 31 self-labeled messages?", width=600, height=350)

    # Summary chart: which graph entities receive self-messages
    _chart_graph = alt.Chart(_df_self).mark_bar().encode(
        x=alt.X("count()", title="Self-messages received"),
        y=alt.Y("graph_sender:N", title="Graph Entity (sender = receiver)", sort="-x"),
        color=alt.Color("actual_sender:N", title="Actual sender"),
        tooltip=["graph_sender", "actual_sender", "count()"]
    ).properties(title="Which entities have misattributed messages?", width=600, height=300)

    # Build styled HTML table
    _rows_html = ""
    for _, _r in _df_self.sort_values("timestamp").iterrows():
        _color = "#ffe0e0" if _r["mislabeled"] else "#e0ffe0"
        _rows_html += f"""<tr style="background:{_color}">
            <td style="padding:4px 8px;font-size:11px">{_r['timestamp']}</td>
            <td style="padding:4px 8px;font-size:11px"><strong>{_r['graph_sender']}</strong> → {_r['graph_receiver']}</td>
            <td style="padding:4px 8px;font-size:11px;color:#c00"><strong>{_r['actual_sender']}</strong> → {_r['actual_receiver']}</td>
            <td style="padding:4px 8px;font-size:11px">{_r['content'][:120]}…</td>
        </tr>"""

    _table_html = mo.md(f"""
    <div style="max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:6px">
    <table style="width:100%;border-collapse:collapse">
    <thead style="position:sticky;top:0;background:#f5f5f5">
    <tr>
    <th style="padding:6px 8px;text-align:left;font-size:11px;border-bottom:2px solid #ddd">Timestamp</th>
    <th style="padding:6px 8px;text-align:left;font-size:11px;border-bottom:2px solid #ddd">Graph Says</th>
    <th style="padding:6px 8px;text-align:left;font-size:11px;border-bottom:2px solid #ddd">Actually</th>
    <th style="padding:6px 8px;text-align:left;font-size:11px;border-bottom:2px solid #ddd">Content</th>
    </tr></thead>
    <tbody>{_rows_html}</tbody>
    </table></div>
    """)

    mo.vstack([
        mo.md(f"""### Self-Message Audit: {len(_df_self)} messages where sender = receiver in graph
    These messages have the same entity as both sender and receiver in the knowledge graph,
    but the message content reveals a different actual sender (radio-style: *"RecipientName, ActualSender here..."*).
    Red rows = mislabeled, green = correctly labeled."""),
        mo.hstack([_chart_actual, _chart_graph]),
        _table_html,
    ])
    return


@app.cell
def _(df_intents, mo):
    _unique_cats = ["All"] + sorted(df_intents["category"].unique().tolist())
    category_dropdown = mo.ui.dropdown(options=_unique_cats, value="All", label="Filter by Category")

    _entity_types = ["All"] + sorted(set(
        [t for t in df_intents["sender_type"].unique().tolist() if t] +
        [t for t in df_intents["receiver_type"].unique().tolist() if t]
    ))
    entity_type_dropdown = mo.ui.dropdown(options=_entity_types, value="All", label="Filter by Entity Type")

    _all_entities = sorted(set(
        df_intents["sender_name"].dropna().unique().tolist() +
        df_intents["receiver_name"].dropna().unique().tolist()
    ))
    entity_dropdown = mo.ui.multiselect(options=_all_entities, value=[], label="Filter by Entity")

    suspicion_slider = mo.ui.slider(
        start=0, stop=10, step=1, value=0,
        label="Min. Suspicion",
        show_value=True
    )
    return (
        category_dropdown,
        entity_dropdown,
        entity_type_dropdown,
        suspicion_slider,
    )


@app.cell
def _(
    category_dropdown,
    df_intents,
    entity_dropdown,
    entity_type_dropdown,
    json_lib,
    mo,
    suspicion_slider,
):
    _selected_cat = category_dropdown.value
    _min_suspicion = suspicion_slider.value

    # Apply filters
    _df_filtered = df_intents.copy()

    if _selected_cat != "All":
        _df_filtered = _df_filtered[_df_filtered["category"] == _selected_cat]

    if entity_type_dropdown.value != "All":
        _df_filtered = _df_filtered[
            (_df_filtered["sender_type"] == entity_type_dropdown.value) |
            (_df_filtered["receiver_type"] == entity_type_dropdown.value)
        ]

    if len(entity_dropdown.value) > 0:
        _selected_entities = list(entity_dropdown.value)
        _df_filtered = _df_filtered[
            (_df_filtered["sender_name"].isin(_selected_entities)) |
            (_df_filtered["receiver_name"].isin(_selected_entities))
        ]

    if _min_suspicion > 0:
        _df_filtered = _df_filtered[_df_filtered["suspicion"] >= _min_suspicion]

    _unique_dates = json_lib.dumps(sorted(df_intents["date_str"].unique().tolist()))

    _category_colors = {
        "routine operations": "#5B9BD5",
        "environmental monitoring": "#3cb44b",
        "permit and regulatory": "#42d4f4",
        "covert coordination": "#f58231",
        "illegal activity": "#e6194b",
        "surveillance and intelligence": "#911eb4",
        "cover story": "#ffe119",
        "music and tourism": "#f032e6",
        "interpersonal and social": "#a9a9a9",
        "command and control": "#800000",
        "unknown": "#999999"
    }

    # Build filtered records for D3
    _records = []
    for _, _row in _df_filtered.iterrows():
        _records.append({
            "node_id": str(_row["node_id"]),
            "sender_name": str(_row["sender_name"]),
            "receiver_name": str(_row["receiver_name"]),
            "sender_type": str(_row["sender_type"]),
            "receiver_type": str(_row["receiver_type"]),
            "date_str": str(_row["date_str"]),
            "hour_float": float(_row["hour_float"]),
            "timestamp": str(_row["timestamp"]),
            "category": str(_row["category"]),
            "suspicion": int(_row["suspicion"]) if str(_row["suspicion"]).isdigit() else 0,
            "content": str(_row["content"]),
            "category_color": _category_colors.get(str(_row["category"]), "#999"),
        })

    # Build ALL records (unfiltered) for chat history and ego network
    _all_records = []
    for _, _row in df_intents.iterrows():
        _all_records.append({
            "node_id": str(_row["node_id"]),
            "sender_name": str(_row["sender_name"]),
            "receiver_name": str(_row["receiver_name"]),
            "sender_type": str(_row["sender_type"]),
            "receiver_type": str(_row["receiver_type"]),
            "date_str": str(_row["date_str"]),
            "hour_float": float(_row["hour_float"]),
            "timestamp": str(_row["timestamp"]),
            "category": str(_row["category"]),
            "suspicion": int(_row["suspicion"]) if str(_row["suspicion"]).isdigit() else 0,
            "content": str(_row["content"]),
            "category_color": _category_colors.get(str(_row["category"]), "#999"),
        })

    _filtered_json = json_lib.dumps(_records)
    _all_json = json_lib.dumps(_all_records)
    _category_colors_json = json_lib.dumps(_category_colors)
    _msg_count = len(_df_filtered)

    # === THE BIG COMBINED D3 IFRAME ===
    _dashboard = mo.iframe(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #fafafa; }}
    #container {{ display: flex; width: 100%; height: 830px; }}
    #timeline-panel {{ width: 60%; height: 100%; overflow-y: auto; border-right: 2px solid #ddd; background: white; }}
    #right-panel {{ width: 40%; height: 100%; display: flex; flex-direction: column; }}
    #chat-panel {{ height: 50%; border-bottom: 2px solid #ddd; display: flex; flex-direction: column; background: white; }}
    #ego-panel {{ height: 50%; background: white; position: relative; }}
    #chat-header {{ padding: 6px 10px; background: #f0f0f0; border-bottom: 1px solid #ddd;
                    font-weight: bold; font-size: 12px; flex-shrink: 0; }}
    #chat-tabs {{ display: flex; gap: 0; border-bottom: 1px solid #ddd; flex-shrink: 0; }}
    .chat-tab {{ padding: 5px 12px; font-size: 11px; cursor: pointer; border: none;
                 background: #f0f0f0; border-bottom: 2px solid transparent; color: #666;
                 transition: all 0.15s; }}
    .chat-tab:hover {{ background: #e8e8e8; }}
    .chat-tab.active {{ background: white; color: #333; font-weight: bold;
                        border-bottom: 2px solid #5B9BD5; }}
    #chat-messages {{ flex: 1; overflow-y: auto; padding: 8px; }}
    .chat-msg {{ padding: 8px 10px; margin: 4px 0; border-radius: 8px; font-size: 12px;
                 border-left: 4px solid #ccc; background: #f9f9f9; cursor: pointer; transition: all 0.15s; }}
    .chat-msg:hover {{ background: #eef; }}
    .chat-msg.highlighted {{ background: #fff3cd; border-left-color: #ff5555; box-shadow: 0 0 6px rgba(255,85,85,0.3); }}
    .chat-msg .full-content {{ white-space: pre-wrap; word-break: break-word; }}
    .chat-msg .meta {{ font-size: 10px; color: #888; margin-top: 3px; }}
    .chat-msg .sender {{ font-weight: bold; }}
    .chat-msg .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
                        font-size: 9px; color: white; margin-left: 4px; }}
    .tooltip {{
        position: fixed; background: white; border: 1px solid #ccc; border-radius: 6px;
        padding: 8px 12px; font-size: 12px; pointer-events: none;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15); display: none; max-width: 450px; z-index: 1000;
    }}
    #ego-title {{ position: absolute; top: 4px; left: 10px; font-size: 12px; font-weight: bold; color: #333; z-index: 10; }}
    #chat-empty {{ padding: 30px; text-align: center; color: #aaa; font-size: 13px; }}
    #ego-empty {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
                  color: #aaa; font-size: 13px; text-align: center; }}
    </style>
    </head>
    <body>
    <div id="container">
    <div id="timeline-panel"><div id="chart"></div></div>
    <div id="right-panel">
        <div id="chat-panel">
            <div id="chat-header">Message History <span id="chat-entity" style="color:#555"></span></div>
            <div id="chat-tabs">
                <button class="chat-tab active" id="tab-all" onclick="switchTab('all')">All from Sender</button>
                <button class="chat-tab" id="tab-convo" onclick="switchTab('convo')">Conversation</button>
            </div>
            <div id="chat-messages"><div id="chat-empty">Click a message in the timeline to see conversation history</div></div>
        </div>
        <div id="ego-panel">
            <div id="ego-title">Ego Network</div>
            <div id="ego-empty">Click a message to see the sender's communication network</div>
            <svg id="ego-svg" width="100%" height="100%"></svg>
        </div>
    </div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
    try {{

    var filteredData = {_filtered_json};
    var allData = {_all_json};
    var allDates = {_unique_dates};
    var categoryColors = {_category_colors_json};

    var typeColors = {{
    "Person": "#7B68EE",
    "Organization": "#DC143C",
    "Vessel": "#00CED1",
    "Group": "#FF8C00",
    "Location": "#4169E1"
    }};

    // ============================================================
    // LEFT PANEL: TIMELINE
    // ============================================================
    var margin = {{top: 60, right: 15, bottom: 35, left: 55}};
    var rowHeight = 55;
    var summaryHeight = 50;
    var tlWidth = document.getElementById("timeline-panel").offsetWidth - margin.left - margin.right - 10;
    if (tlWidth < 400) tlWidth = 620;
    var tlHeight = allDates.length * rowHeight;
    var totalHeight = tlHeight + summaryHeight;
    var dotSize = 9;
    var dotGap = 2;
    var maxCols = 6;

    var svg = d3.select("#chart")
    .append("svg")
    .attr("width", tlWidth + margin.left + margin.right)
    .attr("height", totalHeight + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var x = d3.scaleLinear().domain([8, 15]).range([0, tlWidth]);
    var y = d3.scaleBand().domain(allDates).range([0, tlHeight]).padding(0.1);
    var tickVals = [8,9,10,11,12,13,14,15];

    svg.append("g")
    .call(d3.axisTop(x).tickValues(tickVals).tickFormat(function(d) {{ return d + ":00"; }}))
    .selectAll("text").style("font-size", "11px");

    tickVals.forEach(function(h) {{
    svg.append("line").attr("x1", x(h)).attr("x2", x(h))
        .attr("y1", 0).attr("y2", totalHeight)
        .attr("stroke", "#eee").attr("stroke-width", 1);
    }});

    allDates.forEach(function(date, i) {{
    svg.append("rect").attr("x", 0).attr("y", y(date))
        .attr("width", tlWidth).attr("height", y.bandwidth())
        .attr("fill", i % 2 === 0 ? "#f9f9f9" : "#ffffff").attr("stroke", "#eee");
    svg.append("text").attr("x", -6).attr("y", y(date) + y.bandwidth() / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .style("font-size", "10px").style("font-weight", "bold").text(i + 1);
    }});

    // Density curves
    allDates.forEach(function(date) {{
    var dayData = filteredData.filter(function(d) {{ return d.date_str === date; }});
    if (dayData.length === 0) return;
    var values = dayData.map(function(d) {{ return d.hour_float; }});
    var bins = d3.bin().domain([8, 15.5]).thresholds(25)(values);
    var maxCount = d3.max(bins, function(b) {{ return b.length; }});
    var areaY = d3.scaleLinear().domain([0, maxCount || 1])
        .range([y(date) + y.bandwidth(), y(date) + y.bandwidth() * 0.2]);
    var area = d3.area()
        .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
        .y0(y(date) + y.bandwidth())
        .y1(function(b) {{ return areaY(b.length); }})
        .curve(d3.curveBasis);
    svg.append("path").datum(bins).attr("d", area).attr("fill", "#5B9BD5").attr("opacity", 0.15);
    }});

    // Group by date+hour
    var grouped = {{}};
    filteredData.forEach(function(d) {{
    var hour = Math.floor(d.hour_float);
    var key = d.date_str + "|" + hour;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(d);
    }});

    var tooltip = d3.select("#tooltip");
    var allDotGroups = [];

    // Draw clickable dots
    Object.keys(grouped).forEach(function(key) {{
    var parts = key.split("|");
    var date = parts[0];
    var hour = parseInt(parts[1]);
    var items = grouped[key];
    var binLeft = x(hour) + 4;
    var rowTop = y(date);

    items.forEach(function(d, idx) {{
        var col = idx % maxCols;
        var row = Math.floor(idx / maxCols);
        var px = binLeft + col * (dotSize + dotGap);
        var py = rowTop + row * (dotSize + dotGap);

        var g = svg.append("g").style("cursor", "pointer").datum(d);

        g.append("rect").attr("class", "dot-rect")
            .attr("x", px).attr("y", py)
            .attr("width", dotSize).attr("height", dotSize)
            .attr("fill", d.category_color).attr("rx", 2)
            .attr("stroke", "#fff").attr("stroke-width", 0.5);

        g.on("mouseover", function(event) {{
            d3.select(this).select(".dot-rect").attr("stroke", "#333").attr("stroke-width", 2);
            tooltip.style("display", "block")
                .html(
                    "<strong>" + d.sender_name + " &rarr; " + d.receiver_name + "</strong><br/>"
                    + d.timestamp + "<br/>"
                    + "<strong>Category:</strong> " + d.category
                    + " <span style='background:" + (categoryColors[d.category]||"#999")
                    + ";color:#fff;padding:1px 5px;border-radius:3px;font-size:10px'>"
                    + d.suspicion + "/10</span><br/>"
                    + "<div style='max-height:200px;overflow-y:auto;margin-top:4px;font-style:italic;white-space:pre-wrap'>" + d.content + "</div>"
                )
                .style("left", (event.clientX + 14) + "px")
                .style("top", (event.clientY - 20) + "px");
        }})
        .on("mouseout", function() {{
            d3.select(this).select(".dot-rect").attr("stroke", "#fff").attr("stroke-width", 0.5);
            tooltip.style("display", "none");
        }})
        .on("click", function(event, datum) {{
            onMessageClick(datum);
        }});

        allDotGroups.push({{g: g, d: d}});
    }});
    }});

    // Summary row
    var summaryTop = tlHeight + 10;
    svg.append("rect").attr("x", 0).attr("y", summaryTop)
    .attr("width", tlWidth).attr("height", summaryHeight)
    .attr("fill", "#f0f0f0").attr("stroke", "#ddd");
    svg.append("text").attr("x", -6).attr("y", summaryTop + summaryHeight / 2)
    .attr("text-anchor", "end").attr("dominant-baseline", "middle")
    .style("font-size", "10px").style("font-weight", "bold").text("All");

    var allValues = filteredData.map(function(d) {{ return d.hour_float; }});
    if (allValues.length > 0) {{
    var allBins = d3.bin().domain([8, 15.5]).thresholds(40)(allValues);
    var allMax = d3.max(allBins, function(b) {{ return b.length; }});
    var summaryY = d3.scaleLinear().domain([0, allMax || 1])
        .range([summaryTop + summaryHeight, summaryTop + 10]);
    var summaryArea = d3.area()
        .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
        .y0(summaryTop + summaryHeight)
        .y1(function(b) {{ return summaryY(b.length); }})
        .curve(d3.curveBasis);
    svg.append("path").datum(allBins).attr("d", summaryArea)
        .attr("fill", "#5B9BD5").attr("opacity", 0.35);
    svg.append("path").datum(allBins)
        .attr("d", d3.line()
            .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
            .y(function(b) {{ return summaryY(b.length); }})
            .curve(d3.curveBasis))
        .attr("fill", "none").attr("stroke", "#5B9BD5").attr("stroke-width", 2);
    }}

    svg.append("g")
    .attr("transform", "translate(0," + (summaryTop + summaryHeight) + ")")
    .call(d3.axisBottom(x).tickValues(tickVals).tickFormat(function(d) {{ return d + ":00"; }}))
    .selectAll("text").style("font-size", "11px");

    // Legends
    var cleg = svg.append("g").attr("transform", "translate(0,-50)");
    cleg.append("text").attr("x", 0).attr("y", 0).text("Category:").style("font-size", "9px").style("font-weight", "bold");
    Object.entries(categoryColors).forEach(function(e, i) {{
    var col = i % 5;
    var row = Math.floor(i / 5);
    cleg.append("rect").attr("x", 60 + col * 125).attr("y", -8 + row * 14)
        .attr("width", 9).attr("height", 9).attr("fill", e[1]).attr("rx", 1);
    cleg.append("text").attr("x", 60 + col * 125 + 12).attr("y", row * 14).text(e[0]).style("font-size", "8px");
    }});

    // ============================================================
    // CLICK HANDLER — updates chat box + ego network
    // ============================================================
    function onMessageClick(d) {{
    updateChatBox(d);
    updateEgoNetwork(d);

    // Highlight the clicked dot in timeline
    allDotGroups.forEach(function(item) {{
        if (item.d.node_id === d.node_id) {{
            item.g.select(".dot-rect").attr("stroke", "#ff0000").attr("stroke-width", 2.5);
        }} else {{
            item.g.select(".dot-rect").attr("stroke", "#fff").attr("stroke-width", 0.5);
        }}
    }});
    }}

    // ============================================================
    // RIGHT TOP: CHAT BOX (tabbed)
    // ============================================================
    var currentClickedMsg = null;
    var currentTab = "all";

    function switchTab(tab) {{
    currentTab = tab;
    document.getElementById("tab-all").className = "chat-tab" + (tab === "all" ? " active" : "");
    document.getElementById("tab-convo").className = "chat-tab" + (tab === "convo" ? " active" : "");
    if (currentClickedMsg) renderChat(currentClickedMsg);
    }}

    function updateChatBox(clickedMsg) {{
    currentClickedMsg = clickedMsg;
    renderChat(clickedMsg);
    }}

    function renderChat(clickedMsg) {{
    var entity = clickedMsg.sender_name;
    var receiver = clickedMsg.receiver_name;

    if (currentTab === "all") {{
        document.getElementById("chat-entity").textContent = "— all from " + entity;
    }} else {{
        document.getElementById("chat-entity").textContent = "— " + entity + " ↔ " + receiver;
    }}

    // Filter messages based on active tab
    var history;
    if (currentTab === "all") {{
        history = allData.filter(function(m) {{
            return m.sender_name === entity || m.receiver_name === entity;
        }});
    }} else {{
        history = allData.filter(function(m) {{
            return (m.sender_name === entity && m.receiver_name === receiver)
                || (m.sender_name === receiver && m.receiver_name === entity);
        }});
    }}

    history.sort(function(a, b) {{
        return a.timestamp.localeCompare(b.timestamp);
    }});

    var container = document.getElementById("chat-messages");
    container.innerHTML = "";

    if (history.length === 0) {{
        container.innerHTML = "<div style='padding:20px;text-align:center;color:#aaa'>No messages found</div>";
        return;
    }}

    history.forEach(function(m) {{
        var div = document.createElement("div");
        var isClicked = m.node_id === clickedMsg.node_id;
        div.className = "chat-msg" + (isClicked ? " highlighted" : "");
        div.setAttribute("data-nodeid", m.node_id);

        var isSender = m.sender_name === entity;
        var catColor = categoryColors[m.category] || "#999";
        var suspColor = m.suspicion >= 7 ? "#e6194b" : m.suspicion >= 4 ? "#f58231" : "#3cb44b";

        div.style.borderLeftColor = catColor;

        var contentText = m.content;

        div.innerHTML =
            "<span class='sender' style='color:" + (isSender ? "#333" : "#666") + "'>"
            + m.sender_name + " &rarr; " + m.receiver_name + "</span>"
            + "<span class='badge' style='background:" + suspColor + "'>" + m.suspicion + "/10</span>"
            + "<span class='badge' style='background:" + catColor + "'>" + m.category + "</span>"
            + "<div class='full-content' style='margin-top:4px;font-size:12px;color:#333'>"
            + contentText + "</div>"
            + "<div class='meta'>" + m.timestamp + "</div>";

        div.addEventListener("click", function() {{
            onMessageClick(m);
        }});

        container.appendChild(div);
    }});

    // Scroll to highlighted
    var highlighted = container.querySelector(".highlighted");
    if (highlighted) {{
        highlighted.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}
    }}

    // ============================================================
    // RIGHT BOTTOM: EGO NETWORK (hub-and-spoke, ego in center)
    // ============================================================
    function updateEgoNetwork(clickedMsg) {{
    var entity = clickedMsg.sender_name;
    document.getElementById("ego-title").textContent = "Network — " + entity;
    var emptyEl = document.getElementById("ego-empty");
    if (emptyEl) emptyEl.remove();

    // Gather ALL messages involving this entity
    var entityMsgs = allData.filter(function(m) {{
        return m.sender_name === entity || m.receiver_name === entity;
    }});

    // Build partner stats
    var partnerMap = {{}};
    entityMsgs.forEach(function(m) {{
        var partner = m.sender_name === entity ? m.receiver_name : m.sender_name;
        var direction = m.sender_name === entity ? "out" : "in";
        if (!partnerMap[partner]) {{
            partnerMap[partner] = {{ name: partner, type: "", out: 0, in: 0,
                categories: {{}}, maxSusp: 0, totalSusp: 0, totalMsgs: 0 }};
        }}
        partnerMap[partner][direction]++;
        partnerMap[partner].totalMsgs++;
        partnerMap[partner].totalSusp += (m.suspicion || 0);
        partnerMap[partner].categories[m.category] = (partnerMap[partner].categories[m.category] || 0) + 1;
        if (m.suspicion > partnerMap[partner].maxSusp) partnerMap[partner].maxSusp = m.suspicion;
    }});

    // Get types
    allData.forEach(function(m) {{
        if (partnerMap[m.sender_name]) partnerMap[m.sender_name].type = m.sender_type;
        if (partnerMap[m.receiver_name]) partnerMap[m.receiver_name].type = m.receiver_type;
    }});

    var partners = Object.values(partnerMap);
    var nPartners = partners.length;

    // Clear
    var egoSvg = d3.select("#ego-svg");
    egoSvg.selectAll("*").remove();

    var egoEl = document.getElementById("ego-panel");
    var egoW = egoEl.offsetWidth || 400;
    var egoH = egoEl.offsetHeight || 350;
    var egoR = Math.min(egoW, egoH) / 2 - 65;
    var ecx = egoW / 2;
    var ecy = egoH / 2;

    egoSvg.attr("viewBox", "0 0 " + egoW + " " + egoH);

    // Arrow markers
    var defs = egoSvg.append("defs");
    defs.append("marker").attr("id", "ego-arrow-out")
        .attr("viewBox", "0 0 10 10").attr("refX", 26).attr("refY", 5)
        .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
        .append("path").attr("d", "M 0 0 L 10 5 L 0 10 Z").attr("fill", "#555");

    var egoTooltip = d3.select("#tooltip");

    // Get ego type
    var egoType = "";
    allData.forEach(function(m) {{
        if (m.sender_name === entity) egoType = m.sender_type;
    }});

    // Position partners in a circle around center
    partners.forEach(function(p, i) {{
        var ang = (2 * Math.PI * i / nPartners) - Math.PI / 2;
        p.px = ecx + Math.cos(ang) * egoR;
        p.py = ecy + Math.sin(ang) * egoR;
        p.ang = ang;
    }});

    // Draw edges (center to each partner)
    partners.forEach(function(p) {{
        var total = p.out + p.in;
        var avgSusp = p.totalMsgs > 0 ? (p.totalSusp / p.totalMsgs).toFixed(1) : "0";
        var dominantCat = Object.entries(p.categories).sort(function(a,b){{return b[1]-a[1];}})[0][0];
        var edgeColor = categoryColors[dominantCat] || "#999";
        var suspColor = parseFloat(avgSusp) >= 7 ? "#e6194b" : parseFloat(avgSusp) >= 4 ? "#f58231" : "#3cb44b";

        var dx = p.px - ecx;
        var dy = p.py - ecy;
        var dist = Math.sqrt(dx*dx + dy*dy) || 1;

        if (p.out > 0 && p.in > 0) {{
            // Bidirectional: curve both slightly apart
            var offset = 14;
            var mx1 = (ecx+p.px)/2 + (-dy/dist)*offset;
            var my1 = (ecy+p.py)/2 + (dx/dist)*offset;
            var mx2 = (ecx+p.px)/2 + (dy/dist)*offset;
            var my2 = (ecy+p.py)/2 + (-dx/dist)*offset;

            egoSvg.append("path")
                .attr("d","M"+ecx+","+ecy+" Q"+mx1+","+my1+" "+p.px+","+p.py)
                .attr("fill","none").attr("stroke",edgeColor)
                .attr("stroke-width",Math.max(1.5,Math.min(p.out*1.3,5)))
                .attr("opacity",0.65).attr("marker-end","url(#ego-arrow-out)");
            egoSvg.append("path")
                .attr("d","M"+p.px+","+p.py+" Q"+mx2+","+my2+" "+ecx+","+ecy)
                .attr("fill","none").attr("stroke",edgeColor)
                .attr("stroke-width",Math.max(1,Math.min(p.in*1.1,4)))
                .attr("opacity",0.45).attr("stroke-dasharray","5,3");
        }} else if (p.out > 0) {{
            egoSvg.append("line")
                .attr("x1",ecx).attr("y1",ecy).attr("x2",p.px).attr("y2",p.py)
                .attr("stroke",edgeColor)
                .attr("stroke-width",Math.max(1.5,Math.min(p.out*1.3,5)))
                .attr("opacity",0.65).attr("marker-end","url(#ego-arrow-out)");
        }} else {{
            egoSvg.append("line")
                .attr("x1",p.px).attr("y1",p.py).attr("x2",ecx).attr("y2",ecy)
                .attr("stroke",edgeColor)
                .attr("stroke-width",Math.max(1,Math.min(p.in*1.1,4)))
                .attr("opacity",0.45).attr("stroke-dasharray","5,3");
        }}

        // Invisible wide hover target for edge tooltip
        egoSvg.append("line")
            .attr("x1",ecx).attr("y1",ecy).attr("x2",p.px).attr("y2",p.py)
            .attr("stroke","transparent").attr("stroke-width",14)
            .style("cursor","pointer")
            .on("mouseover",function(event) {{
                var catHTML = Object.entries(p.categories)
                    .sort(function(a,b){{return b[1]-a[1];}})
                    .map(function(kv){{
                        return "<span style='color:"+(categoryColors[kv[0]]||"#999")+"'>&#9632;</span> "+kv[0]+": "+kv[1];
                    }}).join("<br/>");
                egoTooltip.style("display","block")
                    .html(
                        "<strong>"+entity+" &harr; "+p.name+"</strong><br/>"
                        +"<div style='margin:4px 0;padding:4px 8px;background:#f5f5f5;border-radius:4px'>"
                        +"<span style='font-size:16px;font-weight:bold'>"+total+"</span> messages"
                        +" &nbsp;&middot;&nbsp; "
                        +"<span style='font-size:10px'>sent "+p.out+" / received "+p.in+"</span>"
                        +"</div>"
                        +"<div style='margin:3px 0'>"
                        +"Avg suspicion: <strong style='color:"+suspColor+"'>"+avgSusp+"/10</strong>"
                        +" &nbsp;&middot;&nbsp; Max: <strong>"+p.maxSusp+"/10</strong>"
                        +"</div>"
                        +"<div style='margin-top:4px;font-size:11px'>"+catHTML+"</div>"
                    )
                    .style("left",(event.clientX+14)+"px")
                    .style("top",(event.clientY-20)+"px");
            }})
            .on("mouseout",function(){{ egoTooltip.style("display","none"); }});

        // Count label on edge midpoint
        if (total > 1) {{
            var labelDist = dist * 0.45;
            var lx = ecx + (dx/dist)*labelDist;
            var ly = ecy + (dy/dist)*labelDist;
            egoSvg.append("text").attr("x",lx).attr("y",ly-5)
                .attr("text-anchor","middle").style("font-size","9px").style("fill","#555")
                .style("pointer-events","none").text(total);
        }}
    }});

    // Center ego node (on top)
    egoSvg.append("circle").attr("cx",ecx).attr("cy",ecy).attr("r",16)
        .attr("fill",typeColors[egoType]||"#999")
        .attr("stroke","#333").attr("stroke-width",2.5);
    egoSvg.append("circle").attr("cx",ecx).attr("cy",ecy).attr("r",22)
        .attr("fill","none").attr("stroke",typeColors[egoType]||"#999")
        .attr("stroke-width",1.5).attr("opacity",0.3);
    egoSvg.append("text").attr("x",ecx).attr("y",ecy+30)
        .attr("text-anchor","middle").style("font-size","11px").style("font-weight","bold")
        .style("pointer-events","none")
        .text(entity.length>20 ? entity.substring(0,20)+"\u2026" : entity);

    // Partner nodes
    partners.forEach(function(p) {{
        var suspBorder = p.maxSusp >= 7 ? "#e6194b" : p.maxSusp >= 4 ? "#f58231" : "#666";
        var r = 9;
        var g = egoSvg.append("g")
            .attr("transform","translate("+p.px+","+p.py+")")
            .style("cursor","pointer");
        g.append("circle").attr("r",r)
            .attr("fill",typeColors[p.type]||"#999")
            .attr("stroke",suspBorder)
            .attr("stroke-width",p.maxSusp>=4?2.5:1.2);

        var anchor = (p.ang>Math.PI/2||p.ang<-Math.PI/2) ? "end" : "start";
        var lxOff = Math.cos(p.ang)*(r+6);
        var lyOff = Math.sin(p.ang)*(r+6);
        egoSvg.append("text")
            .attr("x",p.px+lxOff).attr("y",p.py+lyOff)
            .attr("text-anchor",anchor).attr("dominant-baseline","middle")
            .style("font-size","9px").style("fill","#333")
            .style("pointer-events","none")
            .text(p.name.length>16?p.name.substring(0,16)+"\u2026":p.name);

        g.on("mouseover",function(event) {{
            d3.select(this).select("circle").attr("stroke","#333").attr("stroke-width",2.5);
            var avgS = p.totalMsgs>0?(p.totalSusp/p.totalMsgs).toFixed(1):"0";
            egoTooltip.style("display","block")
                .html("<strong>"+p.name+"</strong><br/>"+p.type
                    +"<br/>"+p.totalMsgs+" msgs with "+entity
                    +"<br/>Avg susp: "+avgS+"/10 &middot; Max: "+p.maxSusp+"/10")
                .style("left",(event.clientX+12)+"px")
                .style("top",(event.clientY-20)+"px");
        }})
        .on("mouseout",function() {{
            d3.select(this).select("circle")
                .attr("stroke",suspBorder)
                .attr("stroke-width",p.maxSusp>=4?2.5:1.2);
            egoTooltip.style("display","none");
        }})
        .on("click",function() {{
            var fakeMsg = allData.find(function(m){{ return m.sender_name===p.name; }})
                || allData.find(function(m){{ return m.receiver_name===p.name; }});
            if(fakeMsg) {{
                var newMsg = Object.assign({{}},fakeMsg);
                newMsg.sender_name = p.name;
                onMessageClick(newMsg);
            }}
        }});
    }});

    // Legend
    var eLeg = egoSvg.append("g").attr("transform","translate(6,"+(egoH-50)+")");
    eLeg.append("text").attr("x",0).attr("y",0).style("font-size","10px").style("font-weight","bold").text("Entity type:");
    Object.entries(typeColors).forEach(function(e,i) {{
        eLeg.append("circle").attr("cx",8+i*78).attr("cy",14).attr("r",5).attr("fill",e[1]);
        eLeg.append("text").attr("x",16+i*78).attr("y",18).style("font-size","9px").text(e[0]);
    }});
    eLeg.append("line").attr("x1",0).attr("y1",32).attr("x2",18).attr("y2",32)
        .attr("stroke","#888").attr("stroke-width",2);
    eLeg.append("text").attr("x",22).attr("y",36).style("font-size","9px").text("sent");
    eLeg.append("line").attr("x1",56).attr("y1",32).attr("x2",74).attr("y2",32)
        .attr("stroke","#888").attr("stroke-width",1.5).attr("stroke-dasharray","4,3");
    eLeg.append("text").attr("x",78).attr("y",36).style("font-size","9px").text("received");
    eLeg.append("circle").attr("cx",120).attr("cy",32).attr("r",5)
        .attr("fill","none").attr("stroke","#e6194b").attr("stroke-width",2);
    eLeg.append("text").attr("x",128).attr("y",36).style("font-size","9px").text("susp ≥ 7");
    eLeg.append("circle").attr("cx",185).attr("cy",32).attr("r",5)
        .attr("fill","none").attr("stroke","#f58231").attr("stroke-width",2);
    eLeg.append("text").attr("x",193).attr("y",36).style("font-size","9px").text("susp ≥ 4");
    }}

    }} catch(e) {{
    document.getElementById("chart").innerHTML = "<pre style='color:red'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height="850px")

    # === Summary statistics ===
    _mean_susp = round(_df_filtered["suspicion"].mean(), 1) if len(_df_filtered) > 0 else 0
    _max_susp_entity = ""
    _max_susp_val = 0
    if len(_df_filtered) > 0:
        _entity_susp = _df_filtered.groupby("sender_name")["suspicion"].mean()
        if len(_entity_susp) > 0:
            _max_susp_entity = _entity_susp.idxmax()
            _max_susp_val = round(_entity_susp.max(), 1)

    _top_cats = _df_filtered["category"].value_counts().head(4)
    _cat_html = "".join([
        f"<span style='display:inline-block;margin:1px 2px;padding:1px 6px;border-radius:3px;"
        f"font-size:10px;background:{_category_colors.get(c, '#999')};color:white'>"
        f"{c}: {n}</span>"
        for c, n in _top_cats.items()
    ])

    _n_high = len(_df_filtered[_df_filtered["suspicion"] >= 7])
    _n_entities = len(set(
        _df_filtered["sender_name"].tolist() + _df_filtered["receiver_name"].tolist()
    ))

    _stats_panel = mo.md(f"""
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <div style="text-align:center;padding:3px 10px;background:#f5f5f5;border-radius:5px;border:1px solid #e0e0e0">
    <div style="font-size:18px;font-weight:bold;color:#333">{_msg_count}</div>
    <div style="font-size:9px;color:#888">Messages</div></div>
    <div style="text-align:center;padding:3px 10px;background:#f5f5f5;border-radius:5px;border:1px solid #e0e0e0">
    <div style="font-size:18px;font-weight:bold;color:#333">{_n_entities}</div>
    <div style="font-size:9px;color:#888">Entities</div></div>
    <div style="text-align:center;padding:3px 10px;background:#f5f5f5;border-radius:5px;border:1px solid #e0e0e0">
    <div style="font-size:18px;font-weight:bold;color:{'#e6194b' if _mean_susp >= 5 else '#f58231' if _mean_susp >= 3 else '#3cb44b'}">{_mean_susp}</div>
    <div style="font-size:9px;color:#888">Avg Suspicion</div></div>
    <div style="text-align:center;padding:3px 10px;background:#f5f5f5;border-radius:5px;border:1px solid #e0e0e0">
    <div style="font-size:18px;font-weight:bold;color:#e6194b">{_n_high}</div>
    <div style="font-size:9px;color:#888">High Risk (≥7)</div></div>
    <div style="text-align:center;padding:3px 10px;background:#f5f5f5;border-radius:5px;border:1px solid #e0e0e0">
    <div style="font-size:13px;font-weight:bold;color:#333">{_max_susp_entity[:16]}</div>
    <div style="font-size:9px;color:#888">Top Suspect (avg {_max_susp_val})</div></div>
    </div>
    <div style="margin-top:2px">{_cat_html}</div>
    """)

    mo.vstack([
        mo.md(f"### Communication Intelligence Dashboard"),
        mo.hstack([
            mo.vstack([
                mo.hstack([category_dropdown, entity_type_dropdown, entity_dropdown, suspicion_slider]),
            ]),
            _stats_panel,
        ], justify="space-between", align="start"),
        _dashboard,
    ])
    return


@app.cell
def _(alt, df_intents, mo):
    _intent_heatmap = alt.Chart(df_intents).mark_rect().encode(
        x=alt.X("date_str:O", title="Date"),
        y=alt.Y("category:N", title="Category"),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), title="Count"),
        tooltip=["date_str", "category", "count()"]
    ).properties(title="Category Frequency Over Time", width=700, height=350)
    mo.vstack([mo.md("### Category Heatmap — Date x Category"), _intent_heatmap])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Findings Question 1

    1. Clepper found that messages frequently came in at around the same time each day.
    - Develop a graph-based visual analytics approach to identify any daily temporal patterns in communications.
    - How do these patterns shift over the two weeks of observations?

    Firstly we filter all the messages that have a suspicion score less than 4. We do this as we are most interested in the communications that might involve illegal activity. After filtering out these messages, we do not find a clear pattern shift in the communications between all the entities in the graph. However, when filtering for example for Boss, you can see that they are actively sending and receiving messages during the first week, while in the second week they only receive them. For Mrs. Money, she sends and receives 35 out of 38 messages during the first 9 days. In the next 5 days, she is only involved in 3 messages.

    - Focus on a specific entity and use this information to determine who has influence over them.

      For this question we choose to further investigate Mrs. Money, as her messages have an average suspicion score of 6.5, indicating that she might be involved in illegal activities. When investigating her communication network, you can see that most of her communications are with Boss and The Intern. If you then look further into the message history, it becomes quite clear that Mrs. Money reports to boss, while The Intern reports to Mrs. Money. The communications with other entities show mostly covert coordination, but no clear signs of influence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2.1: Understanding Interactions Between Vessels and People

    - *2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.*
        - *a. Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.*

    To answer these questions, we created interactive visualization to explore:
    1. **Communication Network** - Who talks to whom and how frequently?
    2. **Communication Frequency Matrix** - Heatmap of message intensity between entities
    3. **Formal Relationship Network** - Structural relationships (Colleagues, Operates, Reports, etc.)
    4. **Entity Profiles** - Deep dive into individual actors
    5. **Communication Timeline** - Temporal patterns over the two-week period
    6. **Key Statistics** - Summary metrics and top communicators
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2. Communication Network**

    In this visualization, we show the ***communication flow*** between all entities (persons, vessels, and organizations). This answers the fundamental question: **who talks to whom and how often?**

    **Visual Encodings:**
    - **Node Size** = Communication activity (larger nodes = more messages sent/received)
    - **Edge Thickness** = Message frequency between two entities
    - **Node Color** = Entity type (Teal = Person, Coral Red = Vessel, Mint = Organization, Salmon = Group)
    """)
    return


@app.cell
def _(mo):
    node_type_filter = mo.ui.multiselect(
        options=['Person', 'Vessel', 'Organization', 'Group'],
        value=['Person', 'Vessel', 'Organization'],
        label="Show Entity Types:"
    )

    min_comm_slider = mo.ui.slider(
        start=1, stop=20, value=2, step=1,
        label="Minimum Communications to Show Edge:"
    )

    mo.hstack([node_type_filter, min_comm_slider], justify='start', gap=2)
    return min_comm_slider, node_type_filter


@app.cell
def _(
    all_entities,
    comm_matrix,
    json_lib,
    min_comm_slider,
    mo,
    node_type_filter,
):
    _node_color_map = {
        'Person':       '#4ECDC4',
        'Vessel':       '#FF6B6B',
        'Organization': '#95E1D3',
        'Group':        '#F38181',
    }

    _filtered_entities = {
        eid: e for eid, e in all_entities.items()
        if e.get('sub_type') in node_type_filter.value
    }

    # Build edges above threshold
    _edges_raw = {}
    for _s in comm_matrix:
        if _s not in _filtered_entities:
            continue
        for _r, _comms in comm_matrix[_s].items():
            if _r not in _filtered_entities:
                continue
            _w = len(_comms)
            if _w >= min_comm_slider.value:
                _edges_raw[(_s, _r)] = _w

    # Per-node activity totals (from visible edges only)
    _sent_counts = {}
    _recv_counts = {}
    for (_s, _r), _w in _edges_raw.items():
        _sent_counts[_s] = _sent_counts.get(_s, 0) + _w
        _recv_counts[_r] = _recv_counts.get(_r, 0) + _w

    _active_nodes = set(_sent_counts) | set(_recv_counts)

    _nodes_data = [
        {
            "id": nid,
            "sub_type": _filtered_entities[nid].get("sub_type", "Unknown"),
            "total_sent":     _sent_counts.get(nid, 0),
            "total_received": _recv_counts.get(nid, 0),
        }
        for nid in _active_nodes
        if nid in _filtered_entities
    ]
    _edges_data = [
        {"source": s, "target": r, "weight": w}
        for (s, r), w in _edges_raw.items()
    ]

    _nodes_json      = json_lib.dumps(_nodes_data)
    _edges_json      = json_lib.dumps(_edges_data)
    _node_colors_json = json_lib.dumps(_node_color_map)
    _n_nodes = len(_nodes_data)
    _n_edges = len(_edges_data)

    mo.iframe(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #fafafa; }}
    #container {{ display: flex; width: 100%; height: 740px; }}
    #graph-wrap {{ flex: 1; height: 100%; background: white; overflow: hidden; position: relative; }}
    #detail-panel {{
        width: 280px; height: 100%; background: white;
        border-left: 2px solid #ddd; display: flex; flex-direction: column;
    }}
    #detail-header {{
        padding: 8px 12px; background: #f0f0f0; border-bottom: 1px solid #ddd;
        font-weight: bold; font-size: 13px; flex-shrink: 0;
    }}
    #detail-body {{ flex: 1; overflow-y: auto; padding: 10px 12px; }}
    #detail-empty {{ color: #aaa; font-size: 13px; text-align: center; margin-top: 60px; }}
    .partner-row {{
        padding: 5px 8px; margin: 3px 0; border-radius: 6px;
        background: #f9f9f9; font-size: 12px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .partner-row .dir {{ color: #888; font-size: 13px; margin-right: 6px; flex-shrink: 0; }}
    .partner-row .pname {{ flex: 1; font-weight: bold; white-space: nowrap;
                           overflow: hidden; text-overflow: ellipsis; }}
    .partner-row .count {{ font-size: 11px; color: white; padding: 1px 6px;
                           border-radius: 10px; flex-shrink: 0; margin-left: 6px; }}
    .section-label {{ font-size: 11px; font-weight: bold; color: #666;
                      margin: 10px 0 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
    #graph-stats {{
        position: absolute; top: 8px; left: 10px; font-size: 11px; color: #666;
        background: rgba(255,255,255,0.85); padding: 3px 8px;
        border-radius: 4px; border: 1px solid #eee; pointer-events: none;
    }}
    #graph-hint {{
        position: absolute; bottom: 8px; left: 10px; font-size: 10px; color: #aaa;
        pointer-events: none;
    }}
    .tooltip {{
        position: fixed; background: white; border: 1px solid #ccc; border-radius: 6px;
        padding: 8px 12px; font-size: 12px; pointer-events: none;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15); display: none;
        max-width: 280px; z-index: 1000;
    }}
    </style>
    </head>
    <body>
    <div id="container">
      <div id="graph-wrap">
        <div id="graph-stats">{_n_nodes} nodes &nbsp;·&nbsp; {_n_edges} edges</div>
        <div id="graph-hint">Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Click node to explore &nbsp;·&nbsp; Drag node to reposition</div>
      </div>
      <div id="detail-panel">
        <div id="detail-header">Communications <span id="detail-name" style="color:#555;font-weight:normal"></span></div>
        <div id="detail-body"><div id="detail-empty">Click a node to see its communications</div></div>
      </div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
    try {{

    var nodesData  = {_nodes_json};
    var linksData  = {_edges_json};
    var nodeColors = {_node_colors_json};

    // ── Node radius (scales with total activity) ──────────────────────
    var nodeMap = {{}};
    nodesData.forEach(function(n) {{ nodeMap[n.id] = n; }});

    function nodeR(d) {{
        var total = (d.total_sent || 0) + (d.total_received || 0);
        return Math.max(10, Math.min(28, 10 + total * 0.28));
    }}

    // ── SVG setup ─────────────────────────────────────────────────────
    var wrap   = document.getElementById("graph-wrap");
    var width  = wrap.offsetWidth || 720;
    var height = 740;

    var svg = d3.select("#graph-wrap").append("svg")
        .attr("width", "100%").attr("height", height);

    var zoomG = svg.append("g");

    svg.call(d3.zoom().scaleExtent([0.15, 4]).on("zoom", function(event) {{
        zoomG.attr("transform", event.transform);
    }}));

    // ── Arrow marker ──────────────────────────────────────────────────
    var defs = svg.append("defs");
    defs.append("marker")
        .attr("id", "comm-arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 0).attr("refY", 0)
        .attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#aaa");

    // ── Force simulation ──────────────────────────────────────────────
    var simulation = d3.forceSimulation(nodesData)
        .force("link", d3.forceLink(linksData).id(function(d) {{ return d.id; }}).distance(150))
        .force("charge", d3.forceManyBody().strength(-420))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(function(d) {{ return nodeR(d) + 12; }}));

    // ── Edges ─────────────────────────────────────────────────────────
    var edgeSel = zoomG.append("g").selectAll("path")
        .data(linksData).enter().append("path")
        .attr("fill", "none")
        .attr("stroke", "#bbb")
        .attr("stroke-width", function(d) {{ return Math.min(1.5 + d.weight * 0.35, 10); }})
        .attr("opacity", 0.55)
        .attr("marker-end", "url(#comm-arrow)");

    // ── Nodes ─────────────────────────────────────────────────────────
    var nodeSel = zoomG.append("g").selectAll("g")
        .data(nodesData).enter().append("g")
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", function(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            }})
            .on("drag",  function(event, d) {{ d.fx = event.x; d.fy = event.y; }})
            .on("end",   function(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
            }})
        );

    nodeSel.append("circle")
        .attr("r", nodeR)
        .attr("fill", function(d) {{ return nodeColors[d.sub_type] || "#ccc"; }})
        .attr("stroke", "#fff").attr("stroke-width", 2.5);

    nodeSel.append("text")
        .attr("text-anchor", "middle")
        .style("font-size", "10px").style("pointer-events", "none")
        .style("fill", "#333").style("font-family", "'Segoe UI', sans-serif")
        .each(function(d) {{
            // Position label below the node circle
            d3.select(this).attr("dy", nodeR(d) + 13);
        }})
        .text(function(d) {{ return d.id; }});

    // ── Tooltip ───────────────────────────────────────────────────────
    var tooltip = d3.select("#tooltip");

    nodeSel
        .on("mouseover", function(event, d) {{
            var outMsgs = linksData.filter(function(l) {{ return l.source.id === d.id; }})
                .reduce(function(s, l) {{ return s + l.weight; }}, 0);
            var inMsgs  = linksData.filter(function(l) {{ return l.target.id === d.id; }})
                .reduce(function(s, l) {{ return s + l.weight; }}, 0);
            tooltip.style("display", "block")
                .html(
                    "<strong>" + d.id + "</strong>"
                    + " <span style='color:#888;font-size:10px'>" + d.sub_type + "</span><br/>"
                    + "<span style='color:#555'>→ Sent: <b>" + outMsgs + "</b></span>"
                    + " &nbsp; "
                    + "<span style='color:#555'>← Received: <b>" + inMsgs + "</b></span>"
                )
                .style("left", (event.clientX + 14) + "px")
                .style("top",  (event.clientY - 20) + "px");
        }})
        .on("mousemove", function(event) {{
            tooltip.style("left", (event.clientX + 14) + "px")
                   .style("top",  (event.clientY - 20) + "px");
        }})
        .on("mouseout", function() {{ tooltip.style("display", "none"); }})
        .on("click", function(event, d) {{
            event.stopPropagation();
            if (selectedNode === d.id) {{
                selectedNode = null;
                resetHighlight();
                clearDetail();
            }} else {{
                selectedNode = d.id;
                highlightEgo(d.id);
                showDetail(d);
            }}
        }});

    svg.on("click", function() {{ selectedNode = null; resetHighlight(); clearDetail(); }});

    // ── Ego highlight ─────────────────────────────────────────────────
    var selectedNode = null;

    function highlightEgo(nodeId) {{
        var neighbours = new Set([nodeId]);
        linksData.forEach(function(l) {{
            if (l.source.id === nodeId || l.target.id === nodeId) {{
                neighbours.add(l.source.id);
                neighbours.add(l.target.id);
            }}
        }});
        nodeSel.transition().duration(200)
            .attr("opacity", function(d) {{ return neighbours.has(d.id) ? 1.0 : 0.07; }});
        edgeSel.transition().duration(200)
            .attr("opacity", function(l) {{
                return (l.source.id === nodeId || l.target.id === nodeId) ? 0.85 : 0.03;
            }});
        nodeSel.select("circle")
            .attr("stroke", function(d) {{ return d.id === nodeId ? "#333" : "#fff"; }})
            .attr("stroke-width", function(d) {{ return d.id === nodeId ? 3.5 : 2.5; }});
    }}

    function resetHighlight() {{
        nodeSel.transition().duration(200).attr("opacity", 1);
        edgeSel.transition().duration(200).attr("opacity", 0.55);
        nodeSel.select("circle").attr("stroke", "#fff").attr("stroke-width", 2.5);
    }}

    // ── Detail panel ──────────────────────────────────────────────────
    function showDetail(d) {{
        document.getElementById("detail-name").textContent = "— " + d.id;

        var sent = linksData.filter(function(l) {{ return l.source.id === d.id; }})
            .sort(function(a, b) {{ return b.weight - a.weight; }});
        var recv = linksData.filter(function(l) {{ return l.target.id === d.id; }})
            .sort(function(a, b) {{ return b.weight - a.weight; }});

        var totalSent = sent.reduce(function(s, l) {{ return s + l.weight; }}, 0);
        var totalRecv = recv.reduce(function(s, l) {{ return s + l.weight; }}, 0);
        var nodeColor = nodeColors[d.sub_type] || "#ccc";

        var html =
            "<div style='font-size:11px;color:#888;margin-bottom:8px'>"
            + d.sub_type
            + "</div>"
            + "<div style='display:flex;gap:8px;margin-bottom:10px'>"
            + "<div style='flex:1;text-align:center;padding:6px;background:#f5f5f5;border-radius:5px'>"
            + "<div style='font-size:20px;font-weight:bold;color:" + nodeColor + "'>" + totalSent + "</div>"
            + "<div style='font-size:10px;color:#888'>Sent</div></div>"
            + "<div style='flex:1;text-align:center;padding:6px;background:#f5f5f5;border-radius:5px'>"
            + "<div style='font-size:20px;font-weight:bold;color:" + nodeColor + "'>" + totalRecv + "</div>"
            + "<div style='font-size:10px;color:#888'>Received</div></div>"
            + "</div>";

        if (sent.length > 0) {{
            html += "<div class='section-label'>→ Sent to</div>";
            sent.forEach(function(l) {{
                var partnerColor = nodeColors[(nodeMap[l.target.id] || {{}}).sub_type] || "#ccc";
                html += "<div class='partner-row'>"
                    + "<span class='dir'>→</span>"
                    + "<span class='pname'>" + l.target.id + "</span>"
                    + "<span class='count' style='background:" + partnerColor + "'>" + l.weight + "</span>"
                    + "</div>";
            }});
        }}

        if (recv.length > 0) {{
            html += "<div class='section-label'>← Received from</div>";
            recv.forEach(function(l) {{
                var partnerColor = nodeColors[(nodeMap[l.source.id] || {{}}).sub_type] || "#ccc";
                html += "<div class='partner-row'>"
                    + "<span class='dir'>←</span>"
                    + "<span class='pname'>" + l.source.id + "</span>"
                    + "<span class='count' style='background:" + partnerColor + "'>" + l.weight + "</span>"
                    + "</div>";
            }});
        }}

        document.getElementById("detail-body").innerHTML = html;
    }}

    function clearDetail() {{
        document.getElementById("detail-name").textContent = "";
        document.getElementById("detail-body").innerHTML =
            "<div id='detail-empty'>Click a node to see its communications</div>";
    }}

    // ── Tick ──────────────────────────────────────────────────────────
    simulation.on("tick", function() {{
        edgeSel.attr("d", function(d) {{
            var dx = d.target.x - d.source.x;
            var dy = d.target.y - d.source.y;
            var dist = Math.sqrt(dx * dx + dy * dy) || 1;

            // Start on source circle boundary, end short of target circle (for arrowhead)
            var rs = nodeR(d.source) + 2;
            var rt = nodeR(d.target) + 9;
            var sx = d.source.x + dx / dist * rs;
            var sy = d.source.y + dy / dist * rs;
            var tx = d.target.x - dx / dist * rt;
            var ty = d.target.y - dy / dist * rt;

            // Curve to the right of the direction of travel — this means A→B and B→A
            // get opposite perpendicular offsets and are always visually separated
            var curve = 22;
            var mx = (sx + tx) / 2 - dy / dist * curve;
            var my = (sy + ty) / 2 + dx / dist * curve;

            return "M" + sx + "," + sy + " Q" + mx + "," + my + " " + tx + "," + ty;
        }});
        nodeSel.attr("transform", function(d) {{
            return "translate(" + d.x + "," + d.y + ")";
        }});
    }});

    // ── Legend ────────────────────────────────────────────────────────
    var legSvg = d3.select("#graph-wrap").append("svg")
        .attr("width", "100%").attr("height", 30)
        .style("position", "absolute").style("bottom", "28px").style("left", "0");

    var nLeg = legSvg.append("g").attr("transform", "translate(10, 8)");
    nLeg.append("text").attr("y", 9).style("font-size", "10px")
        .style("font-weight", "bold").style("fill", "#555").text("Entity:");
    Object.entries(nodeColors).forEach(function(e, i) {{
        nLeg.append("circle").attr("cx", 56 + i * 100).attr("cy", 5).attr("r", 6).attr("fill", e[1]);
        nLeg.append("text").attr("x", 65 + i * 100).attr("y", 9)
            .style("font-size", "10px").style("fill", "#444").text(e[0]);
    }});
    // Edge thickness hint
    nLeg.append("text").attr("x", 56 + Object.keys(nodeColors).length * 100).attr("y", 9)
        .style("font-size", "10px").style("fill", "#888")
        .text("Edge thickness = message count");

    }} catch(e) {{
        document.getElementById("graph-wrap").innerHTML =
            "<pre style='color:red;padding:12px'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height="800px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **3. Communication Frequency Matrix (Heatmap)**

    This heatmap provides a detailed overview of the ***communication intensity*** between entities. The darker the cell color, the more frequent the communication between that sender-receiver pair.

    The matrix is organized with **[P] = Person, [V] = Vessel, [O] = Organization** prefixes for easy identification.
    """)
    return


@app.cell
def _(mo):
    heatmap_type_filter = mo.ui.multiselect(
        options=['Person', 'Vessel', 'Organization'],
        value=['Person', 'Vessel'],
        label="Include Entity Types in Heatmap:"
    )
    heatmap_type_filter
    return (heatmap_type_filter,)


@app.cell
def _(all_entities, comm_matrix, go, heatmap_type_filter, np):
    def build_heatmap_data(entity_types):
        filtered = [eid for eid, e in all_entities.items()
                   if e.get('sub_type') in entity_types]

        filtered = sorted(filtered, key=lambda x: (all_entities[x].get('sub_type', ''), x))

        n = len(filtered)
        matrix = np.zeros((n, n))

        for i, sender in enumerate(filtered):
            for j, receiver in enumerate(filtered):
                if sender in comm_matrix and receiver in comm_matrix[sender]:
                    matrix[i][j] = len(comm_matrix[sender][receiver])

        return filtered, matrix

    entities_hm, matrix_hm = build_heatmap_data(heatmap_type_filter.value)

    _labels_hm = []
    for _eid in entities_hm:
        _etype = all_entities[_eid].get('sub_type', '?')[0]
        _labels_hm.append(f"[{_etype}] {_eid}")

    _fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_hm,
        x=_labels_hm,
        y=_labels_hm,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Messages: %{z}<extra></extra>'
    ))

    _fig_heatmap.update_layout(
        title=dict(
            text='<b>Communication Frequency Matrix</b><br><sup>Row = Sender, Column = Receiver. [P]=Person, [V]=Vessel, [O]=Organization</sup>',
            x=0.5
        ),
        xaxis=dict(title='Receiver', tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(title='Sender', tickfont=dict(size=8)),
        height=800,
        width=900,
        margin=dict(l=150, r=50, t=100, b=150)
    )

    _fig_heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **4. Formal Relationship Network**

    Beyond communications, entities in Oceanus also have ***formal relationships*** such as Colleagues, Operates, Reports, Coordinates, and Suspicious. This network visualization displays those structural connections.

    - **Green** for Colleagues
    - **Blue** for Operates
    - **Purple** for Reports
    - **Orange** for Coordinates
    - **Red** for Suspicious
    """)
    return


@app.cell
def _(mo):
    rel_type_filter = mo.ui.multiselect(
        options=['Colleagues', 'Operates', 'Reports', 'Coordinates', 'Suspicious', 'Friends', 'Unfriendly'],
        value=['Colleagues', 'Operates', 'Reports', 'Suspicious'],
        label="Show Relationship Types:"
    )
    rel_type_filter
    return (rel_type_filter,)


@app.cell
def _(all_entities, json_lib, mo, rel_type_filter, relationship_data):
    _edge_color_map = {
        'Colleagues':       '#2ECC71',
        'Operates':         '#3498DB',
        'Reports':          '#9B59B6',
        'Coordinates':      '#F39C12',
        'Suspicious':       '#E74C3C',
        'Friends':          '#1ABC9C',
        'Unfriendly':       '#C0392B',
        'AccessPermission': '#7F8C8D',
        'Jurisdiction':     '#BDC3C7',
    }
    _node_color_map = {
        'Person':       '#4ECDC4',
        'Vessel':       '#FF6B6B',
        'Organization': '#95E1D3',
        'Group':        '#F38181',
    }

    # Filter to selected relationship types and entities present in all_entities
    _filtered_rels = [
        r for r in relationship_data
        if r['type'] in rel_type_filter.value
        and r['entity1'] in all_entities
        and r['entity2'] in all_entities
    ]

    # Build node list from entities that appear in at least one filtered edge
    _node_ids = set()
    for _r in _filtered_rels:
        _node_ids.add(_r['entity1'])
        _node_ids.add(_r['entity2'])

    _nodes_data = [
        {"id": nid, "sub_type": all_entities[nid].get("sub_type", "Unknown")}
        for nid in _node_ids
    ]
    _edges_data = [
        {
            "source": r['entity1'],
            "target": r['entity2'],
            "type":   r['type'],
            "bidirectional": r['bidirectional'],
        }
        for r in _filtered_rels
    ]

    _nodes_json      = json_lib.dumps(_nodes_data)
    _edges_json      = json_lib.dumps(_edges_data)
    _node_colors_json = json_lib.dumps(_node_color_map)
    _edge_colors_json = json_lib.dumps(_edge_color_map)
    _n_nodes = len(_nodes_data)
    _n_edges = len(_edges_data)

    mo.iframe(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #fafafa; }}
    #container {{ display: flex; width: 100%; height: 740px; }}
    #graph-wrap {{ flex: 1; height: 100%; background: white; overflow: hidden; position: relative; }}
    #detail-panel {{
        width: 280px; height: 100%; background: white;
        border-left: 2px solid #ddd; display: flex; flex-direction: column;
    }}
    #detail-header {{
        padding: 8px 12px; background: #f0f0f0; border-bottom: 1px solid #ddd;
        font-weight: bold; font-size: 13px; flex-shrink: 0;
    }}
    #detail-body {{ flex: 1; overflow-y: auto; padding: 10px 12px; }}
    #detail-empty {{ color: #aaa; font-size: 13px; text-align: center; margin-top: 60px; }}
    .rel-row {{
        padding: 6px 8px; margin: 4px 0; border-radius: 6px;
        background: #f9f9f9; font-size: 12px; border-left: 4px solid #ccc;
    }}
    .rel-row .other {{ font-weight: bold; }}
    .rel-row .rtype {{ font-size: 10px; color: white; padding: 1px 5px;
                       border-radius: 3px; display: inline-block; margin-left: 4px; }}
    .rel-row .dir {{ color: #888; font-size: 11px; margin: 0 4px; }}
    #graph-stats {{
        position: absolute; top: 8px; left: 10px; font-size: 11px; color: #666;
        background: rgba(255,255,255,0.85); padding: 3px 8px;
        border-radius: 4px; border: 1px solid #eee; pointer-events: none;
    }}
    #graph-hint {{
        position: absolute; bottom: 8px; left: 10px; font-size: 10px; color: #aaa;
        pointer-events: none;
    }}
    .tooltip {{
        position: fixed; background: white; border: 1px solid #ccc; border-radius: 6px;
        padding: 8px 12px; font-size: 12px; pointer-events: none;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15); display: none;
        max-width: 300px; z-index: 1000;
    }}
    </style>
    </head>
    <body>
    <div id="container">
      <div id="graph-wrap">
        <div id="graph-stats">{_n_nodes} nodes &nbsp;·&nbsp; {_n_edges} edges</div>
        <div id="graph-hint">Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Click node to explore &nbsp;·&nbsp; Drag node to reposition</div>
      </div>
      <div id="detail-panel">
        <div id="detail-header">Relationships <span id="detail-name" style="color:#555;font-weight:normal"></span></div>
        <div id="detail-body"><div id="detail-empty">Click a node to see its relationships</div></div>
      </div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
    try {{

    var nodesData   = {_nodes_json};
    var linksData   = {_edges_json};
    var nodeColors  = {_node_colors_json};
    var edgeColors  = {_edge_colors_json};

    // ── pair offset bookkeeping for parallel edges ──────────────────────
    var pairCount = {{}};
    linksData.forEach(function(l) {{
        var key = [l.source, l.target].sort().join("|||");
        pairCount[key] = (pairCount[key] || 0) + 1;
    }});
    var pairCur = {{}};
    linksData.forEach(function(l) {{
        var key = [l.source, l.target].sort().join("|||");
        l._pairIndex = pairCur[key] || 0;
        l._pairCount = pairCount[key];
        pairCur[key] = (pairCur[key] || 0) + 1;
    }});

    // ── SVG setup ────────────────────────────────────────────────────────
    var wrap   = document.getElementById("graph-wrap");
    var width  = wrap.offsetWidth || 720;
    var height = 740;

    var svg = d3.select("#graph-wrap").append("svg")
        .attr("width", "100%").attr("height", height);

    var zoomG = svg.append("g");

    svg.call(d3.zoom().scaleExtent([0.2, 4]).on("zoom", function(event) {{
        zoomG.attr("transform", event.transform);
    }}));

    // ── Arrow markers (one per relationship type) ─────────────────────
    var defs = svg.append("defs");
    Object.entries(edgeColors).forEach(function(e) {{
        var rtype = e[0], color = e[1];
        var mid = rtype.replace(/\s/g, "-");
        defs.append("marker")
            .attr("id", "arr-" + mid)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 24).attr("refY", 0)
            .attr("markerWidth", 5).attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", color);
    }});

    // ── Force simulation ──────────────────────────────────────────────
    var simulation = d3.forceSimulation(nodesData)
        .force("link", d3.forceLink(linksData).id(function(d) {{ return d.id; }}).distance(130))
        .force("charge", d3.forceManyBody().strength(-380))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(32));

    // ── Edges ─────────────────────────────────────────────────────────
    var edgeSel = zoomG.append("g").selectAll("path")
        .data(linksData).enter().append("path")
        .attr("fill", "none")
        .attr("stroke", function(d) {{ return edgeColors[d.type] || "#bbb"; }})
        .attr("stroke-width", 2)
        .attr("opacity", 0.65)
        .attr("marker-end", function(d) {{
            return d.bidirectional ? null
                : "url(#arr-" + d.type.replace(/\s/g, "-") + ")";
        }});

    // ── Nodes ─────────────────────────────────────────────────────────
    var nodeSel = zoomG.append("g").selectAll("g")
        .data(nodesData).enter().append("g")
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", function(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            }})
            .on("drag", function(event, d) {{ d.fx = event.x; d.fy = event.y; }})
            .on("end",  function(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
            }})
        );

    nodeSel.append("circle")
        .attr("r", 14)
        .attr("fill", function(d) {{ return nodeColors[d.sub_type] || "#ccc"; }})
        .attr("stroke", "#fff").attr("stroke-width", 2.5);

    nodeSel.append("text")
        .attr("dy", 27).attr("text-anchor", "middle")
        .style("font-size", "10px").style("pointer-events", "none")
        .style("fill", "#333").style("font-family", "'Segoe UI', sans-serif")
        .text(function(d) {{ return d.id; }});

    // ── Tooltip ───────────────────────────────────────────────────────
    var tooltip = d3.select("#tooltip");

    nodeSel
        .on("mouseover", function(event, d) {{
            if (selectedNode !== null && selectedNode !== d.id) return;
            var conn = linksData.filter(function(l) {{
                return l.source.id === d.id || l.target.id === d.id;
            }});
            var lines = conn.map(function(l) {{
                var other = l.source.id === d.id ? l.target.id : l.source.id;
                var dir   = l.bidirectional ? "↔" : (l.source.id === d.id ? "→" : "←");
                var col   = edgeColors[l.type] || "#999";
                return "<span style='color:" + col + "'>■</span> <b>" + l.type + "</b> " + dir + " " + other;
            }}).join("<br/>");
            tooltip.style("display", "block")
                .html("<strong>" + d.id + "</strong> <span style='color:#888;font-size:10px'>" + d.sub_type + "</span><br/><br/>" + lines)
                .style("left", (event.clientX + 14) + "px")
                .style("top",  (event.clientY - 20) + "px");
        }})
        .on("mousemove", function(event) {{
            tooltip.style("left", (event.clientX + 14) + "px")
                   .style("top",  (event.clientY - 20) + "px");
        }})
        .on("mouseout", function() {{ tooltip.style("display", "none"); }})
        .on("click", function(event, d) {{
            event.stopPropagation();
            if (selectedNode === d.id) {{
                selectedNode = null;
                resetHighlight();
                clearDetail();
            }} else {{
                selectedNode = d.id;
                highlightEgo(d.id);
                showDetail(d);
            }}
        }});

    svg.on("click", function() {{ selectedNode = null; resetHighlight(); clearDetail(); }});

    // ── Ego highlight ─────────────────────────────────────────────────
    var selectedNode = null;

    function highlightEgo(nodeId) {{
        var neighbours = new Set([nodeId]);
        linksData.forEach(function(l) {{
            if (l.source.id === nodeId || l.target.id === nodeId) {{
                neighbours.add(l.source.id);
                neighbours.add(l.target.id);
            }}
        }});
        nodeSel.transition().duration(200)
            .attr("opacity", function(d) {{ return neighbours.has(d.id) ? 1.0 : 0.08; }});
        edgeSel.transition().duration(200)
            .attr("opacity", function(l) {{
                return (l.source.id === nodeId || l.target.id === nodeId) ? 0.85 : 0.04;
            }});
        // Bold ring on selected node
        nodeSel.select("circle")
            .attr("stroke", function(d) {{ return d.id === nodeId ? "#333" : "#fff"; }})
            .attr("stroke-width", function(d) {{ return d.id === nodeId ? 3.5 : 2.5; }});
    }}

    function resetHighlight() {{
        nodeSel.transition().duration(200).attr("opacity", 1);
        edgeSel.transition().duration(200).attr("opacity", 0.65);
        nodeSel.select("circle").attr("stroke", "#fff").attr("stroke-width", 2.5);
    }}

    // ── Detail panel ──────────────────────────────────────────────────
    function showDetail(d) {{
        document.getElementById("detail-name").textContent = "— " + d.id;
        var conn = linksData.filter(function(l) {{
            return l.source.id === d.id || l.target.id === d.id;
        }});
        // sort: suspicious first, then alphabetical by type
        conn.sort(function(a, b) {{
            if (a.type === "Suspicious" && b.type !== "Suspicious") return -1;
            if (b.type === "Suspicious" && a.type !== "Suspicious") return  1;
            return a.type.localeCompare(b.type);
        }});
        var html = "<div style='font-size:11px;color:#888;margin-bottom:6px'>"
            + d.sub_type + " &nbsp;·&nbsp; " + conn.length + " relationship" + (conn.length !== 1 ? "s" : "")
            + "</div>";
        conn.forEach(function(l) {{
            var other = l.source.id === d.id ? l.target.id : l.source.id;
            var otherType = "";
            nodesData.forEach(function(n) {{ if (n.id === other) otherType = n.sub_type; }});
            var dir   = l.bidirectional ? "↔" : (l.source.id === d.id ? "→" : "←");
            var col   = edgeColors[l.type] || "#999";
            html += "<div class='rel-row' style='border-left-color:" + col + "'>"
                + "<span class='rtype' style='background:" + col + "'>" + l.type + "</span>"
                + " <span class='dir'>" + dir + "</span>"
                + "<span class='other'>" + other + "</span>"
                + " <span style='color:#aaa;font-size:10px'>(" + otherType + ")</span>"
                + "</div>";
        }});
        var body = document.getElementById("detail-body");
        body.innerHTML = html;
    }}

    function clearDetail() {{
        document.getElementById("detail-name").textContent = "";
        document.getElementById("detail-body").innerHTML =
            "<div id='detail-empty'>Click a node to see its relationships</div>";
    }}

    // ── Tick ──────────────────────────────────────────────────────────
    simulation.on("tick", function() {{
        edgeSel.attr("d", function(d) {{
            var dx = d.target.x - d.source.x;
            var dy = d.target.y - d.source.y;
            var dist = Math.sqrt(dx * dx + dy * dy) || 1;
            if (d._pairCount > 1) {{
                var offset = (d._pairIndex - (d._pairCount - 1) / 2) * 14;
                var mx = (d.source.x + d.target.x) / 2 + (-dy / dist) * offset;
                var my = (d.source.y + d.target.y) / 2 + (dx  / dist) * offset;
                return "M" + d.source.x + "," + d.source.y
                     + " Q" + mx + "," + my
                     + " " + d.target.x + "," + d.target.y;
            }}
            return "M" + d.source.x + "," + d.source.y
                 + " L" + d.target.x + "," + d.target.y;
        }});
        nodeSel.attr("transform", function(d) {{
            return "translate(" + d.x + "," + d.y + ")";
        }});
    }});

    // ── Legends ───────────────────────────────────────────────────────
    var legSvg = d3.select("#graph-wrap").append("svg")
        .attr("width", "100%").attr("height", 56)
        .style("position", "absolute").style("bottom", "28px").style("left", "0");

    // Node type legend
    var nodeTypes = Object.entries(nodeColors);
    var nLeg = legSvg.append("g").attr("transform", "translate(10, 10)");
    nLeg.append("text").attr("y", 9).style("font-size", "10px")
        .style("font-weight", "bold").style("fill", "#555").text("Entity:");
    nodeTypes.forEach(function(e, i) {{
        nLeg.append("circle").attr("cx", 56 + i * 90).attr("cy", 5).attr("r", 6).attr("fill", e[1]);
        nLeg.append("text").attr("x", 65 + i * 90).attr("y", 9)
            .style("font-size", "10px").style("fill", "#444").text(e[0]);
    }});

    // Edge type legend
    var edgeTypes = Object.entries(edgeColors);
    var eLeg = legSvg.append("g").attr("transform", "translate(10, 32)");
    eLeg.append("text").attr("y", 9).style("font-size", "10px")
        .style("font-weight", "bold").style("fill", "#555").text("Relation:");
    edgeTypes.forEach(function(e, i) {{
        eLeg.append("rect").attr("x", 58 + i * 100).attr("y", 0)
            .attr("width", 14).attr("height", 4).attr("rx", 2).attr("fill", e[1]);
        eLeg.append("text").attr("x", 75 + i * 100).attr("y", 9)
            .style("font-size", "10px").style("fill", "#444").text(e[0]);
    }});

    }} catch(e) {{
        document.getElementById("graph-wrap").innerHTML =
            "<pre style='color:red;padding:12px'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height="800px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **5. Individual Communication Profiles (Entity Deep Dive)**

    By selecting an entity from the dropdown, we can examine:
    - **Messages Sent** - Who does the entity send messages to and how many?
    - **Messages Received** - Who sends messages to the entity and how many?
    - **Formal Relationships** - What structural relationships does this entity have?
    """)
    return


@app.cell
def _(all_entities, mo):
    entity_selector = mo.ui.dropdown(
        options=sorted(all_entities.keys()),
        value='Nadia Conti',
        label="Select Entity to Analyze:"
    )
    entity_selector
    return (entity_selector,)


@app.cell
def _(
    all_entities_full,
    comm_matrix,
    entity_selector,
    go,
    make_subplots,
    pd,
    relationship_data,
):
    selected_entity = entity_selector.value

    _sent_to = {}
    _received_from = {}

    if selected_entity in comm_matrix:
        for _receiver, _comms in comm_matrix[selected_entity].items():
            _sent_to[_receiver] = len(_comms)

    for _sender in comm_matrix:
        if selected_entity in comm_matrix[_sender]:
            _received_from[_sender] = len(comm_matrix[_sender][selected_entity])

    fig_entity_profile = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Messages Sent by {selected_entity}',
            f'Messages Received by {selected_entity}'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    _sent_data = sorted(_sent_to.items(), key=lambda x: -x[1])[:15]
    if _sent_data:
        _recipients, _counts = zip(*_sent_data)

        def get_color(entity_id):
            sub_type = all_entities_full.get(entity_id, {}).get('sub_type', 'Unknown')
            color_map = {'Vessel': '#FF6B6B', 'Person': '#4ECDC4', 'Location': '#2ECC71',
                        'Organization': '#95E1D3', 'Group': '#F38181'}
            return color_map.get(sub_type, '#CCCCCC')

        _colors_sent = [get_color(_r) for _r in _recipients]
        fig_entity_profile.add_trace(
            go.Bar(x=list(_recipients), y=list(_counts), marker_color=_colors_sent, name='Sent', showlegend=False),
            row=1, col=1
        )

    _recv_data = sorted(_received_from.items(), key=lambda x: -x[1])[:15]
    if _recv_data:
        _senders, _counts_r = zip(*_recv_data)
        _colors_recv = [get_color(_s) for _s in _senders]
        fig_entity_profile.add_trace(
            go.Bar(x=list(_senders), y=list(_counts_r), marker_color=_colors_recv, name='Received', showlegend=False),
            row=1, col=2
        )

    fig_entity_profile.update_layout(
        title=dict(
            text=f'<b>Communication Profile: {selected_entity}</b><br><sup>Type: {all_entities_full.get(selected_entity, {}).get("sub_type", "Unknown")} | Teal=Person, Red=Vessel, Green=Location, Mint=Org</sup>',
            x=0.5
        ),
        showlegend=False,
        height=450
    )

    fig_entity_profile.update_xaxes(tickangle=45)

    _entity_rels = [_r for _r in relationship_data
                  if _r['entity1'] == selected_entity or _r['entity2'] == selected_entity]

    _rel_summary_data = []
    for _r in _entity_rels:
        _other = _r['entity2'] if _r['entity1'] == selected_entity else _r['entity1']
        _direction = '↔' if _r['bidirectional'] else ('→' if _r['entity1'] == selected_entity else '←')
        _rel_summary_data.append({
            'Relationship': _r['type'],
            'Direction': _direction,
            'Other Entity': _other,
            'Other Type': all_entities_full.get(_other, {}).get('sub_type', 'Unknown')
        })

    rel_df = pd.DataFrame(_rel_summary_data) if _rel_summary_data else pd.DataFrame(columns=['Relationship', 'Direction', 'Other Entity', 'Other Type'])

    fig_entity_profile
    return rel_df, selected_entity


@app.cell
def _(mo, selected_entity):
    mo.md(f"""
    ### Formal Relationships of {selected_entity}

    The table below shows all formal relationships that **{selected_entity}** has with other entities. The direction indicates:
    - **↔** for bidirectional relationships (Colleagues, Friends)
    - **→** for outgoing relationships (the entity is the source)
    - **←** for incoming relationships (the entity is the target)
    """)
    return


@app.cell
def _(mo, rel_df):
    if len(rel_df) > 0:
        mo.ui.table(rel_df)
    else:
        mo.md("*No formal relationships found for this entity.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **6. Communication Timeline**

    Visualizing the **temporal distribution** of communications over the two-week observation period (October 1-14, 2040).
    """)
    return


@app.cell
def _(comm_events, go, pd):
    _timeline_data = []
    for _comm in comm_events:
        _ts = _comm.get('timestamp', '')
        if _ts:
            _timeline_data.append({
                'timestamp': pd.to_datetime(_ts),
                'comm_id': _comm['id']
            })

    timeline_df = pd.DataFrame(_timeline_data)
    timeline_df['date'] = timeline_df['timestamp'].dt.date
    timeline_df['hour'] = timeline_df['timestamp'].dt.hour
    timeline_df['day_name'] = timeline_df['timestamp'].dt.day_name()

    daily_counts = timeline_df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])

    fig_timeline = go.Figure()

    fig_timeline.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['count'],
        marker_color='#3498DB',
        name='Daily Messages',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Messages: %{y}<extra></extra>'
    ))

    fig_timeline.update_layout(
        title=dict(
            text='<b>Communication Volume Over Time</b><br><sup>Daily message counts across the two-week observation period (Oct 1-14, 2040)</sup>',
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='%b %d', tickangle=45),
        plot_bgcolor='#fafafa'
    )

    fig_timeline
    return (timeline_df,)


@app.cell
def _(go, timeline_df):
    _hourly_daily = timeline_df.groupby(['day_name', 'hour']).size().reset_index(name='count')

    _day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    _pivot_data = _hourly_daily.pivot(index='day_name', columns='hour', values='count').fillna(0)
    _pivot_data = _pivot_data.reindex(columns=range(24), fill_value=0)

    fig_hourly_heatmap = go.Figure(data=go.Heatmap(
        z=_pivot_data.values,
        x=[f'{h:02d}:00' for h in range(24)],
        y=_day_order,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Messages: %{z}<extra></extra>'
    ))

    fig_hourly_heatmap.update_layout(
        title=dict(
            text='<b>Communication Activity by Hour and Day of Week</b><br><sup>Darker cells indicate more communication activity</sup>',
            x=0.5
        ),
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10))
    )

    fig_hourly_heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **7. Key Statistics**
    """)
    return


@app.cell
def _(all_entities, comm_events, mo, relationship_data):
    _total_entities = len(all_entities)
    _total_comms = len(comm_events)
    _total_rels = len(relationship_data)

    mo.hstack([
        mo.stat(value=_total_entities, label="Total Entities", bordered=True),
        mo.stat(value=_total_comms, label="Communications", bordered=True),
        mo.stat(value=_total_rels, label="Formal Relationships", bordered=True),
    ], justify='center', gap=2)
    return


@app.cell
def _(all_entities_full, comm_matrix, mo, pd):
    _sent_counts = {}
    _recv_counts = {}

    for _sender_k in comm_matrix:
        for _receiver_k, _comms_k in comm_matrix[_sender_k].items():
            _sent_counts[_sender_k] = _sent_counts.get(_sender_k, 0) + len(_comms_k)
            _recv_counts[_receiver_k] = _recv_counts.get(_receiver_k, 0) + len(_comms_k)

    _total_activity = {_k: _sent_counts.get(_k, 0) + _recv_counts.get(_k, 0)
                      for _k in set(_sent_counts) | set(_recv_counts)}

    _top_active = sorted(_total_activity.items(), key=lambda x: -x[1])[:15]

    top_active_df = pd.DataFrame(_top_active, columns=['Entity', 'Total Messages'])
    top_active_df['Type'] = top_active_df['Entity'].apply(lambda x: all_entities_full.get(x, {}).get('sub_type', 'Unknown'))
    top_active_df['Sent'] = top_active_df['Entity'].apply(lambda x: _sent_counts.get(x, 0))
    top_active_df['Received'] = top_active_df['Entity'].apply(lambda x: _recv_counts.get(x, 0))
    top_active_df = top_active_df[['Entity', 'Type', 'Sent', 'Received', 'Total Messages']]

    mo.ui.table(top_active_df)
    return


@app.cell
def _(all_entities, go):
    _type_counts = {}
    for _eid, _entity in all_entities.items():
        _etype = _entity.get('sub_type', 'Unknown')
        _type_counts[_etype] = _type_counts.get(_etype, 0) + 1

    fig_entity_distribution = go.Figure(data=[go.Pie(
        labels=list(_type_counts.keys()),
        values=list(_type_counts.values()),
        hole=0.4,
        marker_colors=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig_entity_distribution.update_layout(
        title=dict(
            text='<b>Entity Type Distribution</b><br><sup>Breakdown of actors in the knowledge graph</sup>',
            x=0.5
        ),
        height=400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )

    fig_entity_distribution
    return


@app.cell
def _(go, relationship_data):
    _rel_type_counts = {}
    for _rel in relationship_data:
        _rtype = _rel['type']
        _rel_type_counts[_rtype] = _rel_type_counts.get(_rtype, 0) + 1

    _sorted_rel_types = sorted(_rel_type_counts.items(), key=lambda x: -x[1])

    fig_rel_distribution = go.Figure(data=[go.Bar(
        x=[_v for _k, _v in _sorted_rel_types],
        y=[_k for _k, _v in _sorted_rel_types],
        orientation='h',
        marker_color=['#F39C12', '#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#C0392B', '#1ABC9C'][:len(_sorted_rel_types)],
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )])

    fig_rel_distribution.update_layout(
        title=dict(
            text='<b>Relationship Type Distribution</b><br><sup>Count of each relationship type in the knowledge graph</sup>',
            x=0.5
        ),
        xaxis_title='Count',
        yaxis_title='Relationship Type',
        height=400,
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='#fafafa'
    )

    fig_rel_distribution
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Findings for Question 2.1**

    Based on the visual analytics executed above, several key insights emerge about the interactions and relationships between vessels and people in Oceanus. The following findings are derived directly from the data and verified against the raw knowledge graph.

    The knowledge graph contains 43 entities (18 persons, 15 vessels, 5 organizations, 5 groups), of which **37 actively communicate** across the 584 intercepted messages. Six entities — Sailor Shift, Tourists, Recreational Fishing Boats, Diving Tour Operators, Conservation Vessels, and Mariner's Dream appear in the graph but send or receive no messages, suggesting they are referenced but not direct radio participants. Among the 37 active entities, the top communicators (by total messages) are: **Mako** (80 total: 29 sent, 51 received), **Green Guardians** (70: 43 sent, 27 received), **Oceanus City Council** (60: 26 sent, 34 received), **Reef Guardian** (57), and **Neptune** (54). Mako's strongly receive-heavy ratio (51 received vs 29 sent) is consistent with it functioning as an operational vessel receiving instructions from multiple coordinators rather than initiating activity.

    The Communication Network graph reveals a clear hub-and-spoke topology. The majority of nodes connect only through one or two links, meaning most actors communicate via intermediaries rather than directly. This structure immediately narrows Clepper's focus to the most connected nodes, they are the gatekeepers of information in Oceanus. The Frequency Heatmap (sender = row, receiver = column) reinforces this: several person-to-vessel pairs are strongly asymmetric, with one side consistently sending while the other receives a pattern consistent with operator-to-vessel or supervisor-to-subordinate command dynamics. Truly bilateral dark cells on both sides of the diagonal represent closer, more collaborative relationships.

    The Formal Relationship Network reveals the structural ties underlying the communication data. The 259 extracted relationship edges span nine types: Coordinates (62), AccessPermission (63), Operates (35), Colleagues (29), Suspicious (26), Reports (25), Jurisdiction (12), Unfriendly (5), Friends (2). Note: these counts represent expanded entity-pair edges derived from 259 relationship nodes in the graph, one node connecting multiple entities produces multiple edges. The 26 Suspicious-labelled edges are the most directly investigative relevant: entities appearing in both the suspicious relationship network and the top communication hubs are the highest-priority targets for Clepper's investigation.

    The individual profile tool (demonstrated with Nadia Conti as default) allows Clepper to quickly characterize any actor: messages sent vs. received, specific communication partners, and formal relationships — all in one view. The directional indicators (↔, →, ←) clarify whether an entity is a peer, superior, or subordinate in each connection. The daily bar chart and hour-by-day-of-week heatmap together show that activity peaks between 08:00–14:00 and is unevenly distributed across the Oct 1–14, 2040 window — recurring dark time slots indicate deliberately scheduled coordination, a key behavioural signature that distinguishes organized operations from casual interaction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2.2 / 3: Community Detection & Topic Modeling

    ### Community Detection

    Using a bipartite projection of Entity ↔ Event edges to find clusters of entities that share the most events in common.
    """)
    return


@app.cell
def _(G):
    _keep_subtypes = {"Person", "Organization", "Group", "Vessel"}

    entity_nodes = {
        n for n, a in G.nodes(data=True)
        if a.get("type") == "Entity" and a.get("sub_type") in _keep_subtypes
    }
    event_nodes = {n for n, a in G.nodes(data=True) if a.get("type") == "Event"}
    return entity_nodes, event_nodes


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
    for x, c in enumerate(communities, 1):
        print(x, len(c), sorted(list(c)))
    return


@app.cell
def _(E):
    top_edges = sorted(E.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True)[:10]
    top_edges
    return


@app.cell
def _(communities, mo):
    community_list = [sorted(list(c)) for c in communities]
    community_sizes = [len(c) for c in community_list]

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
    _idx = community_labels.index(community_dd.value)
    _members = community_list[_idx]

    _rows = []
    for _n in _members:
        _a = G.nodes[_n]
        _rows.append({
            "node": _n,
            "sub_type": _a.get("sub_type"),
            "label": _a.get("label"),
            "type": _a.get("type"),
        })

    _df_members = pd.DataFrame(_rows).sort_values(["sub_type", "node"])
    mo.ui.table(_df_members)
    return


@app.cell
def _(E, G, alt, community_dd, community_labels, community_list, nx, pd):
    def _():
        idx = community_labels.index(community_dd.value)
        members = community_list[idx]

        H = E.subgraph(members).copy()

        pos = nx.spring_layout(H, seed=42)

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
                color="sub_type:N",
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
    from sklearn.feature_extraction.text import TfidfVectorizer

    def extract_topics_tfidf(texts, num_topics=15):
        """Extract topics using TF-IDF keywords"""
        if len(texts) < 2:
            return [["insufficient", "data"]], [[1.0] for _ in texts]

        try:
            tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                max_df=0.8,
            )
            tfidf_matrix = tfidf.fit_transform(texts)

            doc_scores = tfidf_matrix.toarray()

            feature_names = tfidf.get_feature_names_out()
            total_scores = np.sum(doc_scores, axis=0)
            top_indices = total_scores.argsort()[-num_topics:][::-1]

            topics = [[feature_names[i]] for i in top_indices]

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

            topic_keywords = {}
            unique_topics = set(topics)
            if -1 in unique_topics:
                unique_topics.remove(-1)

            for topic_id in unique_topics:
                try:
                    keywords = topic_model.get_topic(topic_id)
                    topic_keywords[topic_id] = [word for word, _ in keywords[:top_n]]
                except:
                    continue

            if not topic_keywords:
                return extract_topics_tfidf(texts) + (None,)

            doc_topics_list = []
            max_topic_id = max(topic_keywords.keys()) if topic_keywords else 0

            for i, doc_topic in enumerate(topics):
                weights = [0.0] * (max_topic_id + 1)
                if doc_topic in topic_keywords:
                    weights[doc_topic] = 1.0
                doc_topics_list.append(weights)

            topics_list = []
            for i in range(max_topic_id + 1):
                if i in topic_keywords:
                    topics_list.append(topic_keywords[i])
                else:
                    topics_list.append([])

            return topics_list, doc_topics_list, topic_model
        except Exception as e:
            print(f"BERTopic failed: {str(e)}")
            topics, doc_topics_list = extract_topics_tfidf(texts)
            return topics, doc_topics_list, None

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
def _(topic_model):
    topic_model.visualize_hierarchy()
    return


@app.cell
def _(messages_df, topic_model):
    import plotly.express as _px
    import umap
    from sentence_transformers import SentenceTransformer

    docs = messages_df["content"].tolist()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    reduced = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)

    topics_umap, probs_umap = topic_model.transform(docs)

    topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
    topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    plot_df = messages_df.copy()
    plot_df["x"] = reduced[:, 0]
    plot_df["y"] = reduced[:, 1]
    plot_df["topic"] = topics_umap
    plot_df["topic_name"] = plot_df["topic"].map(topic_name_map)

    def truncate_text(text, n=100):
        text = str(text)
        return text if len(text) <= n else text[:n] + "..."

    plot_df["content_short"] = plot_df["content"].apply(truncate_text)

    fig_umap = _px.scatter(
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

    fig_umap.update_traces(
        customdata=plot_df[["source", "target", "content_short", "topic_name"]].to_numpy(),
        hovertemplate=(
        "<b>Topic:</b> %{customdata[3]}<br>"
        "<b>From:</b> %{customdata[0]}<br>"
        "<b>To:</b> %{customdata[1]}<br><br>"
        "<b>Message preview:</b><br>%{customdata[2]}<br><br>"
        )
    )

    fig_umap.update_layout(
        title="Topic Clusters",
        legend_title="Clusters",
    )

    fig_umap
    return (plot_df,)


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


@app.cell
def _():
    from itertools import combinations
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist, squareform
    import plotly.figure_factory as ff

    return combinations, leaves_list, linkage, pdist


@app.cell
def _(defaultdict, edges_from, edges_to, entity_ids, graph_data):
    _comm_events_q3 = [_n for _n in graph_data['nodes'] if _n.get('sub_type') == 'Communication']
    q3_comm_matrix = defaultdict(lambda: defaultdict(int))
    comm_records = []

    for _comm in _comm_events_q3:
        _comm_id = _comm['id']
        _timestamp = _comm.get('timestamp', '')
        _content = _comm.get('content', '')
        _senders = [_e['source'] for _e in edges_to[_comm_id] if _e.get('type') == 'sent']
        _receivers = [_e['target'] for _e in edges_from[_comm_id] if _e.get('type') == 'received']

        for _sender in _senders:
            for _receiver in _receivers:
                if _sender in entity_ids and _receiver in entity_ids:
                    q3_comm_matrix[_sender][_receiver] += 1
                    comm_records.append({
                        'sender': _sender, 'receiver': _receiver,
                        'timestamp': _timestamp, 'comm_id': _comm_id,
                        'content': _content
                    })

    print(f"Q3: Extracted {len(comm_records)} communication records")
    return comm_records, q3_comm_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Dashboard Controls

    The controls below filter **all visualizations simultaneously**, implementing the coordinated multiple views paradigm (Heer & Shneiderman, 2012). This enables cross-referencing patterns: for example, adjusting the similarity threshold updates both the network visualization AND the resolution candidates table.
    """)
    return


@app.cell
def _(mo):
    # Global filter controls that affect multiple visualizations
    sim_threshold = mo.ui.slider(
        start=0.1, stop=0.8, step=0.05, value=0.3, 
        label="🔗 Jaccard Similarity Threshold"
    )

    show_pseudonyms_only = mo.ui.checkbox(value=False, label="Show only pseudonym-involved pairs")

    entity_type_filter = mo.ui.multiselect(
        options=["Person", "Vessel", "Organization", "Group"],
        value=["Person", "Vessel", "Organization", "Group"],
        label="Entity Types to Include"
    )

    mo.hstack([sim_threshold, show_pseudonyms_only, entity_type_filter], justify="start", gap=2)
    return entity_type_filter, show_pseudonyms_only, sim_threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Pseudonym Detection via Naming Pattern Analysis

    **Method**: We identify potential pseudonyms using heuristic pattern matching on entity names. This approach is grounded in domain knowledge: criminal organizations commonly use role-based aliases to obscure true identities (Krebs, 2002).

    | Pattern | Examples | Rationale |
    |---------|----------|-----------|
    | **"The X"** | The Lookout, The Middleman, The Accountant | Role-based aliases commonly used in covert operations |
    | **"Mrs./Mr. X"** | Mrs. Money | Formal title with role-based surname |
    | **Title-only** | Boss, Small Fry | Single descriptive words suggesting rank/role |
    | **Single-word Person** | Sam, Kelly, Davis | Potentially first-name-only aliases (common obfuscation) |

    **Validation**: We cross-reference detected pseudonyms with the challenge problem statement, which confirms "Boss" and "The Lookout" as known aliases. Our heuristics correctly identify both, providing confidence in the approach.

    > **Transition to Next Section**: The detected pseudonyms become the focus of our similarity analysis we specifically look for entities that communicate with similar partners, suggesting they may be the same person using multiple aliases.
    """)
    return


@app.cell
def _(all_entities, entity_type_filter, pd):
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

    # Apply entity type filter
    _filtered_entities = {k: v for k, v in all_entities.items() 
                         if v.get('sub_type', '') in entity_type_filter.value}

    pseudonym_df = pd.DataFrame([detect_pseudonym(_eid, _ed) for _eid, _ed in _filtered_entities.items()])
    pseudonym_df = pseudonym_df.sort_values('pseudonym_score', ascending=False)
    likely_pseudonyms = pseudonym_df[pseudonym_df['is_likely_pseudonym']].copy()

    print(f"✓ Identified {len(likely_pseudonyms)} likely pseudonyms out of {len(_filtered_entities)} entities (filtered)")
    return likely_pseudonyms, pseudonym_df


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
            marker_color='#FFD700',  # Gold for all pseudonyms
            text=f"{_row['detected_patterns']}",
            textposition='outside',
            hovertemplate=f"<b>{_row['label']}</b><br>Type: {_row['sub_type']}<br>Pattern: {_row['detected_patterns']}<br>Score: {_row['pseudonym_score']}<extra></extra>",
            showlegend=False
        ))

    fig_pseudo_bar.update_layout(
        title=dict(text='<b>Visualization 1: Detected Pseudonyms by Pattern Score</b><br><sup>Higher score = more pseudonym indicators | All confirmed aliases shown in gold</sup>', x=0.5),
        xaxis_title='Pseudonym Score (number of pattern matches)',
        yaxis_title='Entity',
        height=max(400, len(_pseudo_summary) * 35),
        showlegend=False,
        bargap=0.3
    )

    mo.vstack([
        mo.md("### Visualization 1: Pseudonym Detection Results"),
        mo.md(f"**Summary**: Detected **{len(likely_pseudonyms)}** entities with pseudonym-like naming patterns. The role-based naming convention ('The Lookout', 'The Accountant', etc.) suggests an **organized operational hierarchy**."),
        fig_pseudo_bar
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Entity Similarity Analysis via Communication Partners

    **Method**: To identify entities that might be the **same person using different pseudonyms**, we compute **Jaccard similarity** based on shared communication partners:

    $$J(A, B) = \frac{|Partners(A) \cap Partners(B)|}{|Partners(A) \cup Partners(B)|}$$

    **Why This Works**: If two aliases are controlled by the same person, they likely communicate with many of the same partners (colleagues, clients, contacts). High Jaccard similarity between two entities suggests they:
    - Communicate with the same people/vessels
    - May be the same person operating under different aliases
    - Share operational roles within the network

    **Method Validation**:
    - Jaccard similarity is the standard metric for set-based entity resolution (Christen, 2012)
    - Bilgic et al. (2006) demonstrated its effectiveness specifically for social network deduplication
    - We validate by checking that known collaborators (e.g., vessels in the same organization) show high similarity

    > **Transition to Visualizations**: The similarity matrix computed here drives all subsequent visualizations. Each visualization offers a different perspective on the same underlying data, following the coordinated multiple views paradigm.
    """)
    return


@app.cell
def _(
    all_entities,
    combinations,
    datetime,
    entity_ids,
    entity_type_filter,
    np,
    pd,
    q3_comm_matrix,
):
    def get_partners(eid, comm_mat):
        _sent_to = set(comm_mat.get(eid, {}).keys())
        _recv_from = set(_s for _s, _targets in comm_mat.items() if eid in _targets)
        return _sent_to | _recv_from

    # Filter entities by type
    _filtered_entity_ids = {eid for eid in entity_ids 
                           if all_entities.get(eid, {}).get('sub_type', '') in entity_type_filter.value}

    entity_partners = {_eid: get_partners(_eid, q3_comm_matrix) for _eid in _filtered_entity_ids}

    def jaccard(set_a, set_b):
        if not set_a and not set_b:
            return 0.0
        _union = set_a | set_b
        return len(set_a & set_b) / len(_union) if len(_union) > 0 else 0.0

    # Build list of entities with at least one communication
    entity_list = sorted([_e for _e in _filtered_entity_ids if len(entity_partners.get(_e, set())) > 0])
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

    print(f"✓ Computed similarity for {len(entity_list)} active entities")
    print(f"✓ Found {len(similarity_df)} entity pairs with similarity > 0")
    print(f"✓ Top similarity: {similarity_df['jaccard'].max():.3f}" if len(similarity_df) > 0 else "No pairs found")
    return entity_list, entity_partners, similarity_df, similarity_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Bipartite Communication Network

    **Purpose**: This visualization shows **pseudonyms on the left** and their **communication partners on the right**, revealing the direct connections of suspected aliases.

    **Visual Encodings** (following Munzner, 2014):
    - **Position**: Bipartite layout separates pseudonyms (left) from non-pseudonyms (right)
    - **Node size**: Proportional to total message volume (sent + received)
    - **Edge thickness**: Proportional to number of messages between entities
    - **Color**: Gold = identified pseudonyms, Teal = other entities

    **Analytical Value**: Bipartite layouts are particularly effective for analyzing relationships between two distinct classes of entities. This view immediately reveals which pseudonyms are communication hubs and which partners are contacted by multiple aliases (suggesting coordination).

    > **What to Look For**: Pseudonyms that share many common partners on the right side are strong candidates for being the same person.
    """)
    return


@app.cell
def _(
    all_entities,
    comm_records,
    go,
    likely_pseudonyms,
    mo,
    np,
    q3_comm_matrix,
    sim_threshold,
):
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    _thresh = sim_threshold.value

    # Get communication partners of pseudonyms
    _partner_ids = set()
    _edges = []

    for _pid in _pseudonym_ids:
        for _target, _count in q3_comm_matrix.get(_pid, {}).items():
            if _target not in _pseudonym_ids:
                _partner_ids.add(_target)
                _edges.append((_pid, _target, _count))
        for _source, _targets in q3_comm_matrix.items():
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
            _sent = sum(q3_comm_matrix.get(_eid, {}).values())
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
            title=dict(text='<b>Visualization 2: Bipartite Network Pseudonyms ↔ Communication Partners</b><br><sup>Gold = Pseudonyms (left) | Teal = Partners (right) | Edge width ∝ message count</sup>', x=0.5),
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

    _n_shared = len(set(_partner_list) & set([p for e in _edges for p in [e[0], e[1]] if p not in _pseudonym_ids]))

    mo.vstack([
        mo.md("### Visualization 2: Bipartite Communication Network"),
        mo.md(f"**Insight**: {len(_pseudo_list)} pseudonyms communicate with {len(_partner_list)} unique partners. Partners contacted by multiple pseudonyms (visible as multiple edge connections on right) suggest coordinated operations."),
        fig_bipartite
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Hierarchical Clustering of Entity Similarity

    **Purpose**: This heatmap shows **pairwise Jaccard similarity** between all communicating entities, with rows and columns reordered by **hierarchical clustering** to reveal hidden structure.

    **Visual Encodings**:
    - **Position**: Entities ordered by average-linkage hierarchical clustering
    - **Color intensity**: Jaccard similarity (Viridis colorscale: dark = low, bright = high)
    - **Diagonal structure**: Clusters of related entities appear as bright blocks along diagonal

    **Method**: We use average-linkage clustering, which balances single-linkage's chaining tendency and complete-linkage's tight clusters (Kaufman & Rousseeuw, 1990).

    > **What to Look For**: Bright off-diagonal cells indicate entity pairs with high similarity—strong candidates for being the same person under different aliases.
    """)
    return


@app.cell
def _(
    all_entities,
    entity_list,
    go,
    leaves_list,
    likely_pseudonyms,
    linkage,
    mo,
    np,
    pdist,
    sim_threshold,
    similarity_matrix,
):
    _thresh = sim_threshold.value

    if len(entity_list) > 3:
        # Filter to entities with some similarity
        _row_sums = similarity_matrix.sum(axis=1)
        _active_mask = _row_sums > 0
        _active_indices = np.where(_active_mask)[0]

        if len(_active_indices) > 3:
            _sub_matrix = similarity_matrix[np.ix_(_active_indices, _active_indices)]
            _sub_entities = [entity_list[_i] for _i in _active_indices]
            _sub_labels = [all_entities.get(_e, {}).get('label', _e)[:15] for _e in _sub_entities]

            # Mark pseudonyms with ★
            _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
            _sub_labels = [f"★{_l}" if _sub_entities[_i] in _pseudonym_ids else _l 
                          for _i, _l in enumerate(_sub_labels)]

            # Hierarchical clustering
            _dist = pdist(_sub_matrix + 0.001)
            _linkage_mat = linkage(_dist, method='average')
            _order = leaves_list(_linkage_mat)

            # Reorder matrix and labels
            _ordered_matrix = _sub_matrix[_order, :][:, _order]
            _ordered_labels = [_sub_labels[_i] for _i in _order]

            # Apply threshold masking for visualization
            _display_matrix = np.where(_ordered_matrix >= _thresh, _ordered_matrix, 0)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=_display_matrix,
                x=_ordered_labels,
                y=_ordered_labels,
                colorscale='Viridis',
                hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Jaccard: %{z:.3f}<extra></extra>',
                colorbar=dict(title=f'Jaccard<br>(≥{_thresh})')
            ))

            fig_heatmap.update_layout(
                title=dict(text=f'<b>Visualization 3: Entity Similarity Heatmap (threshold ≥ {_thresh})</b><br><sup>★ = Pseudonym | Bright = high similarity | Hierarchically clustered</sup>', x=0.5),
                height=700, width=850,
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8), autorange='reversed')
            )

            # Count high-similarity pairs
            _n_high = np.sum(_ordered_matrix >= _thresh) // 2
        else:
            fig_heatmap = go.Figure()
            fig_heatmap.add_annotation(text="Insufficient similar entities for clustering", showarrow=False)
            fig_heatmap.update_layout(height=300)
            _n_high = 0
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.add_annotation(text="Insufficient data for heatmap", showarrow=False)
        fig_heatmap.update_layout(height=300)
        _n_high = 0

    mo.vstack([
        mo.md("### Visualization 3: Similarity Heatmap — Clustered by Communication Patterns"),
        mo.md(f"**Insight**: Hierarchical clustering reveals {_n_high if '_n_high' in dir() else 0} entity pairs with similarity ≥ {_thresh}. Diagonal blocks indicate communities of entities that share communication partners."),
        fig_heatmap
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Temporal Activity Fingerprint Matrix

    **Purpose**: This heatmap shows **hourly activity patterns** for each entity, enabling temporal fingerprint comparison for identity disambiguation.

    **Visual Encodings**:
    - **Rows**: Entities (sorted by total activity, ★ marks pseudonyms)
    - **Columns**: Hours of the day (0-23)
    - **Color intensity**: Normalized message count per hour (YlOrRd: yellow = low, red = high)

    **Analytical Principle**: A single person cannot communicate as two aliases simultaneously. Therefore:
    - **Non-overlapping patterns** → Could be the same person using different aliases at different times
    - **Overlapping patterns** → Likely different people (both active at same hours)

    This temporal fingerprinting technique is widely used in social network forensics (Hochheiser & Shneiderman, 2004).

    > **What to Look For**: Pseudonyms (★) with complementary temporal patterns one active when the other is silent are prime candidates for being the same person.
    """)
    return


@app.cell
def _(
    all_entities,
    comm_records,
    datetime,
    entity_type_filter,
    go,
    likely_pseudonyms,
    mo,
    np,
    pd,
):
    # Build hourly activity matrix for all active entities
    _activity_data = []

    for _rec in comm_records:
        _sender_type = all_entities.get(_rec['sender'], {}).get('sub_type', '')
        _receiver_type = all_entities.get(_rec['receiver'], {}).get('sub_type', '')

        if _sender_type in entity_type_filter.value or _receiver_type in entity_type_filter.value:
            try:
                _ts = datetime.fromisoformat(_rec['timestamp'].replace('Z', '+00:00'))
                _hour = _ts.hour
                if _sender_type in entity_type_filter.value:
                    _activity_data.append({'entity': _rec['sender'], 'hour': _hour})
                if _receiver_type in entity_type_filter.value:
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

            # Get labels with pseudonym markers
            _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
            _labels = []
            _n_pseudo_temporal = 0
            for _eid in _pivot_norm.index:
                _label = all_entities.get(_eid, {}).get('label', _eid)[:15]
                if _eid in _pseudonym_ids:
                    _label = f"★ {_label}"
                    _n_pseudo_temporal += 1
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
                title=dict(text='<b>Visualization 4: Temporal Activity Fingerprints</b><br><sup>★ = Pseudonym | Red = peak activity | Non-overlapping patterns suggest same person</sup>', x=0.5),
                xaxis_title='Hour of Day',
                yaxis_title='Entity',
                height=max(400, len(_labels) * 20),
                yaxis=dict(tickfont=dict(size=9))
            )
        else:
            fig_temporal = go.Figure()
            fig_temporal.add_annotation(text="Insufficient active entities", showarrow=False)
            fig_temporal.update_layout(height=300)
            _n_pseudo_temporal = 0
    else:
        fig_temporal = go.Figure()
        fig_temporal.add_annotation(text="No temporal data available", showarrow=False)
        fig_temporal.update_layout(height=300)
        _n_pseudo_temporal = 0

    mo.vstack([
        mo.md("### Visualization 4: Temporal Activity Matrix"),
        mo.md(f"**Insight**: Activity concentrated between 08:00-14:00 (business hours). {_n_pseudo_temporal} pseudonyms visible (★). Look for complementary patterns pseudonyms active at different hours could be the same person."),
        fig_temporal
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Interactive Similarity Network

    **Purpose**: This **force-directed graph** positions entities based on their communication similarity, with the threshold controlled by the global slider above.

    **Visual Encodings**:
    - **Edges**: Connect entities with Jaccard similarity ≥ threshold (controlled by slider)
    - **Node proximity**: Similar entities pulled together by spring forces
    - **Node color**: Gold = identified pseudonyms, colored by type for others
    - **Node size**: Larger = pseudonym

    **Interaction**: Adjust the global "Jaccard Similarity Threshold" slider to explore different levels of entity clustering. Higher thresholds reveal only the strongest potential identity matches.

    **Layout Algorithm**: Fruchterman-Reingold spring layout (NetworkX), which models edges as springs pulling connected nodes together while unconnected nodes repel.

    > **What to Look For**: Tightly clustered groups containing multiple pseudonyms suggest those aliases may be controlled by the same person or coordinated group.
    """)
    return


@app.cell
def _(
    all_entities,
    go,
    likely_pseudonyms,
    mo,
    np,
    nx,
    show_pseudonyms_only,
    sim_threshold,
    similarity_df,
):
    _thresh = sim_threshold.value
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())

    _G = nx.Graph()
    _filtered = similarity_df[similarity_df['jaccard'] >= _thresh]

    # Apply pseudonym-only filter if checked
    if show_pseudonyms_only.value:
        _filtered = _filtered[
            (_filtered['entity_a'].isin(_pseudonym_ids)) | 
            (_filtered['entity_b'].isin(_pseudonym_ids))
        ]

    for _, _row in _filtered.iterrows():
        _ea, _eb = _row['entity_a'], _row['entity_b']
        _G.add_node(_ea, label=all_entities[_ea].get('label', _ea),
                   sub_type=all_entities[_ea].get('sub_type', 'Unknown'))
        _G.add_node(_eb, label=all_entities[_eb].get('label', _eb),
                   sub_type=all_entities[_eb].get('sub_type', 'Unknown'))
        _G.add_edge(_ea, _eb, weight=_row['jaccard'])

    if len(_G.nodes) > 1:
        _pos = nx.spring_layout(_G, k=2/np.sqrt(len(_G.nodes)+1), iterations=50, seed=42)

        # Draw edges with width proportional to similarity
        _edge_traces = []
        for _e in _G.edges(data=True):
            _x0, _y0 = _pos[_e[0]]
            _x1, _y1 = _pos[_e[1]]
            _weight = _e[2].get('weight', 0.5)
            _edge_traces.append(go.Scatter(
                x=[_x0, _x1, None], y=[_y0, _y1, None],
                mode='lines',
                line=dict(width=max(1, _weight * 5), color=f'rgba(150,150,150,{0.3 + _weight * 0.5})'),
                hoverinfo='none',
                showlegend=False
            ))

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
                _sizes.append(25)
            else:
                _colors.append(_type_colors.get(_n[1].get('sub_type', ''), '#999'))
                _sizes.append(15)

        fig_force = go.Figure()
        for _trace in _edge_traces:
            fig_force.add_trace(_trace)
        fig_force.add_trace(go.Scatter(
            x=_node_x, y=_node_y, mode='markers+text',
            marker=dict(size=_sizes, color=_colors, line=dict(width=2, color='white')),
            text=_texts, textposition='top center', textfont=dict(size=9),
            hovertext=_hovers, hoverinfo='text'
        ))

        _n_pseudo_in_net = len([n for n in _G.nodes() if n in _pseudonym_ids])
        _n_components = nx.number_connected_components(_G)

        fig_force.update_layout(
            title=dict(text=f'<b>Visualization 5: Similarity Network (threshold ≥ {_thresh:.2f})</b><br><sup>Nodes: {len(_G.nodes)} | Edges: {len(_G.edges)} | Components: {_n_components} | Gold = Pseudonym</sup>', x=0.5),
            height=600, showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#fafafa'
        )
    else:
        fig_force = go.Figure()
        fig_force.add_annotation(text=f"No entity pairs with similarity ≥ {_thresh:.2f}", showarrow=False)
        fig_force.update_layout(height=300)
        _n_pseudo_in_net = 0
        _n_components = 0

    mo.vstack([
        mo.md(f"### Visualization 5: Force-Directed Network (Threshold = {_thresh:.2f})"),
        mo.md(f"**Insight**: At threshold {_thresh:.2f}, the network has **{len(_G.nodes) if '_G' in dir() and len(_G.nodes) > 0 else 0} entities** in **{_n_components} connected components**. Pseudonyms in the same component share communication partners."),
        fig_force
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Sankey Diagram: Pseudonym Resolution Flow

    **Purpose**: This **Sankey diagram** visualizes potential entity resolution mappings, showing which real identities might correspond to each pseudonym.

    **Visual Encodings**:
    - **Left side (Gold)**: Identified pseudonyms
    - **Right side (Teal)**: Candidate real identities (non-pseudonyms with high similarity)
    - **Flow width**: Proportional to Jaccard similarity score

    **Analytical Value**: Sankey diagrams excel at showing many-to-many relationships. A pseudonym flowing to multiple candidates indicates uncertainty; a single strong flow suggests a confident match.

    > **What to Look For**: Wider flows indicate stronger evidence. Multiple thin flows suggest the pseudonym's identity remains ambiguous.
    """)
    return


@app.cell
def _(go, likely_pseudonyms, mo, sim_threshold, similarity_df):
    _thresh = sim_threshold.value
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())
    _flows = []

    # Filter by threshold
    _filtered_sim = similarity_df[similarity_df['jaccard'] >= _thresh]

    for _, _row in _filtered_sim.head(50).iterrows():
        _is_pa = _row['entity_a'] in _pseudonym_ids
        _is_pb = _row['entity_b'] in _pseudonym_ids

        # We want flows FROM pseudonym TO candidate
        if _is_pa and not _is_pb:
            _flows.append({'source': _row['label_a'], 'target': _row['label_b'] + ' ', 'value': _row['jaccard']})
        elif _is_pb and not _is_pa:
            _flows.append({'source': _row['label_b'], 'target': _row['label_a'] + ' ', 'value': _row['jaccard']})
        elif _is_pa and _is_pb:
            # Both are pseudonyms - show connection (may be same person with two aliases)
            _flows.append({'source': _row['label_a'], 'target': _row['label_b'] + ' (alias?)', 'value': _row['jaccard']})

    if _flows:
        _nodes = list(set([_f['source'] for _f in _flows] + [_f['target'] for _f in _flows]))
        _node_idx = {_n: _i for _i, _n in enumerate(_nodes)}
        _node_colors = ['#FFD700' if not _n.endswith(' ') and 'alias' not in _n else '#4ECDC4' for _n in _nodes]

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

        _n_pseudo_sankey = len(set([_f['source'] for _f in _flows]))
        _n_candidates = len(set([_f['target'] for _f in _flows]))

        fig_sankey.update_layout(
            title=dict(text=f'<b>Visualization 6: Pseudonym Resolution Candidates (threshold ≥ {_thresh})</b><br><sup>Gold = Pseudonyms | Teal = Candidates | Flow width ∝ similarity</sup>', x=0.5),
            height=500, font=dict(size=10)
        )
    else:
        fig_sankey = go.Figure()
        fig_sankey.add_annotation(text=f"No resolution candidates at threshold ≥ {_thresh}", showarrow=False)
        fig_sankey.update_layout(height=300)
        _n_pseudo_sankey = 0
        _n_candidates = 0

    mo.vstack([
        mo.md("### Visualization 6: Sankey Diagram: Entity Resolution Flow"),
        mo.md(f"**Insight**: At threshold {_thresh:.2f}, **{_n_pseudo_sankey if '_n_pseudo_sankey' in dir() else 0} pseudonyms** connect to **{_n_candidates if '_n_candidates' in dir() else 0} candidate identities**. Wider flows indicate stronger evidence of identity match."),
        fig_sankey
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Parallel Coordinates: Multi-Dimensional Entity Comparison

    **Purpose**: This visualization displays each entity as a **polyline** crossing multiple parallel axes, enabling comparison across multiple attributes simultaneously.

    **Dimensions Shown**:
    - **Entity Type**: Person, Vessel, Organization, Group
    - **Pseudonym Score**: Number of alias pattern matches (higher = more suspicious)
    - **Messages Sent/Received**: Communication volume
    - **Unique Partners**: Network breadth
    - **Active Hours**: Temporal spread

    **Interaction**: Click and drag on any axis to filter entities. This implements brushing a key technique for multidimensional data exploration (Inselberg, 2009).

    **Color**: Gold lines = pseudonyms, Teal = others

    > **What to Look For**: Entities with similar profiles across multiple dimensions (parallel lines) may be related or operated by the same person.
    """)
    return


@app.cell
def _(
    comm_records,
    datetime,
    entity_partners,
    entity_type_filter,
    go,
    mo,
    pd,
    pseudonym_df,
    q3_comm_matrix,
):
    _entity_features = []

    for _, _row in pseudonym_df.iterrows():
        if _row['sub_type'] not in entity_type_filter.value:
            continue

        _eid = _row['entity_id']
        _sent = sum(q3_comm_matrix.get(_eid, {}).values())
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

        _n_active = len(_features_df)
        _n_pseudo_pc = len(_features_df[_features_df['is_pseudonym'] == 1])

        fig_parallel.update_layout(
            title=dict(text='<b>Visualization 7: Parallel Coordinates — Multi-Dimensional Entity Comparison</b><br><sup>Each line = one entity | Gold = Pseudonym | Drag on axes to filter</sup>', x=0.5),
            height=500, margin=dict(l=100, r=100)
        )
    else:
        fig_parallel = go.Figure()
        fig_parallel.add_annotation(text="No entity data available", showarrow=False)
        fig_parallel.update_layout(height=300)
        _n_active = 0
        _n_pseudo_pc = 0

    mo.vstack([
        mo.md("### Visualization 7: Parallel Coordinates: Multi-Dimensional Entity Analysis"),
        mo.md(f"**Insight**: Comparing **{_n_active if '_n_active' in dir() else 0} active entities** across 6 dimensions. **{_n_pseudo_pc if '_n_pseudo_pc' in dir() else 0} pseudonyms** (gold) can be filtered by dragging on any axis."),
        fig_parallel
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. Top Entity Resolution Candidates

    **Purpose**: This interactive table shows the **highest-similarity entity pairs**, ranked by Jaccard similarity. These are the strongest candidates for being the same person/entity using different names.

    **Columns**:
    - **Entity A / B**: The two entities being compared
    - **Jaccard Similarity**: Overlap in communication partners (0-1)
    - **Shared / Total Partners**: Raw counts for interpretability

    **Filtering**: Respects the global similarity threshold and pseudonym-only checkbox above.

    > **Actionable Output**: This table provides Clepper with a prioritized list of identity matches to investigate further.
    """)
    return


@app.cell
def _(
    likely_pseudonyms,
    mo,
    show_pseudonyms_only,
    sim_threshold,
    similarity_df,
):
    _thresh = sim_threshold.value
    _pseudonym_ids = set(likely_pseudonyms['entity_id'].tolist())

    # Apply threshold filter
    _filtered = similarity_df[similarity_df['jaccard'] >= _thresh].copy()

    # Apply pseudonym-only filter if checked
    if show_pseudonyms_only.value:
        _filtered = _filtered[
            (_filtered['entity_a'].isin(_pseudonym_ids)) |
            (_filtered['entity_b'].isin(_pseudonym_ids))
        ]

    _relevant = _filtered.head(20).copy()

    if len(_relevant) > 0:
        # Add pseudonym markers
        _relevant['Entity A'] = _relevant.apply(
            lambda r: f"★ {r['label_a']}" if r['entity_a'] in _pseudonym_ids else r['label_a'], axis=1
        )
        _relevant['Entity B'] = _relevant.apply(
            lambda r: f"★ {r['label_b']}" if r['entity_b'] in _pseudonym_ids else r['label_b'], axis=1
        )

        _display_df = _relevant[['Entity A', 'Entity B', 'jaccard', 'shared_partners', 'total_partners']].copy()
        _display_df.columns = ['Entity A', 'Entity B', 'Jaccard Similarity', 'Shared Partners', 'Total Partners']
        _display_df['Jaccard Similarity'] = _display_df['Jaccard Similarity'].round(3)

        _table = mo.ui.table(_display_df, selection=None)

        _n_pairs = len(_relevant)
        _n_pseudo_pairs = len(_relevant[
            (_relevant['entity_a'].isin(_pseudonym_ids)) | 
            (_relevant['entity_b'].isin(_pseudonym_ids))
        ])

        mo.vstack([
            mo.md("### Visualization 8: Top Resolution Candidates Table"),
            mo.md(f"**Summary**: **{_n_pairs} entity pairs** with similarity ≥ {_thresh:.2f}. **{_n_pseudo_pairs}** involve at least one pseudonym (★)."),
            _table
        ])
    else:
        mo.vstack([
            mo.md("### Visualization 8: Top Resolution Candidates Table"),
            mo.md(f"No entity pairs found with similarity ≥ {_thresh:.2f}. Try lowering the threshold.")
        ])
    return


app._unparsable_cell(
    """
    _n_pseudo = len(likely_pseudonyms)
    _thresh = sim_threshold.value
    _top_pairs = similarity_df[similarity_df['jaccard'] >= _thresh].head(5) if len(similarity_df) > 0 else None

    mo.md(f\"\"\"
    ---

    ## Key Findings for Question 3

    ### 3.1 Who is Using Pseudonyms? ({_n_pseudo} entities identified)

    Through naming pattern analysis (Section 1), we identified the following likely pseudonyms:

    | Pseudonym | Pattern | Investigative Significance |
    |-----------|---------|---------------------------|
    | **Boss** | Title-like | Central command/coordination role likely the operation's leader |
    | **The Lookout** | \"The X\" | Surveillance operations monitors activities and reports |
    | **The Middleman** | \"The X\" | Logistics/brokerage facilitates transactions between parties |
    | **The Accountant** | \"The X\" | Financial operations manages money flows |
    | **Mrs. Money** | \"Mrs. X\" | Financial handler possibly works with The Accountant |
    | **The Intern** | \"The X\" | Junior operative newcomer to the organization |
    | **Small Fry** | Title-like | Minor player/low rank likely handles small tasks |
    | **Sam, Kelly, Davis, Elise, Rodriguez** | Single-word heuristic | **Ambiguous** may be real names or radio handles; requires further evidence |

    **Validation**: The challenge confirms \"Boss\" and \"The Lookout\" as known aliases. Our heuristics correctly identify both. The 7 role-based pseudonyms above are high-confidence; the 5 first-name entities are flagged for further corroboration only.

    **Tier 2: Ambiguous first-name handles** (Jaccard evidence provided where available):
    - **Davis**: Jaccard 0.667 with V. Miesel Shipping (8 shared partners) deeply embedded in the criminal cluster
    - **Rodriguez**: Jaccard 0.636 with V. Miesel Shipping (7 shared partners) strong operational overlap
    - **Sam, Kelly, Elise**: Flagged by single-word heuristic only; legitimate maritime first-name use cannot be ruled out

    ### 3.2 How Do Visualizations Help Clepper?

    Each visualization provides a **unique analytical perspective**:

    | # | Visualization | Insight Provided | Analytical Value |
    |---|--------------|------------------|------------------|
    | 1 | Pseudonym Detection Bar | Identifies and ranks suspected aliases | Prioritizes investigation targets |
    | 2 | Bipartite Network | Shows direct communication relationships | Reveals who pseudonyms contact |
    | 3 | Similarity Heatmap | Clusters entities by communication overlap | Identifies entity groups |
    | 4 | Temporal Matrix | Shows hourly activity fingerprints | Finds non-overlapping schedules |
    | 5 | Force-Directed Network | Interactive similarity exploration | Discovers unexpected connections |
    | 6 | Sankey Diagram | Maps pseudonyms to candidate identities | Visualizes resolution hypotheses |
    | 7 | Parallel Coordinates | Multi-dimensional entity comparison | Finds similar entity profiles |
    | 8 | Resolution Table | Ranked list of identity matches | Actionable investigation list |

    **Coordinated Views**: The global threshold slider (Section 2) updates all visualizations simultaneously, enabling cross-referencing. For example, adjusting threshold reveals which pseudonyms remain connected at higher similarity levels.

    ### 3.3 How Does Understanding Change with Pseudonyms?

    With pseudonym awareness, Clepper can:

    1. **Reveal the operational hierarchy**: The \"The X\" naming convention maps onto a command structure, Boss issues orders, The Middleman and Mrs. Money manage logistics and finance, The Intern and Small Fry execute operations. This is invisible until the alias pattern is recognised.
    2. **Link pseudonyms to vessels**: The Lookout shares 4/7 partners with vessel Seawatch (Jaccard 0.571), and Small Fry shares 2/3 partners with vessel Knowles (Jaccard 0.667) strongly suggesting these pseudonyms operate those vessels.
    3. **Identify V. Miesel Shipping as an operational hub**: Davis (J=0.667) and Rodriguez (J=0.636) both share the majority of their partners with V. Miesel Shipping, connecting them to the criminal cluster.
    4. **Detect counter-intelligence activity**: The Lookout holds a Suspicious relationship with Clepper Jensen and reports to the Green Guardians, an operative embedded in a legitimate environmental group to monitor the investigation.
    5. **Consolidate the network**: Resolving the 7 confirmed pseudonyms to probable real identities reduces apparent complexity and reveals a tighter, more coherent criminal structure.

    **Top Resolution Candidates** (at current threshold {_thresh:.2f}):
    {\"\".join([f\"- **{r['label_a']}** ↔ **{r['label_b']}**: Jaccard = {r['jaccard']:.3f} ({r['shared_partners']} shared partners)\" + chr(10) for _, r in (_top_pairs.iterrows() if _top_pairs is not None and len(_top_pairs) > 0 else [])])}
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## **Findings for Question 4**

    4. Clepper suspects that Nadia Conti, who was formerly entangled in an illegal fishing scheme, may have continued illicit activity within Oceanus.

        - a) Through visual analytics, provide evidence that Nadia is, or is not, doing something illegal.
        - b) Summarize Nadia’s actions visually. Are Clepper’s suspicions justified?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The communication profile of Nadia Conti reveals that a significant number of her messages are sent to what appear to be pseudonyms.
    ![alt](public/NadiaContiContactlist.png)
    The Communication Intelligence Dashboard reveals messages sent between Naida and others, many of which have a high suspicion rating. The message categories by sender graph shows Nadia having sent two “cover story” messages, four “covert coordination” messages, and two “illegal activity” messages, strongly indicating that she is, indeed, engaged in illicit activity within Oceanus. The average suspicion risk of her messages are a 6.0 (out of 10), and she is at high risk (12, with >7 being high) of being a person of interest in illegal activities. In particular, her and Liam Thorne, whom the intelligence dashboard marks as the entity with the highest suspicion rating, share four messages between them that have an average suspicion rating of 8.0 (out of 10).
    ![alt](public/NadiaContiCommunicationsDashboard.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## References

    1. **Bilgic, M., Licamele, L., Getoor, L., & Shneiderman, B.** (2006). "D-Dupe: An Interactive Tool for Entity Resolution in Social Networks." *IEEE Symposium on Visual Analytics Science and Technology (VAST)*, pp. 43-50. DOI: 10.1109/VAST.2006.261430

    2. **Bostock, M., Ogievetsky, V., & Heer, J.** (2011). "D3: Data-Driven Documents." *IEEE Transactions on Visualization and Computer Graphics*, 17(12), pp. 2301-2309. DOI: 10.1109/TVCG.2011.185

    3. **Christen, P.** (2012). *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*. Springer. ISBN: 978-3-642-31163-5

    4. **Fruchterman, T. M. J., & Reingold, E. M.** (1991). "Graph Drawing by Force-directed Placement." *Software: Practice and Experience*, 21(11), pp. 1129-1164.

    5. **Heer, J., & Shneiderman, B.** (2012). "Interactive Dynamics for Visual Analysis." *Communications of the ACM*, 55(4), pp. 45-54.

    6. **Hochheiser, H., & Shneiderman, B.** (2004). "Dynamic Query Tools for Time Series Data Sets." *Proc. IEEE InfoVis*, pp. 31-38.

    7. **IEEE VAST Challenge** (2025). Visual Analytics Science and Technology Challenge. https://vast-challenge.github.io/2025/

    8. **Inselberg, A.** (2009). *Parallel Coordinates: Visual Multidimensional Geometry and Its Applications*. Springer.

    9. **Kaufman, L., & Rousseeuw, P. J.** (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.

    10. **Krebs, V. E.** (2002). "Mapping Networks of Terrorist Cells." *Connections*, 24(3), pp. 43-52.

    11. **Munzner, T.** (2014). *Visualization Analysis and Design*. CRC Press. ISBN: 978-1466508910

    12. **NetworkX Developers** (2024). NetworkX: Network Analysis in Python. https://networkx.org/

    13. **Plotly Technologies Inc.** (2024). Plotly Python Graphing Library. https://plotly.com/python/

    14. **Shneiderman, B.** (1996). "The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations." *Proc. IEEE Symposium on Visual Languages*, pp. 336-343.
    "\"\")
    """)
    return


if __name__ == "__main__":
    app.run()
