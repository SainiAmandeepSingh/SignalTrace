import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


# =============================================
# BACKGROUND & QUESTIONS
# =============================================


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


# =============================================
# SHARED: IMPORTS
# =============================================


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
        Counter,
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
        px,
        time,
    )


# =============================================
# SHARED: DATA LOADING
# =============================================


@app.cell
def _(json_graph, json_lib):
    with open("data/MC3_graph.json", "r") as _f:
        graph_data = json_lib.load(_f)
    G = json_graph.node_link_graph(graph_data, edges="edges")
    return G, graph_data


# =============================================
# SHARED: ENTITY LOOKUPS
# =============================================


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


# =============================================
# SHARED: COMMUNICATION DATA & RELATIONSHIPS
# =============================================


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
                if _sender in entity_ids or _receiver in entity_ids:
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

    print(f"Extracted {len(comm_events)} communications and {len(relationship_data)} formal relationships")
    return comm_events, comm_matrix, edges_from, edges_to, relationship_data


# =============================================
# SHARED: messages_df (unified communication DataFrame)
# Columns: event_id, datetime, content, source (name), target (name)
# =============================================


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


# =============================================
# QUESTION 1: Temporal Patterns & Communication Intelligence
# =============================================


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


# =============================================
# QUESTION 2.1: Interactions Between Vessels and People
# =============================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2.1: Understanding Interactions Between Vessels and People

    - *2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.*
        - *a. Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.*

    To answer these questions, I create interactive visualization to explore:
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

    In this visualization, I show the ***communication flow*** between all entities (persons, vessels, and organizations). This answers the fundamental question: **who talks to whom and how often?**

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

    layout_select = mo.ui.dropdown(
        options=['spring', 'circular', 'kamada_kawai'],
        value='spring',
        label="Network Layout:"
    )

    mo.hstack([node_type_filter, min_comm_slider, layout_select], justify='start', gap=2)
    return layout_select, min_comm_slider, node_type_filter


@app.cell
def _(
    all_entities,
    comm_matrix,
    go,
    layout_select,
    min_comm_slider,
    node_type_filter,
    nodes_by_id,
    nx,
):
    def build_comm_network(entity_types, min_comms, layout):
        filtered_entities = {
            eid: e for eid, e in all_entities.items()
            if e.get('sub_type') in entity_types
        }

        G_comm_local = nx.DiGraph()

        for eid, entity in filtered_entities.items():
            G_comm_local.add_node(eid,
                      label=entity.get('name', eid),
                      sub_type=entity.get('sub_type'))

        edge_weights = {}
        for sender in comm_matrix:
            if sender not in filtered_entities:
                continue
            for receiver, comms in comm_matrix[sender].items():
                if receiver not in filtered_entities:
                    continue
                weight = len(comms)
                if weight >= min_comms:
                    G_comm_local.add_edge(sender, receiver, weight=weight)
                    edge_weights[(sender, receiver)] = weight

        if layout == 'spring':
            pos = nx.spring_layout(G_comm_local, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G_comm_local)
        else:
            pos = nx.kamada_kawai_layout(G_comm_local)

        return G_comm_local, pos, edge_weights

    G_comm, pos_comm, edge_weights_comm = build_comm_network(
        node_type_filter.value,
        min_comm_slider.value,
        layout_select.value
    )

    _color_map = {
        'Person': '#4ECDC4',
        'Vessel': '#FF6B6B',
        'Organization': '#95E1D3',
        'Group': '#F38181'
    }

    _edge_traces = []
    for (_u, _v), _weight in edge_weights_comm.items():
        _x0, _y0 = pos_comm[_u]
        _x1, _y1 = pos_comm[_v]

        _edge_trace = go.Scatter(
            x=[_x0, _x1, None],
            y=[_y0, _y1, None],
            mode='lines',
            line=dict(
                width=min(_weight / 2, 8),
                color='rgba(150, 150, 150, 0.4)'
            ),
            hoverinfo='text',
            text=f"{_u} → {_v}: {_weight} messages",
            showlegend=False
        )
        _edge_traces.append(_edge_trace)

    _node_traces = []
    for _entity_type in node_type_filter.value:
        _nodes_of_type = [n for n in G_comm.nodes()
                        if nodes_by_id.get(n, {}).get('sub_type') == _entity_type]

        if not _nodes_of_type:
            continue

        _x_vals = [pos_comm[n][0] for n in _nodes_of_type]
        _y_vals = [pos_comm[n][1] for n in _nodes_of_type]

        _sizes = [max(15, min(50, G_comm.degree(n) * 5)) for n in _nodes_of_type]

        _hover_texts = []
        for _n in _nodes_of_type:
            _in_deg = G_comm.in_degree(_n)
            _out_deg = G_comm.out_degree(_n)
            _hover_texts.append(f"<b>{_n}</b><br>Type: {_entity_type}<br>Sent to: {_out_deg} entities<br>Received from: {_in_deg} entities")

        _trace = go.Scatter(
            x=_x_vals,
            y=_y_vals,
            mode='markers+text',
            marker=dict(
                size=_sizes,
                color=_color_map[_entity_type],
                line=dict(width=2, color='white')
            ),
            text=[n if len(n) < 15 else n[:12]+'...' for n in _nodes_of_type],
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=_hover_texts,
            name=_entity_type
        )
        _node_traces.append(_trace)

    fig_comm_network = go.Figure(data=_edge_traces + _node_traces)

    fig_comm_network.update_layout(
        title=dict(
            text='<b>Communication Network</b><br><sup>Who talks to whom? Edge thickness = message frequency</sup>',
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig_comm_network
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

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_hm,
        x=_labels_hm,
        y=_labels_hm,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Messages: %{z}<extra></extra>'
    ))

    fig_heatmap.update_layout(
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

    fig_heatmap
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
def _(
    Counter,
    all_entities,
    go,
    nodes_by_id,
    nx,
    rel_type_filter,
    relationship_data,
):
    def build_rel_network(rel_types):
        G_rel_local = nx.Graph()

        filtered_rels = [r for r in relationship_data if r['type'] in rel_types]

        _edge_rels = Counter()
        for _rel in filtered_rels:
            _e1, _e2 = _rel['entity1'], _rel['entity2']
            if _e1 in all_entities and _e2 in all_entities:
                _key = tuple(sorted([_e1, _e2]))
                _edge_rels[(_key, _rel['type'])] += 1

        for (_key, _rtype), _count in _edge_rels.items():
            _e1, _e2 = _key
            G_rel_local.add_node(_e1, sub_type=all_entities[_e1].get('sub_type'))
            G_rel_local.add_node(_e2, sub_type=all_entities[_e2].get('sub_type'))

            if G_rel_local.has_edge(_e1, _e2):
                G_rel_local[_e1][_e2]['types'].append(_rtype)
            else:
                G_rel_local.add_edge(_e1, _e2, types=[_rtype])

        return G_rel_local

    G_rel = build_rel_network(rel_type_filter.value)

    if len(G_rel.nodes()) > 0:
        pos_rel = nx.spring_layout(G_rel, k=3, iterations=50, seed=42)
    else:
        pos_rel = {}

    _rel_color_map = {
        'Colleagues': '#2ECC71',
        'Operates': '#3498DB',
        'Reports': '#9B59B6',
        'Coordinates': '#F39C12',
        'Suspicious': '#E74C3C',
        'Friends': '#1ABC9C',
        'Unfriendly': '#C0392B',
    }

    _node_color_map = {
        'Person': '#4ECDC4',
        'Vessel': '#FF6B6B',
        'Organization': '#95E1D3',
        'Group': '#F38181'
    }

    _rel_edge_traces = []
    for _rtype in rel_type_filter.value:
        _x_edges, _y_edges = [], []
        for _u, _v, _data in G_rel.edges(data=True):
            if _rtype in _data['types']:
                _x0, _y0 = pos_rel[_u]
                _x1, _y1 = pos_rel[_v]
                _x_edges.extend([_x0, _x1, None])
                _y_edges.extend([_y0, _y1, None])

        if _x_edges:
            _trace = go.Scatter(
                x=_x_edges, y=_y_edges,
                mode='lines',
                line=dict(width=2, color=_rel_color_map.get(_rtype, 'gray')),
                name=_rtype,
                hoverinfo='none'
            )
            _rel_edge_traces.append(_trace)

    _rel_node_traces = []
    for _ntype in ['Person', 'Vessel', 'Organization', 'Group']:
        _nodes_of_type = [n for n in G_rel.nodes()
                        if nodes_by_id.get(n, {}).get('sub_type') == _ntype]
        if not _nodes_of_type:
            continue

        _x_vals = [pos_rel[n][0] for n in _nodes_of_type]
        _y_vals = [pos_rel[n][1] for n in _nodes_of_type]

        _hover_texts = []
        for _n in _nodes_of_type:
            _neighbors = list(G_rel.neighbors(_n))
            _rel_summary = []
            for _nb in _neighbors:
                _types = G_rel[_n][_nb]['types']
                _rel_summary.append(f"  • {_nb}: {', '.join(_types)}")
            _hover_text = f"<b>{_n}</b> ({_ntype})<br>Relationships:<br>" + "<br>".join(_rel_summary[:10])
            if len(_rel_summary) > 10:
                _hover_text += f"<br>  ...and {len(_rel_summary)-10} more"
            _hover_texts.append(_hover_text)

        _trace = go.Scatter(
            x=_x_vals, y=_y_vals,
            mode='markers+text',
            marker=dict(
                size=20,
                color=_node_color_map[_ntype],
                line=dict(width=2, color='white')
            ),
            text=_nodes_of_type,
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=_hover_texts,
            name=f'{_ntype}s',
            legendgroup='nodes'
        )
        _rel_node_traces.append(_trace)

    fig_rel_network = go.Figure(data=_rel_edge_traces + _rel_node_traces)

    fig_rel_network.update_layout(
        title=dict(
            text='<b>Formal Relationship Network</b><br><sup>Structural connections between entities (colored by relationship type)</sup>',
            x=0.5
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig_rel_network
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

    Based on the visual analytics executed above, several key insights emerge about the interactions and relationships between vessels and people in Oceanus. The following findings are all derived directly from the data analysis.

    The Communication Network graph reveals that message flow is concentrated around a small number of central entities. The majority of nodes connect only through one or two links, meaning most actors communicate via intermediaries rather than directly. This hub-and-spoke structure immediately narrows Clepper's focus to the most connected nodes they are the gatekeepers of information in Oceanus.

    The Frequency Heatmap (sender = row, receiver = column) shows that many entity pairs communicate in one direction only. Several person-to-vessel pairs show one side consistently sending while the other receives a pattern consistent with operator-to-vessel or supervisor-to-subordinate dynamics. Truly bilateral pairs (dark cells on both sides of the diagonal) represent closer, more collaborative relationships worth distinguishing from purely top-down ones.

    The Formal Relationship Network goes beyond communication frequency and reveals the underlying structural ties between entities. Among the relationship types Colleagues, Operates, Reports, Coordinates, Friends, Unfriendly the presence of Suspicious-labelled edges stands out as direct, data-grounded evidence of flagged connections. Entities that appear in both the suspicious relationship network and the high-activity communication hubs represent the most operationally significant actors for Clepper's investigation.

    The individual profile tool (demonstrated with Nadia Conti as default) allows Clepper to quickly characterize any actor: how many messages they sent vs. received, who their communication partners are, and what formal relationships they hold all in one view. The directional indicators (↔, →, ←) in the relationships table clarify whether Nadia is a peer, a superior, or a subordinate in each connection.

    The daily bar chart shows uneven communication volume across the Oct 1–14, 2040 observation window, with certain days spiking significantly. The hour-by-day-of-week heatmap tightens this further: recurring dark cells at consistent time slots point to deliberately scheduled communication a behavioural signature that distinguishes organized coordination from casual interaction, and a key signal for Clepper to investigate further.
    """)
    return


# =============================================
# QUESTION 2.2 / 3: Community Detection & Topic Modeling
# =============================================


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
    idx = community_labels.index(community_dd.value)
    members = community_list[idx]

    rows = []
    for n in members:
        a = G.nodes[n]
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


# =============================================
# TOPIC MODELING
# =============================================


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
    return topic_model, topics_listm


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
    return


if __name__ == "__main__":
    app.run()
