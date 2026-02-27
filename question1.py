import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json as json_lib
    import marimo as mo
    from networkx.readwrite import json_graph
    from datetime import datetime
    import pandas as pd
    import altair as alt
    import networkx as nx
    from collections import Counter
    import time
    return alt, datetime, json_graph, json_lib, mo, pd


@app.cell
def _(json_graph, json_lib):
    with open("data/MC3_graph.json", "r") as _f:
        _json_data = json_lib.load(_f)
    G = json_graph.node_link_graph(_json_data, edges="edges")
    return (G,)


@app.cell
def _(G, datetime, pd):
    _comms = []
    for _n, _a in G.nodes(data=True):
        if _a.get("sub_type") == "Communication":
            _ts = datetime.strptime(_a["timestamp"], "%Y-%m-%d %H:%M:%S")
            _comms.append({
                "node_id": _n,
                "timestamp": _ts,
                "hour": _ts.hour,
                "date": _ts.date(),
                "content": _a.get("content", ""),
            })
    df_comms = pd.DataFrame(_comms)
    return (df_comms,)


@app.cell
def _(G, df_comms, pd):
    _comm_details = []
    for _c_node in df_comms["node_id"]:
        _attrs_c = G.nodes[_c_node]
        _ts_c = _attrs_c["timestamp"]
        _content_c = _attrs_c.get("content", "")

        _sender = None
        _receiver = None
        for _pred in G.predecessors(_c_node):
            if G.nodes[_pred].get("type") == "Entity":
                _edge_data = G.edges[_pred, _c_node]
                if _edge_data.get("type") == "sent":
                    _sender = _pred
        for _succ in G.successors(_c_node):
            if G.nodes[_succ].get("type") == "Entity":
                _edge_data = G.edges[_c_node, _succ]
                if _edge_data.get("type") == "received":
                    _receiver = _succ

        if _sender and _receiver:
            _comm_details.append({
                "node_id": _c_node,
                "timestamp": _ts_c,
                "content": _content_c,
                "sender_name": G.nodes[_sender].get("name", _sender),
                "sender_type": G.nodes[_sender].get("sub_type", ""),
                "receiver_name": G.nodes[_receiver].get("name", _receiver),
                "receiver_type": G.nodes[_receiver].get("sub_type", ""),
            })

    df_details = pd.DataFrame(_comm_details)
    df_details["ts"] = pd.to_datetime(df_details["timestamp"])
    df_details["date_str"] = df_details["ts"].dt.date.astype(str)
    df_details["hour_float"] = df_details["ts"].dt.hour + df_details["ts"].dt.minute / 60
    df_details["ts_str"] = df_details["ts"].astype(str)
    return


@app.cell
def _(mo, pd):
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

    _all_entities = ["All"] + sorted(set(
        df_intents["sender_name"].dropna().unique().tolist() +
        df_intents["receiver_name"].dropna().unique().tolist()
    ))
    entity_dropdown = mo.ui.dropdown(options=_all_entities, value="All", label="Filter by Entity")

    suspicion_slider = mo.ui.slider(
        start=0, stop=10, step=1, value=0,
        label="Min. Suspicion",
        show_value=True
    )
    return category_dropdown, entity_dropdown, entity_type_dropdown, suspicion_slider


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

    if entity_dropdown.value != "All":
        _df_filtered = _df_filtered[
            (_df_filtered["sender_name"] == entity_dropdown.value) |
            (_df_filtered["receiver_name"] == entity_dropdown.value)
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
    #container {{ display: flex; width: 100%; height: 1500px; }}
    #timeline-panel {{ width: 62%; height: 100%; overflow-y: auto; border-right: 2px solid #ddd; background: white; }}
    #right-panel {{ width: 38%; height: 100%; display: flex; flex-direction: column; }}
    #chat-panel {{ height: 50%; border-bottom: 2px solid #ddd; display: flex; flex-direction: column; background: white; }}
    #ego-panel {{ height: 50%; background: white; position: relative; }}
    #chat-header {{ padding: 8px 12px; background: #f0f0f0; border-bottom: 1px solid #ddd;
                    font-weight: bold; font-size: 13px; flex-shrink: 0; }}
    #chat-tabs {{ display: flex; gap: 0; border-bottom: 1px solid #ddd; flex-shrink: 0; }}
    .chat-tab {{ padding: 7px 14px; font-size: 12px; cursor: pointer; border: none;
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
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15); display: none; max-width: 350px; z-index: 1000;
    }}
    #ego-title {{ position: absolute; top: 6px; left: 12px; font-size: 13px; font-weight: bold; color: #333; z-index: 10; }}
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
var margin = {{top: 50, right: 20, bottom: 100, left: 70}};
var rowHeight = 80;
var summaryHeight = 80;
var tlWidth = document.getElementById("timeline-panel").offsetWidth - margin.left - margin.right - 10;
if (tlWidth < 400) tlWidth = 620;
var tlHeight = allDates.length * rowHeight;
var totalHeight = tlHeight + summaryHeight;
var dotSize = 10;
var dotGap = 2;
var maxCols = 6;

var svg = d3.select("#chart")
    .append("svg")
    .attr("width", tlWidth + margin.left + margin.right)
    .attr("height", totalHeight + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear().domain([8, 16]).range([0, tlWidth]);
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
    svg.append("text").attr("x", -8).attr("y", y(date) + y.bandwidth() / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .style("font-size", "11px").style("font-weight", "bold").text(i + 1);
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

        g.append("rect").attr("x", px).attr("y", py)
            .attr("width", dotSize / 2).attr("height", dotSize)
            .attr("fill", typeColors[d.sender_type] || "#999");
        g.append("rect").attr("x", px + dotSize / 2).attr("y", py)
            .attr("width", dotSize / 2).attr("height", dotSize)
            .attr("fill", typeColors[d.receiver_type] || "#999");
        g.append("rect").attr("class", "outline")
            .attr("x", px).attr("y", py)
            .attr("width", dotSize).attr("height", dotSize)
            .attr("fill", "none").attr("rx", 1)
            .attr("stroke", d.category_color).attr("stroke-width", 2);

        g.on("mouseover", function(event) {{
            d3.select(this).select(".outline").attr("stroke", "#333").attr("stroke-width", 3);
            tooltip.style("display", "block")
                .html(
                    "<strong>" + d.sender_name + " &rarr; " + d.receiver_name + "</strong><br/>"
                    + d.timestamp + "<br/>"
                    + "<strong>Category:</strong> " + d.category
                    + " <span style='background:" + (categoryColors[d.category]||"#999")
                    + ";color:#fff;padding:1px 5px;border-radius:3px;font-size:10px'>"
                    + d.suspicion + "/10</span><br/>"
                    + "<em>" + d.content.substring(0,120) + "</em>"
                )
                .style("left", (event.clientX + 14) + "px")
                .style("top", (event.clientY - 20) + "px");
        }})
        .on("mouseout", function() {{
            d3.select(this).select(".outline").attr("stroke", d.category_color).attr("stroke-width", 2);
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
svg.append("text").attr("x", -8).attr("y", summaryTop + summaryHeight / 2)
    .attr("text-anchor", "end").attr("dominant-baseline", "middle")
    .style("font-size", "11px").style("font-weight", "bold").text("All");

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
var leg = svg.append("g").attr("transform", "translate(0," + (summaryTop + summaryHeight + 25) + ")");
leg.append("text").attr("x", 0).attr("y", 0).text("Entity Type:").style("font-size", "10px").style("font-weight", "bold");
Object.entries(typeColors).forEach(function(e, i) {{
    leg.append("rect").attr("x", 80 + i * 105).attr("y", -9).attr("width", 12).attr("height", 12).attr("fill", e[1]).attr("rx", 2);
    leg.append("text").attr("x", 80 + i * 105 + 16).attr("y", 0).text(e[0]).style("font-size", "9px");
}});
var cleg = svg.append("g").attr("transform", "translate(0," + (summaryTop + summaryHeight + 48) + ")");
cleg.append("text").attr("x", 0).attr("y", 0).text("Category (border):").style("font-size", "10px").style("font-weight", "bold");
Object.entries(categoryColors).forEach(function(e, i) {{
    var col = i % 4;
    var row = Math.floor(i / 4);
    cleg.append("rect").attr("x", 120 + col * 160).attr("y", -9 + row * 16)
        .attr("width", 10).attr("height", 10).attr("fill", "none").attr("stroke", e[1]).attr("stroke-width", 2);
    cleg.append("text").attr("x", 120 + col * 160 + 14).attr("y", row * 16).text(e[0]).style("font-size", "8px");
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
            item.g.select(".outline").attr("stroke", "#ff0000").attr("stroke-width", 3);
        }} else {{
            item.g.select(".outline").attr("stroke", item.d.category_color).attr("stroke-width", 2);
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

        // Show full content for: clicked message always, conversation tab always, else truncate
        var showFull = isClicked || currentTab === "convo";
        var contentText = showFull
            ? m.content
            : (m.content.length > 150 ? m.content.substring(0, 150) + "\\u2026" : m.content);

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
// RIGHT BOTTOM: EGO NETWORK (circular layout like screenshot)
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

    // Build set of nodes in subnetwork (ego + partners)
    var nodeSet = new Set([entity]);
    entityMsgs.forEach(function(m) {{
        nodeSet.add(m.sender_name);
        nodeSet.add(m.receiver_name);
    }});
    var nodeList = Array.from(nodeSet);

    // Get entity types
    var nodeTypeMap = {{}};
    allData.forEach(function(m) {{
        if (nodeSet.has(m.sender_name)) nodeTypeMap[m.sender_name] = m.sender_type;
        if (nodeSet.has(m.receiver_name)) nodeTypeMap[m.receiver_name] = m.receiver_type;
    }});

    // Build ALL edges between nodes in this subnetwork (not just to ego)
    var edgePairMap = {{}};
    allData.forEach(function(m) {{
        if (nodeSet.has(m.sender_name) && nodeSet.has(m.receiver_name)) {{
            var k = m.sender_name + "||" + m.receiver_name;
            if (!edgePairMap[k]) {{
                edgePairMap[k] = {{ source: m.sender_name, target: m.receiver_name,
                    count: 0, categories: {{}}, maxSusp: 0 }};
            }}
            edgePairMap[k].count++;
            edgePairMap[k].categories[m.category] = (edgePairMap[k].categories[m.category] || 0) + 1;
            if (m.suspicion > edgePairMap[k].maxSusp) edgePairMap[k].maxSusp = m.suspicion;
        }}
    }});
    var edges = Object.values(edgePairMap);

    // Clear old SVG
    var egoSvg = d3.select("#ego-svg");
    egoSvg.selectAll("*").remove();

    var egoEl = document.getElementById("ego-panel");
    var egoW = egoEl.offsetWidth || 400;
    var egoH = egoEl.offsetHeight || 350;
    var egoR = Math.min(egoW, egoH) / 2 - 70;
    var ecx = egoW / 2;
    var ecy = egoH / 2 + 5;

    egoSvg.attr("viewBox", "0 0 " + egoW + " " + egoH);

    // Arrow marker
    var defs = egoSvg.append("defs");
    defs.append("marker").attr("id", "ego-arrow")
        .attr("viewBox", "0 0 10 10").attr("refX", 28).attr("refY", 5)
        .attr("markerWidth", 4).attr("markerHeight", 4).attr("orient", "auto")
        .append("path").attr("d", "M 0 0 L 10 5 L 0 10 Z").attr("fill", "#666");

    // Arrange nodes in a circle — ego at top
    var nNodes = nodeList.length;
    var egoIdx = nodeList.indexOf(entity);
    // Move ego to index 0 so it sits at top
    if (egoIdx > 0) {{
        nodeList.splice(egoIdx, 1);
        nodeList.unshift(entity);
    }}

    var nodePos = {{}};
    nodeList.forEach(function(name, i) {{
        var ang = (2 * Math.PI * i / nNodes) - Math.PI / 2;
        nodePos[name] = {{
            x: ecx + Math.cos(ang) * egoR,
            y: ecy + Math.sin(ang) * egoR,
            ang: ang
        }};
    }});

    // Check for bidirectional edges
    var edgeKeySet = new Set(edges.map(function(e) {{ return e.source + "||" + e.target; }}));

    // Draw edges
    edges.forEach(function(e) {{
        var src = nodePos[e.source];
        var tgt = nodePos[e.target];
        if (!src || !tgt) return;

        var dx = tgt.x - src.x;
        var dy = tgt.y - src.y;
        var dist = Math.sqrt(dx*dx + dy*dy) || 1;

        // Curve offset for bidirectional edges
        var hasBidi = edgeKeySet.has(e.target + "||" + e.source);
        var offset = hasBidi ? 20 : 0;
        var mx = (src.x + tgt.x) / 2 + (-dy / dist) * offset;
        var my = (src.y + tgt.y) / 2 + (dx / dist) * offset;

        var dominantCat = Object.entries(e.categories).sort(function(a,b){{return b[1]-a[1];}})[0][0];
        var edgeColor = categoryColors[dominantCat] || "#999";

        // Highlight edges connected to ego
        var involvesEgo = (e.source === entity || e.target === entity);
        var opacity = involvesEgo ? 0.7 : 0.2;
        var strokeW = involvesEgo
            ? Math.max(1.5, Math.min(e.count * 1.3, 5))
            : Math.max(0.8, Math.min(e.count * 0.8, 3));

        egoSvg.append("path")
            .attr("d", "M" + src.x + "," + src.y + " Q" + mx + "," + my + " " + tgt.x + "," + tgt.y)
            .attr("fill", "none")
            .attr("stroke", edgeColor)
            .attr("stroke-width", strokeW)
            .attr("opacity", opacity)
            .attr("marker-end", "url(#ego-arrow)");
    }});

    // Draw nodes
    var egoTooltip = d3.select("#tooltip");

    nodeList.forEach(function(name) {{
        var pos = nodePos[name];
        var isEgo = name === entity;
        var nType = nodeTypeMap[name] || "";
        var r = isEgo ? 12 : 8;

        var g = egoSvg.append("g")
            .attr("transform", "translate(" + pos.x + "," + pos.y + ")")
            .style("cursor", "pointer");

        // Glow for ego
        if (isEgo) {{
            g.append("circle").attr("r", r + 4)
                .attr("fill", "none").attr("stroke", typeColors[nType] || "#999")
                .attr("stroke-width", 2).attr("opacity", 0.3);
        }}

        g.append("circle").attr("r", r)
            .attr("fill", typeColors[nType] || "#999")
            .attr("stroke", isEgo ? "#333" : "#555")
            .attr("stroke-width", isEgo ? 2.5 : 1);

        // Label
        var anchor = (pos.ang > Math.PI/2 || pos.ang < -Math.PI/2) ? "end" : "start";
        var labelR = r + 8;
        var lx = Math.cos(pos.ang) * labelR;
        var ly = Math.sin(pos.ang) * labelR;

        egoSvg.append("text")
            .attr("x", pos.x + lx).attr("y", pos.y + ly)
            .attr("text-anchor", anchor).attr("dominant-baseline", "middle")
            .style("font-size", isEgo ? "10px" : "9px")
            .style("font-weight", isEgo ? "bold" : "normal")
            .style("fill", isEgo ? "#111" : "#444")
            .text(name.length > 16 ? name.substring(0,16) + "\\u2026" : name);

        // Click to pivot
        g.on("mouseover", function(event) {{
            d3.select(this).select("circle").attr("stroke", "#333").attr("stroke-width", 2.5);
            egoTooltip.style("display", "block")
                .html("<strong>" + name + "</strong><br/>" + nType)
                .style("left", (event.clientX + 12) + "px")
                .style("top", (event.clientY - 20) + "px");
        }})
        .on("mouseout", function() {{
            d3.select(this).select("circle")
                .attr("stroke", isEgo ? "#333" : "#555")
                .attr("stroke-width", isEgo ? 2.5 : 1);
            egoTooltip.style("display", "none");
        }})
        .on("click", function() {{
            var fakeMsg = allData.find(function(m) {{ return m.sender_name === name; }})
                || allData.find(function(m) {{ return m.receiver_name === name; }});
            if (fakeMsg) {{
                var newMsg = Object.assign({{}}, fakeMsg);
                newMsg.sender_name = name;
                onMessageClick(newMsg);
            }}
        }});
    }});

    // Legend
    var eLeg = egoSvg.append("g").attr("transform", "translate(6," + (egoH - 52) + ")");
    eLeg.append("text").attr("x",0).attr("y",0).style("font-size","10px").style("font-weight","bold").text("Entity type:");
    var legendTypes = Object.entries(typeColors);
    legendTypes.forEach(function(e, i) {{
        eLeg.append("circle").attr("cx", 8 + i * 80).attr("cy", 14).attr("r", 5).attr("fill", e[1]);
        eLeg.append("text").attr("x", 16 + i * 80).attr("y", 18).style("font-size", "9px").text(e[0]);
    }});
    eLeg.append("text").attr("x", 0).attr("y", 34).style("font-size", "9px").style("fill", "#888")
        .text("Click a node to explore its network. Bright edges = connected to center.");
}}

}} catch(e) {{
    document.getElementById("chart").innerHTML = "<pre style='color:red'>" + e.message + "\\n" + e.stack + "</pre>";
}}
</script>
</body>
</html>
    """, width="100%", height="1520px")

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
        f"<span style='display:inline-block;margin:1px 3px;padding:2px 7px;border-radius:4px;"
        f"font-size:11px;background:{_category_colors.get(c, '#999')};color:white'>"
        f"{c}: {n}</span>"
        for c, n in _top_cats.items()
    ])

    _n_high = len(_df_filtered[_df_filtered["suspicion"] >= 7])
    _n_entities = len(set(
        _df_filtered["sender_name"].tolist() + _df_filtered["receiver_name"].tolist()
    ))

    _stats_panel = mo.md(f"""
<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
<div style="text-align:center;padding:4px 12px;background:#f5f5f5;border-radius:6px;border:1px solid #e0e0e0">
<div style="font-size:22px;font-weight:bold;color:#333">{_msg_count}</div>
<div style="font-size:10px;color:#888">Messages</div></div>
<div style="text-align:center;padding:4px 12px;background:#f5f5f5;border-radius:6px;border:1px solid #e0e0e0">
<div style="font-size:22px;font-weight:bold;color:#333">{_n_entities}</div>
<div style="font-size:10px;color:#888">Entities</div></div>
<div style="text-align:center;padding:4px 12px;background:#f5f5f5;border-radius:6px;border:1px solid #e0e0e0">
<div style="font-size:22px;font-weight:bold;color:{'#e6194b' if _mean_susp >= 5 else '#f58231' if _mean_susp >= 3 else '#3cb44b'}">{_mean_susp}</div>
<div style="font-size:10px;color:#888">Mean Suspicion</div></div>
<div style="text-align:center;padding:4px 12px;background:#f5f5f5;border-radius:6px;border:1px solid #e0e0e0">
<div style="font-size:22px;font-weight:bold;color:#e6194b">{_n_high}</div>
<div style="font-size:10px;color:#888">High Risk (≥7)</div></div>
<div style="text-align:center;padding:4px 12px;background:#f5f5f5;border-radius:6px;border:1px solid #e0e0e0">
<div style="font-size:14px;font-weight:bold;color:#333">{_max_susp_entity[:16]}</div>
<div style="font-size:10px;color:#888">Most Suspicious (avg {_max_susp_val})</div></div>
</div>
<div style="margin-top:4px">{_cat_html}</div>
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
def _(alt, df_intents, mo):
    _intent_hour = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("hour_float:O", title="Hour of Day"),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("category:N", title="Category"),
    ).properties(title="Category Distribution by Hour", width=700, height=350)
    mo.vstack([mo.md("### When do different categories occur?"), _intent_hour])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()