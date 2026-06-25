"""
Warehouse Assignment DSS - BFS/Dijkstra routing algorithm
Run: source ~/sklearn_env/bin/activate && streamlit run warehouse_assignment_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Warehouse Assignment DSS", page_icon="🏭", layout="wide")

# ALL 39 CITIES WITH VERIFIED GPS COORDINATES
CITY_COORDS = {
    'אור יהודה': (32.0264, 34.8554), 'אילת': (29.5577, 34.9519),
    'אריאל': (32.1047, 35.1735), 'אשדוד': (31.8040, 34.6553),
    'אשקלון': (31.6688, 34.5743), 'באר שבע': (31.2530, 34.7915),
    'גבעתיים': (32.0717, 34.8118), 'גדרה': (31.8145, 34.7796),
    'דימונה': (31.0700, 35.0333), 'הוד השרון': (32.1500, 34.8875),
    'הרצליה': (32.1627, 34.8447), 'חדרה': (32.4340, 34.9196),
    'חולון': (32.0117, 34.7748), 'חיפה': (32.7940, 34.9896),
    'חצור הגלילית': (32.9864, 35.5428), 'טבריה': (32.7922, 35.5312),
    'יבנה': (31.8750, 34.7375), 'יקנעם': (32.6592, 35.1100),
    'ירושלים': (31.7683, 35.2137), 'כפר סבא': (32.1750, 34.9064),
    'כרמיאל': (32.9196, 35.3042), 'לוד': (31.9514, 34.8886),
    'מבשרת ציון': (31.8000, 35.1500), 'מודיעין': (31.8940, 35.0069),
    'מעלות': (33.0167, 35.2750), 'נהריה': (33.0050, 35.0930),
    'נס ציונה': (31.9317, 34.7933), 'נשר': (32.7700, 35.0400),
    'נתניה': (32.3215, 34.8532), 'עפולה': (32.6085, 35.2890),
    'פרדס חנה': (32.4731, 34.9731), 'פתח תקווה': (32.0841, 34.8878),
    'קריית אונו': (32.0633, 34.8553), 'ראשון לציון': (31.9730, 34.7925),
    'רחובות': (31.8928, 34.8113), 'רמלה': (31.9279, 34.8625),
    'רמת גן': (32.0700, 34.8236), 'רעננה': (32.1839, 34.8709),
    'תל אביב': (32.0853, 34.7818),
}

WAREHOUSES = [
    {'WarehouseID': 'W01', 'City': 'תל אביב', 'NumOfEmployees': 26},
    {'WarehouseID': 'W02', 'City': 'חיפה', 'NumOfEmployees': 18},
    {'WarehouseID': 'W03', 'City': 'אשדוד', 'NumOfEmployees': 14},
]

def haversine(c1, c2):
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, [c1[0], c1[1], c2[0], c2[1]])
    dl, dn = la2-la1, lo2-lo1
    a = math.sin(dl/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dn/2)**2
    return R * 2 * math.asin(math.sqrt(a))

@st.cache_resource
def build_graph():
    G = nx.Graph()
    for city, coords in CITY_COORDS.items():
        G.add_node(city, pos=coords)
    cities = list(CITY_COORDS.keys())
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            d = haversine(CITY_COORDS[cities[i]], CITY_COORDS[cities[j]])
            if d < 80:
                G.add_edge(cities[i], cities[j], weight=round(d, 2))
    for n in list(G.nodes()):
        if G.degree(n) == 0:
            nearest = min([c for c in cities if c != n], key=lambda c: haversine(CITY_COORDS[n], CITY_COORDS[c]))
            G.add_edge(n, nearest, weight=round(haversine(CITY_COORDS[n], CITY_COORDS[nearest]), 2))
    return G

def bfs_path(G, start, end):
    if start == end: return [start], 0
    if start not in G or end not in G: return None, float('inf')
    visited, q = {start}, deque([(start, [start])])
    while q:
        node, path = q.popleft()
        for nb in G.neighbors(node):
            if nb == end:
                fp = path + [nb]
                return fp, sum(G[fp[i]][fp[i+1]]['weight'] for i in range(len(fp)-1))
            if nb not in visited:
                visited.add(nb)
                q.append((nb, path + [nb]))
    return None, float('inf')

def dijkstra_path(G, start, end):
    if start == end: return [start], 0
    if start not in G or end not in G: return None, float('inf')
    try:
        path = nx.dijkstra_path(G, start, end, weight='weight')
        return path, nx.dijkstra_path_length(G, start, end, weight='weight')
    except: return None, float('inf')

def assign_warehouse(city, G, algo='BFS'):
    best_wh, best_d, best_p, results = None, float('inf'), None, []
    for wh in WAREHOUSES:
        wc = wh['City']
        path, dist = bfs_path(G, wc, city) if algo == 'BFS' else dijkstra_path(G, wc, city)
        if path is None and city in CITY_COORDS and wc in CITY_COORDS:
            dist = haversine(CITY_COORDS[wc], CITY_COORDS[city])
            path = [wc, city]
        results.append({'מחסן': wh['WarehouseID'], 'עיר מחסן': wc,
                       'מרחק (km)': f"{dist:.1f}" if dist < float('inf') else '-',
                       'מסלול': ' → '.join(path) if path else '-',
                       'קפיצות': len(path)-1 if path else '-'})
        if dist < best_d: best_d, best_wh, best_p = dist, wh['WarehouseID'], path
    return best_wh, best_d, best_p, results

def draw_graph(G, highlight_path=None):
    fig = go.Figure()
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = CITY_COORDS[e[0]][1], CITY_COORDS[e[0]][0]
        x1, y1 = CITY_COORDS[e[1]][1], CITY_COORDS[e[1]][0]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#ccc'), hoverinfo='none', showlegend=False))
    wh_names = [w['City'] for w in WAREHOUSES]
    reg = [n for n in G.nodes() if n not in wh_names]
    fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1] for c in reg], y=[CITY_COORDS[c][0] for c in reg],
        mode='markers+text', text=reg, textposition='top center', textfont=dict(size=8),
        marker=dict(size=8, color='lightblue', line=dict(width=1, color='blue')), showlegend=False, hoverinfo='text',
        hovertext=[f"{c} ({CITY_COORDS[c][0]:.4f}, {CITY_COORDS[c][1]:.4f})" for c in reg]))
    fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1] for c in wh_names], y=[CITY_COORDS[c][0] for c in wh_names],
        mode='markers+text', text=[f"🏭{c}" for c in wh_names], textposition='bottom center',
        textfont=dict(size=10, color='red'), marker=dict(size=16, color='red', symbol='square'), name='מחסנים'))
    if highlight_path and len(highlight_path) > 1:
        px = [CITY_COORDS[c][1] for c in highlight_path if c in CITY_COORDS]
        py = [CITY_COORDS[c][0] for c in highlight_path if c in CITY_COORDS]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(width=5, color='green'),
            marker=dict(size=12, color='green'), name='מסלול'))
    fig.update_layout(height=600, margin=dict(l=0,r=0,t=40,b=0), xaxis_title='Longitude', yaxis_title='Latitude',
        title='גרף 39 ערים - ישראל', showlegend=True)
    return fig

def main():
    st.title("🏭 מערכת שיוך הזמנות למחסנים (DSS)")
    st.caption("BFS/Dijkstra | 39 ערים | Haversine distance")
    st.divider()
    G = build_graph()
    
    st.sidebar.header("⚙️ אלגוריתם")
    algo = st.sidebar.radio("בחר:", ['BFS (מינימום קפיצות)', 'Dijkstra (מינימום km)'])
    algo_type = 'BFS' if 'BFS' in algo else 'Dijkstra'
    st.sidebar.divider()
    st.sidebar.header("🏭 מחסנים")
    st.sidebar.dataframe(pd.DataFrame(WAREHOUSES), hide_index=True)
    st.sidebar.metric("צמתים", G.number_of_nodes())
    st.sidebar.metric("קשתות", G.number_of_edges())
    
    tab1, tab2, tab3 = st.tabs(["📦 שיוך הזמנה", "🗺️ גרף", "📋 מרחקים"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("בחר עיר משלוח")
            city = st.selectbox("עיר יעד:", sorted(CITY_COORDS.keys()))
            if st.button("🔍 חשב שיוך", type="primary", use_container_width=True):
                best_wh, best_d, best_p, results = assign_warehouse(city, G, algo_type)
                if best_wh:
                    wh_city = [w['City'] for w in WAREHOUSES if w['WarehouseID']==best_wh][0]
                    st.success(f"✅ מחסן מומלץ: **{best_wh}** ({wh_city})")
                    st.metric("מרחק", f"{best_d:.1f} km")
                    if best_p: st.info(f"🗺️ {' → '.join(best_p)}")
                    st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
                    st.session_state['path'] = best_p
                else:
                    st.warning("לא נמצא מסלול")
        with col2:
            fig = draw_graph(G, st.session_state.get('path'))
            st.plotly_chart(fig, key="graph_tab1", use_container_width=True)
    
    with tab2:
        st.subheader("🗺️ גרף רשת הערים")
        fig = draw_graph(G)
        st.plotly_chart(fig, key="graph_tab2", use_container_width=True)
        st.markdown(f"**{G.number_of_nodes()}** ערים | **{G.number_of_edges()}** חיבורים | סף חיבור: **80 km**")
    
    with tab3:
        st.subheader("📋 טבלת מרחקים מכל מחסן")
        rows = []
        for city in sorted(CITY_COORDS.keys()):
            row = {'עיר': city}
            for wh in WAREHOUSES:
                d = haversine(CITY_COORDS[wh['City']], CITY_COORDS[city])
                row[f"{wh['WarehouseID']} ({wh['City']})"] = f"{d:.1f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=600)

if __name__ == "__main__":
    main()
