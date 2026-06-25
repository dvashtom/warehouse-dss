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

# RTL CSS
st.markdown("""<style>
.stApp {direction:rtl}
.stMarkdown,h1,h2,h3,p,label{direction:rtl;text-align:right}
.stDataFrame,.stPlotlyChart{direction:ltr}
[data-testid="stMetric"]{direction:rtl;text-align:center}
[data-testid="stMetricValue"]{direction:ltr}
.block-container{padding-top:1rem}
</style>""", unsafe_allow_html=True)

CITY_COORDS = {
    'אור יהודה':(32.0264,34.8554),'אילת':(29.5577,34.9519),'אריאל':(32.1047,35.1735),
    'אשדוד':(31.8040,34.6553),'אשקלון':(31.6688,34.5743),'באר שבע':(31.2530,34.7915),
    'גבעתיים':(32.0717,34.8118),'גדרה':(31.8145,34.7796),'דימונה':(31.0700,35.0333),
    'הוד השרון':(32.1500,34.8875),'הרצליה':(32.1627,34.8447),'חדרה':(32.4340,34.9196),
    'חולון':(32.0117,34.7748),'חיפה':(32.7940,34.9896),'חצור הגלילית':(32.9864,35.5428),
    'טבריה':(32.7922,35.5312),'יבנה':(31.8750,34.7375),'יקנעם':(32.6592,35.1100),
    'ירושלים':(31.7683,35.2137),'כפר סבא':(32.1750,34.9064),'כרמיאל':(32.9196,35.3042),
    'לוד':(31.9514,34.8886),'מבשרת ציון':(31.8000,35.1500),'מודיעין':(31.8940,35.0069),
    'מעלות':(33.0167,35.2750),'נהריה':(33.0050,35.0930),'נס ציונה':(31.9317,34.7933),
    'נשר':(32.7700,35.0400),'נתניה':(32.3215,34.8532),'עפולה':(32.6085,35.2890),
    'פרדס חנה':(32.4731,34.9731),'פתח תקווה':(32.0841,34.8878),'קריית אונו':(32.0633,34.8553),
    'ראשון לציון':(31.9730,34.7925),'רחובות':(31.8928,34.8113),'רמלה':(31.9279,34.8625),
    'רמת גן':(32.0700,34.8236),'רעננה':(32.1839,34.8709),'תל אביב':(32.0853,34.7818),
}
WAREHOUSES = [
    {'id':'W01','city':'תל אביב','emp':26,'color':'#636EFA'},
    {'id':'W02','city':'חיפה','emp':18,'color':'#EF553B'},
    {'id':'W03','city':'אשדוד','emp':14,'color':'#00CC96'},
]

# Orders per city (from star_schema.xlsx ORDERS table)
ORDERS_PER_CITY = {
    'תל אביב': 283, 'באר שבע': 189, 'הרצליה': 172, 'ראשון לציון': 169,
    'אשדוד': 153, 'ירושלים': 152, 'חיפה': 123, 'נתניה': 117,
    'פתח תקווה': 113, 'רעננה': 81, 'רמת גן': 76, 'חולון': 74,
    'מודיעין': 72, 'נהריה': 56, 'כפר סבא': 54, 'גבעתיים': 52,
    'חצור הגלילית': 51, 'חדרה': 50, 'אילת': 42, 'קריית אונו': 40,
    'לוד': 38, 'רחובות': 32, 'אשקלון': 24, 'דימונה': 23,
    'עפולה': 20, 'מבשרת ציון': 20, 'גדרה': 17, 'כרמיאל': 15,
    'רמלה': 14, 'אור יהודה': 12, 'יקנעם': 11, 'נשר': 11,
    'יבנה': 10, 'טבריה': 10, 'מעלות': 9, 'פרדס חנה': 8,
    'אריאל': 3, 'הוד השרון': 2, 'נס ציונה': 2,
}

def haversine(c1,c2):
    R=6371
    la1,lo1,la2,lo2=map(math.radians,[c1[0],c1[1],c2[0],c2[1]])
    dl,dn=la2-la1,lo2-lo1
    a=math.sin(dl/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dn/2)**2
    return R*2*math.asin(math.sqrt(a))

@st.cache_resource
def build_graph():
    G=nx.Graph()
    for city,coords in CITY_COORDS.items(): G.add_node(city,pos=coords)
    cities=list(CITY_COORDS.keys())
    for i in range(len(cities)):
        for j in range(i+1,len(cities)):
            d=haversine(CITY_COORDS[cities[i]],CITY_COORDS[cities[j]])
            if d<80: G.add_edge(cities[i],cities[j],weight=round(d,2))
    for n in list(G.nodes()):
        if G.degree(n)==0:
            nearest=min([c for c in cities if c!=n],key=lambda c:haversine(CITY_COORDS[n],CITY_COORDS[c]))
            G.add_edge(n,nearest,weight=round(haversine(CITY_COORDS[n],CITY_COORDS[nearest]),2))
    return G

def bfs_path(G,start,end):
    if start==end: return [start],0
    if start not in G or end not in G: return None,float('inf')
    visited,q={start},deque([(start,[start])])
    while q:
        node,path=q.popleft()
        for nb in G.neighbors(node):
            if nb==end:
                fp=path+[nb]
                return fp,sum(G[fp[i]][fp[i+1]]['weight'] for i in range(len(fp)-1))
            if nb not in visited: visited.add(nb); q.append((nb,path+[nb]))
    return None,float('inf')

def dijkstra_path(G,start,end):
    if start==end: return [start],0
    if start not in G or end not in G: return None,float('inf')
    try:
        p=nx.dijkstra_path(G,start,end,weight='weight')
        return p,nx.dijkstra_path_length(G,start,end,weight='weight')
    except: return None,float('inf')

def assign_wh(city,G,algo='BFS'):
    best_wh,best_d,best_p,res=None,float('inf'),None,[]
    for wh in WAREHOUSES:
        wc=wh['city']
        path,dist=bfs_path(G,wc,city) if algo=='BFS' else dijkstra_path(G,wc,city)
        if path is None: dist=haversine(CITY_COORDS[wc],CITY_COORDS[city]); path=[wc,city]
        res.append({'מחסן':wh['id'],'עיר':wc,'מרחק':round(dist,1),'מסלול':' ← '.join(reversed(path)),'קפיצות':len(path)-1})
        if dist<best_d: best_d,best_wh,best_p=dist,wh['id'],path
    return best_wh,best_d,best_p,res

@st.cache_data
def compute_kpis():
    G=build_graph()
    dists,assigns=[],{w['id']:0 for w in WAREHOUSES}
    far_city,far_d='',0
    for city in CITY_COORDS:
        wh,d,_,_=assign_wh(city,G,'Dijkstra')
        dists.append(d); assigns[wh]+=1
        if d>far_d: far_d,far_city=d,city
    avg=sum(dists)/len(dists)
    busy=max(assigns,key=assigns.get)
    return avg,busy,assigns[busy],far_city,far_d,assigns

def draw_map(G,dij_path=None,bfs_p=None):
    fig=go.Figure()
    ex,ey=[],[]
    for e in G.edges():
        x0,y0=CITY_COORDS[e[0]][1],CITY_COORDS[e[0]][0]
        x1,y1=CITY_COORDS[e[1]][1],CITY_COORDS[e[1]][0]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]
    fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(width=0.4,color='rgba(180,180,180,0.4)'),hoverinfo='none',showlegend=False))
    wn=[w['city'] for w in WAREHOUSES]
    reg=[n for n in G.nodes() if n not in wn]
    fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1] for c in reg],y=[CITY_COORDS[c][0] for c in reg],
        mode='markers+text',text=reg,textposition='top center',textfont=dict(size=7,color='#555'),
        marker=dict(size=7,color='#a8d5e2',line=dict(width=1,color='#2c7fb8')),showlegend=False,hoverinfo='text',
        hovertext=[f"{c} ({CITY_COORDS[c][0]:.3f}N)" for c in reg]))
    for wh in WAREHOUSES:
        c=wh['city']
        fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1]],y=[CITY_COORDS[c][0]],
            mode='markers+text',text=[f"🏭 {wh['id']}"],textposition='bottom center',
            textfont=dict(size=11,color=wh['color']),
            marker=dict(size=18,color=wh['color'],symbol='square',line=dict(width=2,color='black')),
            name=f"{wh['id']}-{c}"))
    if dij_path and len(dij_path)>1:
        fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1] for c in dij_path],y=[CITY_COORDS[c][0] for c in dij_path],
            mode='lines+markers',line=dict(width=5,color='#00CC96'),marker=dict(size=12,color='#00CC96'),name='Dijkstra'))
    if bfs_p and len(bfs_p)>1:
        fig.add_trace(go.Scatter(x=[CITY_COORDS[c][1] for c in bfs_p],y=[CITY_COORDS[c][0] for c in bfs_p],
            mode='lines+markers',line=dict(width=4,color='#FFA500',dash='dash'),marker=dict(size=10,color='#FFA500',symbol='diamond'),name='BFS'))
    fig.update_layout(height=620,margin=dict(l=0,r=0,t=30,b=0),
        xaxis=dict(title='Longitude',range=[34.2,35.7],gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title='Latitude',range=[29.3,33.3],gridcolor='rgba(0,0,0,0.05)'),
        title='מפת ישראל - 39 ערים, 3 מחסנים',legend=dict(x=0.01,y=0.99,bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='rgba(240,248,255,0.3)')
    return fig

def main():
    st.title("🏭 מערכת שיוך הזמנות למחסנים")
    st.caption("DSS | BFS & Dijkstra | 39 ערים | Haversine")
    G=build_graph()
    avg,busy,busy_n,far_city,far_d,assigns=compute_kpis()
    busy_c=[w['city'] for w in WAREHOUSES if w['id']==busy][0]
    k1,k2,k3,k4=st.columns(4)
    k1.metric("📏 מרחק ממוצע",f"{avg:.1f} km")
    k2.metric("🏭 מחסן עמוס",f"{busy} ({busy_c})",f"{busy_n} ערים")
    k3.metric("🌍 עיר רחוקה",far_city,f"{far_d:.1f} km")
    k4.metric("🔗 גרף",f"{G.number_of_nodes()} צמתים",f"{G.number_of_edges()} קשתות")
    st.divider()
    

    st.sidebar.header("🏭 מחסנים")
    st.sidebar.dataframe(pd.DataFrame([{'מזהה':w['id'],'עיר':w['city'],'עובדים':w['emp']} for w in WAREHOUSES]),hide_index=True)
    st.sidebar.divider()
    st.sidebar.markdown("### מתי BFS עדיף?")
    st.sidebar.markdown("• כשרוצים **מינימום עצירות** (תחנות ביניים)")
    st.sidebar.markdown("• מתאים למערכות שבהן כל עצירה = עלות קבועה")
    st.sidebar.markdown("### מתי Dijkstra עדיף?")
    st.sidebar.markdown("• כשרוצים **מינימום מרחק** בפועל")
    st.sidebar.markdown("• מתאים כשעלות המשלוח תלויה בק\"מ")
    
    tab1,tab2,tab3=st.tabs(["📦 שיוך הזמנה","⚖️ השוואת אלגוריתמים","📋 טבלת מרחקים"])
    
    with tab1:
        c1,c2=st.columns([1,2])
        with c1:
            city=st.selectbox("עיר יעד:",sorted(CITY_COORDS.keys()))
            if st.button("🔍 חשב שיוך",type="primary",use_container_width=True):
                wh_d,d_d,p_d,res_d=assign_wh(city,G,"Dijkstra")
                wh_b,d_b,p_b,res_b=assign_wh(city,G,"BFS")
                wh_c=[w["city"] for w in WAREHOUSES if w["id"]==wh_d][0]
                st.success(f"✅ מחסן (Dijkstra): **{wh_d}** ({wh_c}) - {d_d:.1f} km")
                st.info(f"✅ מחסן (BFS): **{wh_b}** - {d_b:.1f} km")
                if p_d: st.markdown(f"🟢 Dijkstra: {" ← ".join(reversed(p_d))}")
                if p_b and p_b!=p_d: st.markdown(f"🟠 BFS: {" ← ".join(reversed(p_b))}")
                st.dataframe(pd.DataFrame(res_d),hide_index=True,use_container_width=True)
                st.session_state["dij_p"]=p_d
                st.session_state["bfs_p"]=p_b
        with c2:
            fig=draw_map(G,st.session_state.get('dij_p'),st.session_state.get('bfs_p'))
            st.plotly_chart(fig,key="map1",use_container_width=True)
    
    with tab2:
        st.subheader("⚖️ השוואה ויזואלית: BFS vs Dijkstra")
        st.markdown("""
        | | BFS | Dijkstra |
        |--|-----|----------|
        | **מטרה** | מינימום קפיצות (hops) | מינימום ק"מ |
        | **מתאים ל** | עלות קבועה לכל עצירה | עלות פר ק"מ |
        | **יתרון** | פשוט, מהיר | אופטימלי למרחק |
        | **חיסרון** | מתעלם ממרחק | דורש משקלות |
        """)
        st.divider()
        # Comparison chart
        rows=[]
        for city in sorted(CITY_COORDS.keys()):
            _,d_bfs,p_bfs,_=assign_wh(city,G,'BFS')
            _,d_dij,p_dij,_=assign_wh(city,G,'Dijkstra')
            rows.append({'עיר':city,'BFS (km)':round(d_bfs,1),'Dijkstra (km)':round(d_dij,1),'הפרש':round(d_bfs-d_dij,1)})
        df=pd.DataFrame(rows)
        diff_df=df[df['הפרש']!=0]
        if len(diff_df)>0:
            st.markdown(f"### ערים עם הבדל ({len(diff_df)} מתוך {len(df)}):")
            st.dataframe(diff_df,hide_index=True,use_container_width=True)
            fig=go.Figure()
            fig.add_trace(go.Bar(x=diff_df['עיר'],y=diff_df['BFS (km)'],name='BFS',marker_color='#FFA500'))
            fig.add_trace(go.Bar(x=diff_df['עיר'],y=diff_df['Dijkstra (km)'],name='Dijkstra',marker_color='#00CC96'))
            fig.update_layout(barmode='group',title='השוואת מרחקים: BFS vs Dijkstra',yaxis_title='מרחק (km)',height=350)
            st.plotly_chart(fig,key="comp_chart",use_container_width=True)
        else:
            st.success("✅ שני האלגוריתמים נותנים תוצאות זהות לכל הערים!")
        st.markdown("### 📊 עומס הזמנות לפי מחסן")
        wh_load={w["id"]:0 for w in WAREHOUSES}
        city_to_wh={}
        for city in ORDERS_PER_CITY:
            wh_id,_,_,_=assign_wh(city,G,"Dijkstra")
            wh_load[wh_id]+=ORDERS_PER_CITY[city]
            city_to_wh[city]=wh_id
        fig_load=go.Figure()
        colors=[w["color"] for w in WAREHOUSES]
        wh_ids=[w["id"] for w in WAREHOUSES]
        wh_labels=[f"{w["id"]} ({w["city"]})" for w in WAREHOUSES]
        fig_load.add_trace(go.Bar(x=wh_labels,y=[wh_load[w] for w in wh_ids],marker_color=colors,text=[wh_load[w] for w in wh_ids],textposition="outside"))
        fig_load.update_layout(title="עומס הזמנות לפי מחסן (2,400 הזמנות)",yaxis_title="מספר הזמנות",height=350)
        st.plotly_chart(fig_load,key="load_chart",use_container_width=True)
        st.markdown("### 🏙️ ערים לפי מחסן משויך")
        for wh in WAREHOUSES:
            cities_for_wh=[(c,ORDERS_PER_CITY[c]) for c in city_to_wh if city_to_wh[c]==wh["id"]]
            cities_for_wh.sort(key=lambda x:-x[1])
            total=sum(o for _,o in cities_for_wh)
            st.markdown(f"**{wh["id"]} ({wh["city"]})** - {total} הזמנות מ-{len(cities_for_wh)} ערים:")
            st.caption(", ".join([f"{c} ({n})" for c,n in cities_for_wh]))
    
    with tab3:
        st.subheader("📋 טבלת מרחקים ישירים (Haversine) מכל מחסן")
        rows=[]
        for city in sorted(CITY_COORDS.keys()):
            row={'עיר':city}
            for wh in WAREHOUSES:
                d=haversine(CITY_COORDS[wh['city']],CITY_COORDS[city])
                row[f"{wh['id']} ({wh['city']})"]=f"{d:.1f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=600)

if __name__=="__main__":
    main()
