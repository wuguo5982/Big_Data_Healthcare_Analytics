from __future__ import annotations
import pandas as pd
import networkx as nx

def provider_beneficiary_graph(claims: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in claims.iterrows():
        p = "provider:" + str(r["provider_id"])
        b = "beneficiary:" + str(r["beneficiary_id"])
        G.add_node(p, node_type="provider")
        G.add_node(b, node_type="beneficiary")
        if G.has_edge(p,b): G[p][b]["weight"] += 1
        else: G.add_edge(p,b, weight=1)
    return G

def graph_risk_features(claims: pd.DataFrame) -> pd.DataFrame:
    G = provider_beneficiary_graph(claims)
    rows=[]
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "provider":
            provider_id = n.split(":",1)[1]
            degree = G.degree[n]
            weighted = sum(G[n][nbr].get("weight",1) for nbr in G.neighbors(n))
            rows.append({"provider_id": provider_id, "beneficiary_degree": degree, "provider_edge_weight": weighted})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["graph_risk_score"] = (df["provider_edge_weight"] - df["provider_edge_weight"].min()) / (df["provider_edge_weight"].max()-df["provider_edge_weight"].min()+1e-9)
    return df.sort_values("graph_risk_score", ascending=False)
