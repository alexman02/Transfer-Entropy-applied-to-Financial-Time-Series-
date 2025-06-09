import matplotlib.pyplot as plt
import networkx as nx
import pygenstability as pgs
import itertools
from info_theory import MI, entropy
import backboning
import os
import pandas as pd
from scipy.linalg import eigh

def create_network(data, edge="mi", threshold=0.05):

    G = nx.Graph()
    assets = data.columns

    for asset in assets:
        G.add_node(asset)

    edge_weights = {}

    if edge == "mi":
        for (a1, a2) in itertools.combinations(assets, 2):
            mi = MI(data[a1], data[a2])
            G.add_edge(a1, a2, weight=mi)
            edge_weights[(a1, a2)] = mi

    return G

def pgs_communities(graph, min_scale=0.75, max_scale=3, 
                    n_scale=30, log_scale=False, method='leiden'):
    adj_matrix = nx.to_scipy_sparse_array(graph)
    results = pgs.run(
        graph=adj_matrix,
        min_scale=min_scale,            # Adjusted from default -2.0
        max_scale=max_scale,            # Adjusted from default 0.5
        n_scale=n_scale,               
        log_scale=log_scale,                       
        method=method,         
    )
    return results


def louvain_communities(graph):
    louvain_communities = nx.algorithms.community.louvain_communities(graph)

    louvain_community_dict = {}
    for i, community in enumerate(louvain_communities):
        for node in community:
            louvain_community_dict[node] = i
    louvain_labels = [louvain_community_dict[node] for node in graph.nodes()]
    return louvain_labels

def custom_plot_communities(graph, min_scale=0.75 , max_scale=3, 
                            n_scale=30, log_scale=False, method='leiden', 
                            edge_width_constant=5):
    """Plot communities with using pgs and Louvain"""
    pos = nx.spring_layout(graph, seed=21)  
    
    pgs_results = pgs_communities(
        graph, 
        min_scale=min_scale,
        max_scale=max_scale,
        n_scale=n_scale,               
        log_scale=log_scale,                       
        method=method,
        )
    
    for scale_id in pgs_results['selected_partitions']:
        communities = pgs_results['community_id'][scale_id]
        
        plt.figure(figsize=(10,8))
        nx.draw(graph, pos, 
               node_color=communities,
               cmap=plt.cm.tab20,
               with_labels=True,
               width = [graph[a1][a2]['weight'] * edge_width_constant for a1, a2 in graph.edges()]
               )
        
        plt.title(f"Pygenstability Leiden: Scale {pgs_results['scales'][scale_id]:.2f}")

    louvain_labels = louvain_communities(graph)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos,
        node_color=louvain_labels,
        cmap=plt.cm.tab20,
        with_labels=True,
        width=[graph[a1][a2]['weight'] * edge_width_constant for a1, a2 in graph.edges()]
    )
    plt.title("Louvain Community Detection")

def backbone(graph, graph_name, method, threshold):
    # Prepare graph for backboning package
    edges = [(u, v , d["weight"]) for u, v, d in graph.edges(data=True)]
    df_edges = pd.DataFrame(edges, columns=["src", "trg", "weight"])
    df_edges.to_csv(f"{graph_name}_edges.csv", index=False)
    table, _, _ = backboning.read(f".\{graph_name}_edges.csv", 
                                             column_of_interest="weight", 
                                             sep=",", 
                                             undirected=True)
    os.remove(f"{graph_name}_edges.csv")
    # Backbone
    if method == "noise_corrected":
        nc_table = backboning.noise_corrected(table)
        backbone_table = backboning.thresholding(nc_table, threshold=threshold)
    elif method == "disparity_filter":
        dspf_table = backboning.disparity_filter(table)
        backbone_table = backboning.thresholding(dspf_table, threshold=threshold)
    elif method == "naive_threshold":
        backbone_table = table[table["nij"] > threshold]
    

    # Turn table back into graph
    G_backbone = nx.Graph()
    for _, row in backbone_table.iterrows():
        G_backbone.add_edge(row["src"], row["trg"], weight=row["nij"])

    return G_backbone

def plot_graph(graph, pos, title, edge_width_constant=5):
    plt.figure(figsize=(12, 8))
    edges = graph.edges(data=True)
    weights = [w["weight"] for _, _, w in edges]

    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos, width=[w * 5 for w in weights], alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    
    plt.title(title)
    plt.show()

def create_true_labels(graph, ticker_dict):
    """Map graph nodes to true asset class labels"""
    true_mapping = {}
    for asset_class, symbols in ticker_dict.items():
        for symbol in symbols:
            true_mapping[symbol] = asset_class
    
    classes = list(ticker_dict.keys())
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    return [class_to_id[true_mapping[node]] for node in graph.nodes()]


def nvi(true_labels, detected_labels):

    mi = MI(true_labels, detected_labels)

    h_true = entropy(true_labels)
    h_detected = entropy(detected_labels)

    vi = h_true + h_detected - 2 * mi

    return vi / (h_true + h_detected - mi) if (h_true + h_detected - mi) > 0 else 0

def pgs_optimal_param(graph, true_labels, min_scales):
    nvis = []
    for min_scale in min_scales:
        pgs_results = pgs_communities(graph, 
                                      min_scale=min_scale, 
                                      max_scale=2.5, 
                                      n_scale=20)
        first_scale = pgs_results['selected_partitions'][0]
        pgs_community = pgs_results['community_id'][first_scale]
        print(nvi(true_labels, pgs_community))
        nvis.append(nvi(true_labels, pgs_community))
    return nvis

def network_properties(G):

    analysis = {}

    analysis['n_nodes'] = G.number_of_nodes()
    analysis['n_edges'] = G.number_of_edges()

    degrees = dict(G.degree(weight="weight"))
    analysis['degree'] = pd.Series(degrees)
    analysis['e_vector_centr'] = nx.eigenvector_centrality(G, weight="weight")

    analysis['avg_clustering'] = nx.average_clustering(G, weight="weight")
    
    analysis['avg_shortest_path_length'] = nx.average_shortest_path_length(G)

    L = nx.laplacian_matrix(G).toarray()
    e_vals = eigh(L, eigvals_only=True)
    analysis['alg_connectivity'] = e_vals[1]

    return pd.DataFrame(analysis)
