import random
import networkx as nx
import pandas as pd

#Train a Kolmogorov–Arnold Network (KAN) from scratch to classify whether a link (edge) exists between two Facebook food pages based on graph features.
#set seed for reproducibility
random.seed(42)
EDGE_LIST_FILE = r"fb-pages-food.edges"

G = nx.read_edgelist(EDGE_LIST_FILE, delimiter=",", nodetype=int)

nodes = list(G.nodes())

#positive samples (node pairs that are connected in the graph)
positive_pairs = list(G.edges())
random.shuffle(positive_pairs)
positive_pairs = positive_pairs[:2000]

#negative samples (node pairs that are not connected aka pairs with NO edge between them)
negative_pairs = set()
while len(negative_pairs) < 2000:
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and not (u, v) in negative_pairs and not (v, u) in negative_pairs:
        negative_pairs.add((u, v))

#label every pair as positive or negative sample
data = []
for u, v in positive_pairs:
    data.append((u, v, 1)) #1 = connected
for u, v in negative_pairs:
    data.append((u, v, 0)) #0 = not connected

#create dataframe
df_pairs = pd.DataFrame(data, columns=['node1', 'node2', 'label'])
df_pairs.to_csv("link_prediction_pair.csv", index = False)
print("Saved 4000 labeled node pairs to csv ")


