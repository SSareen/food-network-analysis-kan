import networkx as nx
import matplotlib.pyplot as plt

EDGE_LIST_FILE = r"fb-pages-food.edges"


#Load graph
G = nx.read_edgelist(EDGE_LIST_FILE, delimiter=",", nodetype=int)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

#visualize a subgraph of first 50 nodes
sub_nodes = list(G.nodes())[:50]
subgraph = G.subgraph(sub_nodes)

plt.figure(figsize=(10, 8))
nx.draw(subgraph, node_size=50, with_labels=False, edge_color="gray")
plt.title("Subgraph of fb-pages-food (50 nodes)")
plt.axis("off")
plt.tight_layout()
plt.show()
