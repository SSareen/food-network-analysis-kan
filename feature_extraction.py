##extract features from the dataset

import pandas as pd
import networkx as nx
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    preferential_attachment,
)
from networkx import common_neighbors
import math
import seaborn as sns
import matplotlib.pyplot as plt

df_pairs = pd.read_csv("link_prediction_pair.csv")
EDGE_LIST_FILE = "fb-pages-food.edges"
G = nx.read_edgelist(EDGE_LIST_FILE, delimiter=",", nodetype=int)
degrees = dict(G.degree())

#list of features
features = []

#pair set
pair_set = set(zip(df_pairs["node1"], df_pairs["node2"]))

#Calculating all link prediction scores in our dataset

#Jaccard Coefficient: Fraction of shared neighbors out of all neighbors combined
jaccard = { (u, v): p for u,v,p in jaccard_coefficient(G, pair_set)}

#Adamic-Adar Index: Shared neighbor with few connections is stronger sign of potential link that one connected to everyone
def adamic_adar(u, v):
    total = 0.0
    for w in common_neighbors(G, u, v):
        dw = degrees.get(w, 0)
        if dw > 1: #skip if degree <= 1 (to avoid 1/log(1))
            total += 1.0 / math.log(dw)
    return total

#Preferential Attachment Score: Product of degrees; high-degree nodes are more likely to connect to other high degree nodes
pa_score= { (u,v): p for u,v,p in preferential_attachment(G, pair_set) }

#Loop through each pair and compute features
for row in df_pairs.itertuples(index=False):
    u, v, label = row.node1, row.node2, row.label
    
    #degree of each node
    deg_u = degrees.get(u, 0)
    deg_v = degrees.get(v, 0)

    #count of shared neighbors between two nodes
    num_common = len(list(common_neighbors(G, u, v)))

    jacc = jaccard.get((u, v), jaccard.get((v, u), 0))
    aa = adamic_adar(u, v)
    pa = pa_score.get((u, v), pa_score.get((v, u), 0))

    features.append([u, v, deg_u, deg_v, num_common, jacc, aa, pa, label])

df_features = pd.DataFrame(features, columns = [ "node1", "node2", "deg_u", "deg_v", "common_neighbors",
    "jaccard", "adamic_adar", "pref_attachment", "label"])
df_features.to_csv("link_prediction_features.csv", index = False)

print("Features Succesfully Extracted.")
import seaborn as sns
import matplotlib.pyplot as plt

#select only numeric features + label
selected_cols = ["common_neighbors", "jaccard", "adamic_adar", "label"]

#create pairplot
sns.pairplot(
    df_features[selected_cols],
    hue="label", 
    diag_kind="kde",
    plot_kws={'alpha': 0.5, 's': 20}
)

plt.suptitle("Key Feature Relationships for Link Prediction", y=1.02)
plt.show()
