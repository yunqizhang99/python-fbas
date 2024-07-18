import json
import os
from requests import get
from platformdirs import user_cache_dir
import stellarbeat
import pytest
from fbas import *
from stellarbeat import get_validators_from_file
from test_utils import get_test_data_file_path
import networkx as nx
import matplotlib.pyplot as plt

fbas_list = stellarbeat._fetch_with_fake_nodes()

fbas = FBAS.from_stellarbeat_json(fbas_list)

# print(fbas)

G = fbas.to_graph()

G_color_map = ['green' if isinstance(node, str) else 'red' for node in G]  

count_validators = 0
G_relabel_mapping = {}

count_validators = 0
count_qsets = -1
for node in G:
	if isinstance(node, str):
		G_relabel_mapping[node] = count_validators
		count_validators += 1
	else:
		G_relabel_mapping[node] = count_qsets
		count_qsets -= 1

print(G_relabel_mapping)
G = nx.relabel_nodes(G, G_relabel_mapping)

f1 = plt.figure(1)

nx.draw(G, node_color=G_color_map, with_labels=True)
#Now only add labels to the nodes you require (the hubs in my case)

ppr_vals = nx.pagerank(G)
ppr_vals_personal = nx.pagerank(G, personalization={0: 1})
# print("________")
ppr_vals = { k:v for k,v in ppr_vals.items() if k>=0 }
ppr_vals_personal = { k:v for k,v in ppr_vals_personal.items() if k>=0 }
print("______________")
print(ppr_vals)
print("______________")
print(ppr_vals_personal)
print("______________")

f2 = plt.figure(2)
plt.bar(ppr_vals.keys(), ppr_vals.values(), color='g')
f3 = plt.figure(3)
plt.bar(ppr_vals_personal.keys(), ppr_vals_personal.values(), color='b')
plt.show()
# print("_________________")
# print(G.nodes)