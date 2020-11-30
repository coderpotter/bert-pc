import cnfgen
import networkx as nx
from random import uniform

# SAT is class 1, UNSAT is class 0
dataset = open("./graph-dataset/graph-dataset.txt",'w+')
# write dataset size
dataset.writelines("1000\n")
total_sat = 0
total_unsat = 0

for i in range(0,1000):
    # write each problem
    p = uniform(0.0,1.0)
    G = nx.erdos_renyi_graph(100,p)
    cnf = cnfgen.families.coloring.GraphColoringFormula(G,4)
    sat = cnf.is_satisfiable()[0]
    if sat == False:
        sat = 0
        total_unsat += 1
    else:
        sat = 1
        total_sat += 1
    
    dataset.writelines(f"100 {sat}\n")
    for node in G.nodes():
        line = "0"
        edges = G.edges(node)
        # append the number of outgoing edges for this node
        line += f" {len(edges)}"
        for e in edges:
            # append all of the outgoing node ids
            line += f" {e[1]}"
        
        dataset.writelines(line+"\n")

    # file = open(f'./dataset/formula{i}_{sat}.dimacs','w+')
    # file.writelines(formula.dimacs())
    # file.close()

print(f"total SAT: {total_sat}")
print(f"total UNSAT: {total_unsat}")