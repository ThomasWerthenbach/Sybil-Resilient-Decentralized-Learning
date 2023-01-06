from collections import defaultdict

# todo turn this into a proper class

import networkx as nx

if __name__ == "__main__":
    sums = defaultdict(dict)
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=0.6)
    G.add_edge(0, 2, weight=0.4)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 1, weight=0.5)
    G.add_edge(2, 3, weight=0.5)
    G.add_edge(3, 0, weight=1)

    decay = 0.01
    start_node = 0

    queue = []
    # print(G.out_edges(0))

    for edge in G.out_edges(start_node):
        weight = (G.get_edge_data(*edge)['weight'])
        queue.append((edge, 1 * weight))
    a = 0
    while queue:
        a += 1
        edge, value = queue.pop(0)
        if edge[0] not in sums[edge[1]]:
            sums[edge[1]][edge[0]] = value
        else:
            sums[edge[1]][edge[0]] += value
        # print(edge, value)
        for e in G.out_edges(edge[1]):
            weight = (G.get_edge_data(*e)['weight'])
            out_value = value * weight
            if out_value > 0.0001 and e[0] != e[1]:
                queue.append((e, out_value * (1 - decay)))

    print(dict(sums))
    # delete self
    sums.pop(start_node)

    total = 0
    for k, v in sums.items():
        total += sum(v.values())

    for k, v in sums.items():
        print(k, sum(v.values())/total)

    print(a)



