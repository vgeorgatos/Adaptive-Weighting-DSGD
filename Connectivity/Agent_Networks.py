import numpy as np
import networkx as nx


'Implementations of various network topologies, coded as lists.'
'Including line, ring, 5-regular a hardcoded random and random topology networks implemented via G(n,p) model.'
'Each list contantains num_client elements. Each num_client element contains  a list with the id of node neighbors'


def line_graph(N):


    indeces = [item for item in range(0, N)]

    Graph = []
    first = [indeces[0], indeces[1]]
    Graph.append(first)

    for i in range(1,N-1):
        n = [indeces[i-1], indeces[i], indeces[i+1]]
        Graph.append(n)
    
    #hard code the last element of the list
    last = [indeces[i],indeces[i+1]]
    Graph.append(last)
    return Graph


def ring_graph(N):
    indeces = [item for item in range(0, N)]

    Graph = []
    for i in range(N-1):
        n = [indeces[i-1], indeces[i], indeces[i+1]]
        Graph.append(n)
    
    #hard code the last element of the list
    last = [indeces[i],indeces[i+1],indeces[0]]
    Graph.append(last)
    return Graph



def regular_5_graph(N):
    #N number of nodes
    #k degree of each node
    indeces = [item for item in range(0, N)]

    
    Graph = []
    for i in range(N-2):
        n = [indeces[i-2], indeces[i-1], indeces[i], indeces[i+1], indeces[i+2]]

        Graph.append(n)
    
    #hard code the last k-1 elements of the list: 
    last1 = [indeces[i-1],indeces[i], indeces[i+1],indeces[i+2],indeces[0]]
    Graph.append(last1)
    last1 = [indeces[i], indeces[i+1],indeces[i+2],indeces[0], indeces[1]]
    Graph.append(last1)


    return Graph







def random_20_agents():


    N = []
    N0 = [0,1,2]
    N1 = [0,1,2,3]
    N2 = [0,1,2,3]
    N3 = [1,2,13,4,3]
    N4 = [13, 3, 12, 5, 10,4]
    N5 = [4, 10, 9, 7, 8, 6,5]
    N6 = [5,10,9,7,8,6]
    N7 = [5, 10, 9, 7, 8, 6]
    N8 = [6,7,5,9,8]
    N9 = [5,6,7,8,9,10,11]
    N10 = [4, 5, 6, 7, 9, 10, 11]
    N11 = [9, 10, 11,12, 13]
    N12 = [11, 10, 12, 13, 4, 15, 14]
    N13 = [11, 10, 12, 13, 4,15, 14]
    N14 = [12, 13, 15, 16, 14]
    N15 = [13, 12 ,14, 16, 15 ]
    N16 = [14, 15, 16, 17, 18, 19]
    N17 = [16, 18 , 17, 19]
    N18 = [16,17,18,19]
    N19 = [16,17,18,19]

    N.append(N0)
    N.append(N1)
    N.append(N2)
    N.append(N3)
    N.append(N4)
    N.append(N5)
    N.append(N6)
    N.append(N7)
    N.append(N8)
    N.append(N9)
    N.append(N10)
    N.append(N11)
    N.append(N12)
    N.append(N13)
    N.append(N14)
    N.append(N15)
    N.append(N16)
    N.append(N17)
    N.append(N18)
    N.append(N19)




    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

    G.add_edges_from([(0,1),(0,2)])
    G.add_edges_from([(1,2),(1,3)])
    G.add_edges_from([(2,3)])
    G.add_edges_from([(3,13),(3,4)])
    G.add_edges_from([(4,13),(4,12),(4,5), (4,10)])
    G.add_edges_from([(5,4),(5,10),(5,9),(5,7), (5,8), (5,6)])
    G.add_edges_from([(6,5),(6,10),(6,9),(6,7)])
    G.add_edges_from([(7,5), (7,8),(7,9), (7,6),(7,10)])
    G.add_edges_from([(8,5),(8,6),(8,7),(8,9)])
    G.add_edges_from([(9,5),(9,6),(9,7),(9,8),(9,10), (9,11)])
    G.add_edges_from([(10,4), (10,5), (10,6), (10,7), (10,9), (10,11)])
    G.add_edges_from([(11,9), (11, 10), (11,12), (11,13)])
    G.add_edges_from([(12,11), (12,10), (12,13), (12,4), (12,15), (12,14)])
    G.add_edges_from([(13,11), (13,10), (13,12), (13,3),(13,4) , (13,15) , (13,14)])
    G.add_edges_from([(14,13), (14,12), (14,15) , (14,16)])
    G.add_edges_from([(15,13), (15,12), (15,14), (15,16)])
    G.add_edges_from([(16,14), (16,15), (16,17), (16,18), (16,19)])
    G.add_edges_from([(17,16), (17,18), (17,19) ])
    G.add_edges_from([(18,16), (18,17), (18,19)])
    

    return N,G

def create_random_graphs(n,p,S):
#n number of network nodes
#probability p that edge e_i exists
#S: number of random graphs to create:
    
    TN = []
    si = 0
    while (si < S):


        graph = nx.gnp_random_graph(n,p)
        if nx.is_connected(graph):
            print('Graph is connected')
            E = list(graph.edges)
            V = list(graph.nodes)

            N = [[ ] for _ in range(n)]

            for i in V:
                N[i].append(i)
                for nn in E:
                    if (i == nn[0]):
                        N[i].append(nn[1])
                        N[nn[1]].append(nn[0])

            TN.append(N)    
            si = si + 1
    
    return TN


    
