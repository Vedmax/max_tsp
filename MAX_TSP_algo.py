__author__ = 'Maxim Vedernikov'

from MaxWeightMatching import maxWeightMatching
from functools import reduce
import random
import math

TSP_FILES_PATH = 'C:\\Users\\User\\Documents\\Concorde\\Here\\'

class MyGraph:
    def __init__(self):
        self.edges = {}
        self.nodes = {}
        self.nodes_count = 0

def construct_graph(n, i):
    tsp_filename = str(n) + '_' + str(i) + '.txt'
    graph = MyGraph()
    with open(tsp_filename, 'r') as f:
        f.readline()
        for _ in range(n):
            coord = f.readline()[1:-2].split(', ')
            new_node = (float(coord[0]), float(coord[1]))
            for index, node in graph.nodes.items():
                dist = get_dist(node, new_node)
                graph.edges[index, graph.nodes_count + 1] = (index, graph.nodes_count + 1, dist)
            graph.nodes[graph.nodes_count + 1] = new_node
            graph.nodes_count += 1
    return graph

def get_dist(first_node, second_node):
    return round(math.sqrt((first_node[0] - second_node[0])**2 + (first_node[1] - second_node[1])**2))
    # return math.sqrt((first_node[0] - second_node[0])**2 + (first_node[1] - second_node[1])**2)

def get_sorted_match(g, matching):
    sorted_match = []
    for k, i in enumerate(matching):
        if (k, i) in g.edges:
            sorted_match.append(g.edges[(k,i)])
    sorted_match.sort(key=lambda x: -x[2])
    return sorted_match

def cos_alpha(edge1, edge2, g):
    p1 = g.nodes[edge1[0]]
    p2 = g.nodes[edge1[1]]
    q1 = g.nodes[edge2[0]]
    q2 = g.nodes[edge2[1]]
    # print ('p1', p1, 'p2', p2)
    # print ('q1', q1, 'q2', q2)
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    l1 = math.sqrt(v1[0]**2 + v1[1]**2)
    v2 = (q1[0] - q2[0], q1[1] - q2[1])
    l2 = math.sqrt(v2[0]**2 + v2[1]**2)

    cos =  math.fabs((v1[0] * v2[0] + v1[1] * v2[1])) / (l1 * l2)
    # print ('v1', v1, 'v2', v2)
    # print (cos)
    return cos

def is_appropriate_angle(chain1, chain2, g, t):
    # print ((1 - cos_alpha(chain1[0], chain2[0], g)) / 2)
    return (1 - cos_alpha(chain1[len(chain1)-1], chain2[0], g)) / 2 <= (math.pi ** 2) / (4 * (t**2))

def create_alpha_couplings(heavy, g, t):
    alpha_couplings = list(map(lambda x: [x], heavy))
    for _ in range((g.nodes_count//2)- (2 * t) + 3): # for (i = m - t + 2; i >= t; i--)
        for chain1 in alpha_couplings:
            found = False
            for chain2 in alpha_couplings:
                if chain1 == chain2:
                    continue
                if is_appropriate_angle(chain1, chain2, g, t):
                    # print("appropriate")
                    alpha_couplings.remove(chain1)
                    alpha_couplings.remove(chain2)
                    alpha_couplings.append(chain1 + chain2)
                    found = True
                    break
                # else:
                #     print("inappropriate")
            if (found):
                break
    return alpha_couplings

def construct_chain(g, sorted_match):
    t = math.ceil(math.pow(g.nodes_count, 1/3.))
    m = g.nodes_count//2
    heavy = sorted_match[:m-t+2]
    light = sorted_match[m-t+2:]
    alpha_couplings = create_alpha_couplings(heavy, g , t)
    # print(alpha_couplings)
    chain = []
    for alpha_coupling in alpha_couplings:
        chain.append(alpha_coupling)
        if len(light) > 0:
            chain.append([light[0]])
            light.pop(0)
    chain = reduce(lambda x, y: x + y, chain)
    # print (chain)
    return chain

def get_hamiltonian_edges(g, chain):
    unused_nodes = {i for i in range(1, g.nodes_count + 1)}
    edges = []
    for i in range(len(chain) - 1):
        str1 = get_dist(g.nodes[chain[i][0]], g.nodes[chain[i + 1][0]])
        str2 = get_dist(g.nodes[chain[i][1]], g.nodes[chain[i + 1][1]])
        crs1 = get_dist(g.nodes[chain[i][0]], g.nodes[chain[i + 1][1]])
        crs2 = get_dist(g.nodes[chain[i][1]], g.nodes[chain[i + 1][0]])
        if (str1 + str2 > crs1 + crs2):
            edges.append((chain[i][0], chain[i + 1][0]))
            edges.append((chain[i][1], chain[i + 1][1]))
            # print(str1 + str2, crs1 + crs2)
        else:
            edges.append((chain[i][0], chain[i + 1][1]))
            edges.append((chain[i][1], chain[i + 1][0]))
            # print(str1 + str2, crs1 + crs2)
        unused_nodes.difference_update([chain[i][0], chain[i + 1][0], chain[i][1], chain[i + 1][1]])
        # print (g.nodes[chain[i][0]], g.nodes[chain[i][1]])
    edges.append((chain[len(chain) - 1][0], chain[len(chain) - 1][1]))
    if (len(unused_nodes) > 0):
        unused_node = unused_nodes.pop()
        edges.append((chain[0][0], unused_node))
        edges.append((unused_node, chain[0][1]))
    else:
        edges.append((chain[0][0], chain[0][1]))
    # print(edges)
    return edges

def count_hamiltonian_length(hamiltonian_edges, g):
    current_node = 1
    hamiltonian_cycle = [1]
    total_length = 0
    while(len(hamiltonian_edges) > 0):
        for edge in hamiltonian_edges:
            if (edge[0] == current_node or edge[1] == current_node):
                if edge[0] == current_node:
                    new_node = edge[1]
                else:
                    new_node = edge[0]
                hamiltonian_cycle.append(new_node)
                current_node = new_node
                total_length += get_dist(g.nodes[edge[0]], g.nodes[edge[1]])
                hamiltonian_edges.remove(edge)
                break
    print(hamiltonian_cycle)
    return total_length

def write_result(total_length, concorde_ans, n, i):
    p = total_length / concorde_ans
    result = str(n) + '\t' + str(i) + '\t' + str(p) + '\n'
    with open('result.txt', 'a') as f:
        f.write(result)

def print_answer(hamiltonian_edges, g, i):
    total_length = count_hamiltonian_length(hamiltonian_edges, g)
    print('Hamiltonian Weight:  ', total_length)
    concorde_ans = count_concorde_ans(total_length, g, i)
    write_result(total_length, concorde_ans, g.nodes_count, i)

def print_matrix(matrix):
    print('---------------------------------')
    l = len(matrix)
    with open('matrx.txt', 'w') as f:
        for i in range(l):
            for j in range(l):
                f.write(str(matrix[i][j]) + ' ')
            f.write('\n')
    print('---------------------------------')

def get_data_for_concorde(g):
    max_weight = 0
    for edge in g.edges:
        max_weight = max(g.edges[edge][2], max_weight)
    # print ('M', max_weight)
    # print ('NM', max_weight * g.nodes_count)
    matrix = []
    for i in range (1, g.nodes_count + 1):
        matrix.append([])
        for j in range (1, g.nodes_count + 1):
            if (i, j) in g.edges:
                matrix[i - 1].append(max_weight - g.edges[(i, j)][2])
            elif (j, i) in g.edges:
                matrix[i - 1].append(max_weight - g.edges[(j, i)][2])
            else:
                matrix[i - 1].append(0)
    print_matrix(matrix)
    return (matrix, max_weight * g.nodes_count)

def count_path_weight(i, g, matrix):
    path_filename = TSP_FILES_PATH + str(g.nodes_count) + '_' + str(i) + '.txt'
    with open(path_filename, 'r') as f:
        f.readline()
        nodes = f.read().replace('\n', '').split()
        # print (nodes)
    dist = 0
    for i in range(len(nodes)):
        dist += matrix[int(nodes[i])][int(nodes[(i + 1) % g.nodes_count])]
        # print (nodes[i], nodes[(i + 1) % g.nodes_count])
    print('Counted by Concorde distance:', dist)
    return dist

def count_concorde_ans(dist, g, i):
    (lighted_matrix, NM) = get_data_for_concorde(g)
    dist_from_concorde = count_path_weight(i, g, lighted_matrix)
    print ('OPT', NM - dist_from_concorde)
    print('-----------------------------------')
    return NM - dist_from_concorde

def process(n, i):
    g = construct_graph(n, i)
    matching = maxWeightMatching(list(g.edges.values()), True)
    sorted_match = get_sorted_match(g, matching)
    chain = construct_chain(g, sorted_match)
    hamiltonian_edges = get_hamiltonian_edges(g, chain)
    print_answer(hamiltonian_edges, g, i)

if __name__ == '__main__':
    for n in [500, 1000]:
        for i in range(1, 21):
            process(n, i)
