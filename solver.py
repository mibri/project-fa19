import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

"""
Student Imports
"""
import networkx as nx
from networkx.algorithms import approximation
import matplotlib as plt
import math
import random
import copy

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    ind = { loc:ind for (ind, loc) in enumerate(list_of_locations) }
    list_of_homes_ind = [ind[home] for home in list_of_homes]
    starting_ind = ind[starting_car_location]
    graph, message = adjacency_matrix_to_graph(adjacency_matrix)
    shortest_paths = nx.shortest_path(graph)

    mst = approximation.steinertree.steiner_tree(graph, [starting_ind] + list_of_homes_ind)

    # print("All mst nodes in homes list: " + str(all([home in list(mst.nodes) for home in list_of_homes_ind])))

    nodes = remove_repeats(list(nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(mst, source = ind[starting_car_location])))
    # print("All mst nodes in remove repeats nodes list: " + str(all([home in list(nodes) for home in list_of_homes_ind])))

    # print(nodes)
    nodes = find_path(nodes, graph)
    dropoff_dict = { home:[home] for home in list_of_homes_ind }

    nodes, dropoff_dict = remove_triple(nodes, graph, dropoff_dict)
    return nodes, dropoff_dict


def remove_repeats(nodes):
    """might be a bit inefficient, should fix later"""
    have_seen = set()
    final_nodes = []
    for i in nodes:
        if i not in have_seen:
            have_seen.add(i)
            final_nodes.append(i)
    return final_nodes

def find_path(nodes, graph):
    """returns list of locations and dictionary for drop offs"""
    prev_node = nodes[0]
    final = []
    final.append(prev_node)
    for i in range(1, len(nodes)):
        curr_node = nodes[i]
        if curr_node in graph.neighbors(prev_node):
            final.append(curr_node)
            prev_node = curr_node
            continue
        # nodes = nodes[:i] + nx.shortest_path(graph, prev_node, curr_node)[1:-1] + nodes[i:]
        shortest_path = nx.shortest_path(graph, prev_node, curr_node)[1:]
        final.extend(shortest_path)
        prev_node = shortest_path[-1]
    final.extend(nx.shortest_path(graph, curr_node, nodes[0])[1:])
    # print(final)
    return final

def simulated_annealing(init_path, graph, starting_ind, list_of_homes_ind, shortest_paths, dropoff_dict):
    init_cost = cost_of_solution(graph, init_path, dropoff_dict)
    min_ind = min(list_of_homes_ind)
    max_ind = max(list_of_homes_ind)
    new_path = init_path[:]
    while True: #need to change conditional
        rand_ind = random.randint(min_ind, max_ind)

def remove_triple(nodes, graph, dropoff_dict):
    start_ind = 0
    has_pal = True
    curr_cost_sol = cost_of_solution(graph, nodes, dropoff_dict)
    curr_ind = start_ind
    while True: #might have to change this to be an if condition
        if not has_pal or start_ind + 2 >= len(nodes):
            break
        curr_ind = start_ind
        has_pal = False
        while True:
            if curr_ind + 2 >= len(nodes):
                break
            if nodes[curr_ind] == nodes[curr_ind + 2]:
                has_pal = True
                # if nodes[curr_ind + 1] not in dropoff_dict: #for initial, can remove later when done debugging
                #     print(str(nodes[curr_ind + 1]) + " is not in the dropoff_dict, check this")
                #     curr_ind += 2
                #     break

                test_nodes = nodes[:curr_ind] + nodes[curr_ind + 2:]
                test_dict = copy.deepcopy(dropoff_dict)
                test_dict[nodes[curr_ind]] = test_dict.pop(nodes[curr_ind + 1], []) + test_dict.get(nodes[curr_ind], [])
                new_cost = cost_of_solution(graph, test_nodes, test_dict)

                if new_cost < curr_cost_sol:
                    curr_cost_sol = new_cost
                    dropoff_dict = test_dict
                    nodes = test_nodes
                    curr_ind += 1
                    break
                else:
                    curr_ind += 2
                    start_ind = curr_ind

            else:
                curr_ind += 1
    return nodes, dropoff_dict



def cost_of_solution(G, car_cycle, dropoff_mapping):
    cost = 0
    dropoffs = dropoff_mapping.keys()

    if not car_cycle[0] == car_cycle[-1]:
        print('The start and end vertices are not the same.')
        return

    if len(car_cycle) == 1:
        car_cycle = []
    else:
        car_cycle = get_edges_from_path(car_cycle[:-1]) + [(car_cycle[-2], car_cycle[-1])]
    if len(car_cycle) != 1:
        driving_cost = sum([G.edges[e]['weight'] for e in car_cycle]) * 2 / 3
    else:
        driving_cost = 0
    walking_cost = 0
    shortest = dict(nx.floyd_warshall(G))

    for drop_location in dropoffs:
        for house in dropoff_mapping[drop_location]:
            walking_cost += shortest[drop_location][house]

    cost = driving_cost + walking_cost

    return cost

def get_edges_from_path(path):
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
