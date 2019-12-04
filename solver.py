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

    graph, message = adjacency_matrix_to_graph(adjacency_matrix)
    mst = approximation.steinertree.steiner_tree(graph, [ind[starting_car_location]] + list_of_homes_ind)

    # print("All mst nodes in homes list: " + str(all([home in list(mst.nodes) for home in list_of_homes_ind])))

    nodes = remove_repeats(list(nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(mst, source = ind[starting_car_location])))
    # print("All mst nodes in remove repeats nodes list: " + str(all([home in list(nodes) for home in list_of_homes_ind])))

    # print(nodes)
    nodes = find_path(nodes, graph)
    # print(is_valid_walk(graph, nodes))
    # print(graph.edges)
    # print(nodes)
    return nodes, { home:[home] for home in list_of_homes_ind }

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
