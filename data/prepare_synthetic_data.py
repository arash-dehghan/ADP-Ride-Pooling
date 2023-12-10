import sys
sys.dont_write_bytecode = True  # Prevents Python from generating .pyc files
import os
from datetime import datetime
import geopy.distance  # Library for calculating geodesic distances
import argparse  # Module for parsing command line arguments
import numpy  # Fundamental package for scientific computing
import pandas as pd  # Data analysis and manipulation tool
import networkx as nx  # Library for creating and manipulating complex networks
from collections import Counter
import matplotlib.pyplot as plt  # Plotting library
from scipy.stats import poisson, norm  # Statistical functions
from pylab import arange, exp
from scipy.optimize import curve_fit  # Optimization and root finding
import community.community_louvain as community_louvain  # Community detection in networks
import plotly.express as px  # Plotting library for interactive plots
from copy import deepcopy  # For creating deep copies of objects
from DataGenerator import DataGenerator  # Custom module for data generation
import pickle  # Module for object serialization
from copy import copy, deepcopy  # For creating copies of objects

class DataPreperation(object):
    def __init__(self, initial_vars):
        """
        Initializes the DataPreperation object with given variables.

        Sets various attributes based on the initial_vars dictionary, prepares the graph (network),
        and computes essential information for data analysis such as passenger distribution,
        request distribution, and request information.

        Args:
        initial_vars (dict): Dictionary containing initialization variables.
        """
        # Setting attributes based on initial_vars
        for key in initial_vars:
            setattr(self, key, initial_vars[key])

        # Base percentage for edge removal in the graph
        self.base_edge_removal_percentage = 0.6 if self.graph_directed else 0.4

        # Loading or creating cleaned data for analysis
        self.df = pd.read_pickle(f'synthetic_dataset/CleanedChicagoData.pickle') if os.path.isfile(f'synthetic_dataset/CleanedChicagoData.pickle') else self.create_cleaned_data()

        # Creating a grid-based graph structure
        self.G = self.make_grid()

        # List of nodes in the graph
        self.nodes_list = list(self.G.nodes)

        # Setting zones for analysis
        self.zones = self.set_zones()

        # Calculating passenger distribution
        self.passenger_distribution = self.get_passenger_distribution()

        # Calculating request distribution
        self.request_distribution = self.get_request_distribution()

        # Gathering request information
        self.get_request_information()

	def create_cleaned_data(self):
	    """
	    Cleans and preprocesses the original dataset.

	    This function reads the original dataset, selects relevant columns, handles missing values,
	    processes datetime information, and calculates trip distances. The cleaned data is then
	    saved as a pickle file for future use.

	    Returns:
	    DataFrame: The cleaned and preprocessed dataset.
	    """
	    # Reading the original dataset
	    df = pd.read_csv(f'synthetic_dataset/OriginalChicagoData.csv')
	    # Selecting relevant columns
	    df = df[['Trip Start Timestamp', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 
	             'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']]
	    # Dropping rows with missing values
	    df = df.dropna().reset_index(drop=True)
	    # Converting string timestamps to datetime objects
	    df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'])
	    # Extracting time and date as separate columns
	    df['Time'] = df['Trip Start Timestamp'].apply(lambda x: x.time())
	    df['Date'] = df['Trip Start Timestamp'].apply(lambda x: x.date())
	    # Setting pickup and dropoff coordinates
	    df['Pickup Coordinates'] = df.apply(lambda x: (x['Pickup Centroid Latitude'], x['Pickup Centroid Longitude']), axis=1)
	    df['Dropoff Coordinates'] = df.apply(lambda x: (x['Dropoff Centroid Latitude'], x['Dropoff Centroid Longitude']), axis=1)
	    df['Trip Coordinates'] = df.apply(lambda x: (x['Pickup Coordinates'], x['Dropoff Coordinates']), axis=1)
	    # Dropping redundant columns
	    df = df.drop(['Trip Start Timestamp', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 
	                  'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude'], axis=1)
	    # Removing rows where pickup and dropoff coordinates are the same
	    df = df[df['Pickup Coordinates'] != df['Dropoff Coordinates']]
	    # Calculating trip distance
	    df['Trip Distance'] = df.apply(lambda x: geopy.distance.geodesic(x['Pickup Coordinates'], x['Dropoff Coordinates']).km, axis=1)
	    # Saving the cleaned data as a pickle file
	    df.to_pickle(f'CleanedChicagoData.pickle')
	    return df

	def make_grid(self):
	    """
	    Creates a grid-based graph with added complexity.

	    This function generates a grid graph, labels the nodes, and applies modifications
	    such as removing edges and adding weights. It also generates files for next locations
	    and travel times if they do not exist, and loads them into memory.

	    Returns:
	    Graph: The modified grid-based graph.
	    """
	    # Creating and labeling the grid graph
	    G = nx.grid_graph(dim=self.dims)
	    self.relabeling = {node: i for i, node in enumerate(G.nodes)}
	    G = nx.relabel_nodes(G, self.relabeling)
	    # Adding complexity to the graph (removing edges, making directed, etc.)
	    G = self.remove_edges(G.to_directed()) if self.graph_directed else self.remove_edges(G)
	    G = self.add_edge_weights(G)
	    # Calculating shortest paths
	    shortest_paths = dict(nx.all_pairs_shortest_path(G))
	    # Creating files for next location and travel time if they don't exist
	    if not os.path.isfile(f'synthetic_dataset/travel_data/zone_path_{filename}.csv'):
	        self.create_next_location_file(G.nodes, shortest_paths)
	    if not os.path.isfile(f'synthetic_dataset/travel_data/zone_traveltime_{filename}.csv'):
	        self.create_time_to_location_file(G, shortest_paths)
	    # Loading travel time and shortest path data
	    self.travel_time = pd.read_csv(f'synthetic_dataset/travel_data/zone_traveltime_{filename}.csv', header=None).values
	    self.shorest_paths = pd.read_csv(f'synthetic_dataset/travel_data/zone_path_{filename}.csv', header=None).values
	    return G

	def remove_edges(self, G):
	    """
	    Removes a specified percentage of edges from the graph G.

	    The function randomly removes edges from the graph while ensuring that the
	    graph remains connected (or strongly connected if directed). The percentage of
	    edges to be removed is determined by `base_edge_removal_percentage` and 
	    `edge_percent_reduce`.

	    Args:
	    G (Graph): The graph from which edges are to be removed.

	    Returns:
	    Graph: The graph after edge removal.
	    """
	    assert self.base_edge_removal_percentage >= self.edge_percent_reduce
	    num_edges_to_remove = int(G.number_of_edges() * self.base_edge_removal_percentage)
	    edges_removed = []
	    edges = list(G.edges)
	    np.shuffle(edges)
	    H = G.copy()
	    for edge in edges:
	        if len(edges_removed) != num_edges_to_remove:
	            H.remove_edge(*edge)
	            edges_removed.append(edge)
	            connectiveness = nx.is_strongly_connected(H) if self.graph_directed else nx.is_connected(H)
	            if not connectiveness:
	                H.add_edge(*edge)
	                edges_removed.remove(edge)
	    G_edges_to_remove = edges_removed[:int(G.number_of_edges() * self.edge_percent_reduce)]
	    for edge in G_edges_to_remove:
	        G.remove_edge(*edge)
	    return G

	def add_edge_weights(self, G):
	    """
	    Assigns weights to the edges of the graph G.

	    This function sets a weight for each edge in the graph. The weight can be dynamic,
	    chosen from a set of options, or a fixed value representing road speed.

	    Args:
	    G (Graph): The graph to which edge weights are to be added.

	    Returns:
	    Graph: The graph after adding edge weights.
	    """
	    for start, end in G.edges:
	        new_weight = np.choice(self.edge_length_options) if self.dynamic_weights else self.road_speed
	        G[start][end]['weight'] = new_weight
	    return G

	def create_next_location_file(self, nodes, sp):
	    """
	    Creates a CSV file that maps each node to its next node in the shortest path.

	    Args:
	    nodes (list): A list of nodes in the graph.
	    sp (dict): A dictionary containing the shortest paths between nodes.
	    """
	    overall_next_nodes = []
	    for start_node in nodes:
	        next_node = []
	        for dest_node in nodes:
	            path = sp[start_node][dest_node]
	            next_node.append(path[1] if len(path) > 1 else path[0])
	        overall_next_nodes.append(next_node)
	    next_node_df = pd.DataFrame(overall_next_nodes)
	    next_node_df.to_csv(f'synthetic_dataset/travel_data/zone_path_{filename}.csv', header=False, index=False)

	def create_time_to_location_file(self, G, sp):
	    """
	    Creates a CSV file mapping each node to its travel time to every other node.

	    Args:
	    G (Graph): The graph containing nodes and weighted edges.
	    sp (dict): A dictionary containing the shortest paths between nodes.
	    """
	    overall_times = []
	    for start_node in G.nodes:
	        node_distances = []
	        for end_node in G.nodes:
	            path = sp[start_node][end_node]
	            time = sum([G[path[index]][path[index + 1]]['weight'] for index in range(len(path) - 1)]) if len(path) > 1 else 0.0
	            node_distances.append(time)
	        overall_times.append(node_distances)
	    time_df = pd.DataFrame(overall_times)
	    time_df.to_csv(f'synthetic_dataset/travel_data/zone_traveltime_{filename}.csv', header=False, index=False)

	def set_zones(self):
	    """
	    Sets up the zones for the analysis.

	    Divides the nodes into different zones based on the specified sectioning.
	    Each section is defined by a number in `self.sections`.

	    Returns:
	    dict: A dictionary representing the zones.
	    """
	    zones = {0: {node: node for node in self.nodes_list}}
	    for zone_id, num_zones in enumerate(self.sections, 1):
	        zones[zone_id] = self.get_zone_dict(num_zones)
	    return zones

	def get_zone_dict(self, num_zones):
	    """
	    Reads zone definitions from a file and creates a dictionary mapping nodes to their zone IDs.

	    Args:
	    num_zones (int): The number of zones to read from the file.

	    Returns:
	    dict: A dictionary mapping each node to its corresponding zone ID.
	    """
	    with open(f'synthetic_dataset/preset_zones/{numpy.prod(self.dims)}/preset_zones_{num_zones}.txt') as file:
	        nodes_in_zones = [line.rstrip() for line in file]
	    return {int(node): zone_id for zone_id, nodes in enumerate(nodes_in_zones) for node in nodes.split(' ')}

	def get_passenger_distribution(self):
	    """
	    Generates a passenger distribution based on a Poisson distribution.

	    Returns:
	    dict: A dictionary representing the probability of having a certain number of passengers.
	    """
	    def get_pass():
	        num_pass = poisson.rvs(mu=1)
	        while num_pass not in range(1, 7):
	            num_pass = poisson.rvs(mu=1)
	        return num_pass
	    x = {i: val / 10000 for i, val in dict(Counter([get_pass() for _ in range(10000)])).items()}
	    p = numpy.array(list(x.values()))
	    p /= p.sum()
	    return dict([(value, float(prob)) for value, prob in zip(x.keys(), p)])

	def get_request_distribution(self):
	    """
	    Creates a request distribution based on the dataset and a curve fitting function.

	    Returns:
	    dict: A dictionary mapping each time interval to its corresponding request probability.
	    """
	    data_distribution = numpy.array(self.df['Time'].value_counts().to_frame(name='Counts').sort_index().Counts) / self.scaling_factor
	    curve_func = lambda x, a, x0, s: a * exp(-0.5 * (x - x0) ** 2 / s ** 2)
	    x = arange(len(data_distribution))
	    p, _ = curve_fit(curve_func, x, data_distribution, p0=[100, 5, 2])
	    request_distribution = {}
	    # Different distribution calculations based on data usage
	    if self.use_data_literal and (self.end_hour - self.start_hour) == 1:
	        n = numpy.linspace(0, 96, 60)
	        for t, time in enumerate(range(self.start_hour * 3600, self.end_hour * 3600, 60)):
	            request_distribution[time] = float(curve_func(n[t], *p))
	    else:
	        n = numpy.linspace(0, 96, 1440)
	        for time, num in enumerate(range(len(n))):
	            request_distribution[time * 60] = curve_func(n[num], *p)
	        request_distribution = {time: float(value) for time, value in request_distribution.items() if (time >= self.start_hour * 3600) and (time < self.end_hour * 3600)}
	    return request_distribution

	def get_request_information(self):
	    """
	    Processes the dataset to extract and set information about transportation requests.

	    This function computes various request-related information such as pickup and dropoff nodes,
	    request prevalence, and top locations for the requests.
	    """
	    occurence_df = self.df['Trip Coordinates'].value_counts()[:self.request_total].to_frame(name='Occurences')
	    df = self.df.loc[self.df['Trip Coordinates'].isin(list(occurence_df.index))]
	    pickup_coords, dropoff_coords = list(df['Pickup Coordinates'].unique()), list(df['Dropoff Coordinates'].unique())
	    unique_locations = list(set(pickup_coords + dropoff_coords))
	    locations_to_nodes = self.randomly_set_locations(unique_locations) if self.random_placement else self.cluster_set_locations(unique_locations)
	    self.pickup_nodes = [locations_to_nodes[coord] for coord in pickup_coords]
	    self.dropoff_nodes = [locations_to_nodes[coord] for coord in dropoff_coords]
	    self.prev_of_requests, self.unique_requests, occurence_df = self.set_request_prevalence(occurence_df, locations_to_nodes)
	    self.top_locations = self.get_top_locations(occurence_df)

	def randomly_set_locations(self, locations):
	    """
	    Assigns a random node to each location.

	    Args:
	    locations (list): A list of locations to be assigned to nodes.

	    Returns:
	    dict: A dictionary mapping each location to a randomly chosen node.
	    """
	    locations_dict = {}
	    nodes = deepcopy(self.nodes_list)
	    for location in locations:
	        node = np.choice(nodes)
	        nodes.remove(node)
	        locations_dict[location] = node
	    assert len(locations_dict) == len(locations)
	    return locations_dict

	def cluster_set_locations(self, locations):
	    """
	    Groups locations into clusters and assigns a node to each cluster.

	    This function creates a graph where locations are nodes and edges represent 
	    proximity between locations. It then uses community detection to cluster these 
	    locations and assigns a node to each cluster.

	    Args:
	    locations (list): A list of geographical locations (latitude, longitude tuples).

	    Returns:
	    dict: A dictionary mapping each location to a node corresponding to its cluster.
	    """
	    G = nx.Graph()
	    G.add_nodes_from(locations)
	    # Adding edges between locations within a certain distance
	    for loc_a in locations:
	        for loc_b in locations:
	            distance = float(geopy.distance.geodesic(loc_a, loc_b).km)
	            if distance <= 4:
	                G.add_edge(loc_a, loc_b, weight=distance)
	    # Community detection to find clusters of locations
	    location_communities = community_louvain.best_partition(G, weight='weight', random_state=args.seed)
	    return self.place_in_community(location_communities)

	def place_in_community(self, location_communities):
	    """
	    Places each location in its respective community into a corresponding zone.

	    Args:
	    location_communities (dict): A dictionary mapping locations to their community IDs.

	    Returns:
	    dict: A dictionary mapping each location to a node within the corresponding zone.
	    """
	    num_unique_communities = len(set(location_communities.values()))
	    zones = self.get_zone_dict(num_unique_communities)
	    locations_dict = {}
	    # Assigning a node to each location based on its community
	    for coordinate, zone_of_coordinate in location_communities.items():
	        nodes_in_zone = [node for node, zone in zones.items() if zone == zone_of_coordinate]
	        chosen_node = np.choice(nodes_in_zone)
	        while chosen_node in locations_dict.values():
	            chosen_node = np.choice(nodes_in_zone)
	        locations_dict[coordinate] = chosen_node
	    assert len(locations_dict) == len(location_communities)
	    return locations_dict

	def set_request_prevalence(self, df, locations_to_nodes):
	    """
	    Sets the prevalence of each request in the dataset.

	    Args:
	    df (DataFrame): A DataFrame containing occurrences of each trip.
	    locations_to_nodes (dict): A dictionary mapping locations to nodes.

	    Returns:
	    tuple: Contains a dictionary of trip prevalence, a list of unique trips, and the modified DataFrame.
	    """
	    new_index = []
	    for (pickup_coordinate, dropoff_coordinate) in df.index:
	        start = locations_to_nodes[pickup_coordinate]
	        end = locations_to_nodes[dropoff_coordinate]
	        travel_time = self.travel_time[start][end]
	        new_index.append((start, end, travel_time))
	    df.index = new_index
	    df['Occurences'] /= sum(df['Occurences'])
	    prev_dict = {trip: row['Occurences'] for trip, row in df.iterrows()}
	    return prev_dict, list(df.index), df

	def get_top_locations(self, df):
	    """
	    Identifies the most popular pickup locations in each zone.

	    Args:
	    df (DataFrame): A DataFrame containing trip occurrences.

	    Returns:
	    list: A list of nodes representing the top pickup locations in each zone.
	    """
	    unique_pickups = list(set([index[0] for index, _ in df.iterrows()]))
	    zones = list(range(self.sections[0]))
	    top_node_in_zones = {zone: (-1, -1) for zone in zones}
	    # Finding the top pickup location in each zone
	    for node in unique_pickups:
	        zone_of_node = self.zones[1][node]
	        percent_pickup = sum([row.Occurences for index, row in df.iterrows() if index[0] == node])
	        if percent_pickup > top_node_in_zones[zone_of_node][1]:
	            top_node_in_zones[zone_of_node] = (node, percent_pickup)
	    return [int(n[0]) for n in top_node_in_zones.values() if int(n[0]) != -1]

if __name__ == '__main__':
    # Setting up a parser for command-line arguments
    parser = argparse.ArgumentParser()
    # Adding various arguments for configuration
    parser.add_argument('-dims', '--dims', type=list, default=[15, 15])  # Grid dimensions
    parser.add_argument('-start_hour', '--start_hour', type=int, default=11)  # Start hour
    parser.add_argument('-end_hour', '--end_hour', type=int, default=12)  # End hour
    parser.add_argument('-road_speed', '--road_speed', type=float, default=30.0)  # Road speed
    parser.add_argument('-use_data_literal', '--use_data_literal', type=bin, default=1)  # Use literal data distribution
    parser.add_argument('-request_total', '--request_total', type=int, default=150)  # Total number of requests
    parser.add_argument('-sections', '--sections', type=list, default=[10, 7, 4])  # Zone bisections
    parser.add_argument('-scaling_factor', '--scaling_factor', type=int, default=2500)  # Scaling factor for data
    parser.add_argument('-random_placement', '--random_placement', type=int, default=0)  # Random location placement
    parser.add_argument('-graph_directed', '--graph_directed', type=int, default=1)  # Directed graph
    parser.add_argument('-dynamic_weights', '--dynamic_weights', type=int, default=1)  # Dynamic edge weights
    parser.add_argument('-edge_length_options', '--edge_length_options', type=list, default=[20.0, 25.0, 30.0, 35.0, 40.0])  # Edge length options
    parser.add_argument('-edge_percent_reduce', '--edge_percent_reduce', type=float, default=0.1)  # Edge reduction percentage
    parser.add_argument('-seed', '--seed', type=int, default=1)  # Random seed
    args = parser.parse_args()

    # Setting up a random state based on the seed
    np = numpy.random.RandomState(args.seed)
    # Generating a filename based on arguments
    filename = f'{args.request_total}_{args.graph_directed}_{args.dynamic_weights}_{args.edge_percent_reduce}_{args.seed}'
    # Assertion to check the edge reduction percentage
    assert (args.edge_percent_reduce <= 0.6) if args.graph_directed else (args.edge_percent_reduce <= 0.4)

    # Data preparation using the provided arguments
    Prep = DataPreperation(vars(args))
    # Data generation using parameters from DataPreperation
    Data = DataGenerator(start_hour=args.start_hour, end_hour=args.end_hour, ...)

    # Check if the directory exists, if not, create it
    if not os.path.exists(f'generations/synthetic_{filename}'):
        os.makedirs(f'generations/synthetic_{filename}')

    # Saving the generated data to a pickle file
    with open(f'generations/synthetic_{filename}/data_{filename}.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)




