import sys
sys.dont_write_bytecode = True  # Prevent Python from generating .pyc files
import argparse
import pandas as pd
import numpy
import pickle
from collections import Counter
from DataGenerator import DataGenerator  # Custom module for data generation
import os
from copy import copy

class DataPreperation(object):
    def __init__(self, initial_vars):
        """
        Initialize the data preparation class with the given variables and
        perform initial data loading and processing.

        Args:
        initial_vars (dict): Dictionary containing initialization variables.
        """
        # Set attributes based on initial_vars dictionary
        for key in initial_vars:
            setattr(self, key, initial_vars[key])

        # Load datasets and perform initial processing
        self.locations_df = pd.read_csv('real_dataset/locations.csv', index_col='LocationID')
        self.nodes_list = self.get_nodes_list()
        self.passenger_distribution = {1: 0.717, 2: 0.139, 3: 0.04, 4: 0.018, 5: 0.053, 6: 0.033}
        self.travel_time = pd.read_csv(f'real_dataset/travel_data/zone_traveltime.csv', header=None).values
        with open('real_dataset/unfiltered_requests.pickle', 'rb') as handle: 
            self.unfiltered_requests = pickle.load(handle)

        # Further data processing and preparation
        self.collect_filtered_requests()
        self.get_request_distribution()
        self.get_popular_location_in_each_zone()
        self.get_aggregated_zones()
        self.prev_of_requests = self.temp_prev_of_requests if self.temporal else self.prev_of_requests

	def get_nodes_list(self):
	    """
	    Reads the nodes data from a CSV file and processes it to obtain a list of nodes.

	    This function reads the nodes' coordinates from a CSV file, converts them to floats,
	    and then filters the geographic area based on these coordinates.

	    Returns:
	    list: A list of nodes after filtering the geographic area.
	    """
	    # Reading nodes and coordinates from a CSV file
	    df = pd.read_csv('real_dataset/points.csv', header=None, names=['NumNodes', 'Coords']).set_index('NumNodes')
	    # Splitting the coordinates string into a list of floats
	    self.coordinates = list(map(float, df.loc[self.num_nodes].values[0].split('/')))
	    # Returning the filtered list of nodes based on geographic area
	    return self.filter_geographic_area()

	def filter_geographic_area(self):
	    """
	    Filters nodes based on geographic coordinates.

	    This function uses the latitude and longitude coordinates to filter out nodes
	    that fall within the specified geographic area. The nodes list is shuffled
	    to randomize the order of nodes.

	    Returns:
	    list: A list of nodes filtered based on the geographic area.
	    """
	    # Filtering nodes based on latitude and longitude range
	    nodes_list = list(self.locations_df[self.locations_df['Lat'].between(self.coordinates[1], self.coordinates[0]) & 
	                                        self.locations_df['Long'].between(self.coordinates[2], self.coordinates[3])].index)
	    # Randomly shuffling the list of nodes
	    np.shuffle(nodes_list)
	    return nodes_list

	def collect_filtered_requests(self):
	    """
	    Collects and filters transportation requests.

	    This function gathers total transportation requests, counts the number of each unique request,
	    and then sorts and reduces these requests based on a predefined total request threshold.
	    It also calculates unique requests, previous requests, and pickup points.
	    """
	    # Gathering total transportation requests
	    total_requests = self.get_total_requests()
	    # Counting the number of each unique request
	    num_each_request = {request: value for request, value in dict(Counter(total_requests)).items()}
	    # Sorting and reducing requests based on the request total
	    reduced_requests = sorted(num_each_request.items(), key=lambda item: item[1], reverse=True)[:self.request_total]
	    # Calculating unique requests and previous requests
	    self.unique_requests, self.temp_prev_of_requests, self.prev_of_requests = self.get_temporal_prev_of_requests(reduced_requests)
	    # Extracting and storing unique pickup points
	    self.pickup_points = list(set([request[0] for request in self.unique_requests]))

	def get_total_requests(self):
	    """
	    Gathers all valid transportation requests within a specified time range.

	    Iterates through each day and time interval to compile a list of transportation
	    requests that meet the node list criteria. Only requests where both pickup and dropoff
	    locations are in the nodes list are included.

	    Returns:
	    list: A list of tuples, each containing pickup, dropoff locations, and original travel time.
	    """
	    total_requests = []
	    # Iterating over specified day and time ranges
	    for day in range(1, 21):
	        for time in range(self.start_hour * 3600, self.end_hour * 3600, 60):
	            # Filtering requests based on node list
	            for request in self.unfiltered_requests[day][time]:
	                if (request.pickup in self.nodes_list) and (request.dropoff in self.nodes_list):
	                    total_requests.append((request.pickup, request.dropoff, request.original_travel_time))
	    return total_requests

	def get_temporal_prev_of_requests(self, reduced_requests):
	    """
	    Calculates the temporal precedence of requests.

	    From the reduced set of requests, this function calculates the temporal precedence
	    of each request across different time intervals. This helps in understanding the
	    distribution and frequency of requests over time.

	    Args:
	    reduced_requests (list): A list of reduced transportation requests.

	    Returns:
	    tuple: Contains a list of unique nodes, and dictionaries for previous requests and overall previous requests.
	    """
	    # Extracting unique nodes and their travel times from reduced requests
	    unique_nodes = [(pickup, dropoff) for (pickup, dropoff, _), _ in reduced_requests]
	    node_travel_times = {(pickup, dropoff): time for (pickup, dropoff, time), _ in reduced_requests}
	    overall_request_prev = {node: 0 for node in unique_nodes}
	    prev_of_requests = {}

	    # Calculating precedence for each time interval
	    for time in range(self.start_hour * 3600, self.end_hour * 3600, 60):
	        requests_at_time = {node: 0 for node in unique_nodes}
	        for day in range(1, 21):
	            for request in self.unfiltered_requests[day][time]:
	                if (request.pickup, request.dropoff) in unique_nodes:
	                    requests_at_time[(request.pickup, request.dropoff)] += 1
	                    overall_request_prev[(request.pickup, request.dropoff)] += 1

	        # Calculating and normalizing request prevalence
	        total_requests_overall = sum(overall_request_prev.values())
	        total_request_at_time = sum(requests_at_time.values())
	        assert total_request_at_time > 0
	        prev_of_requests[time] = {(pickup, dropoff, node_travel_times[(pickup, dropoff)]): prev / total_request_at_time for (pickup, dropoff), prev in requests_at_time.items() if prev > 0}

	    overall_prev_of_requests = {(pickup, dropoff, node_travel_times[(pickup, dropoff)]): prev / total_requests_overall for (pickup, dropoff), prev in overall_request_prev.items()}
	    return [node for node, _ in reduced_requests], prev_of_requests, overall_prev_of_requests

	def get_request_distribution(self):
	    """
	    Calculates the distribution of requests for each time interval.

	    This function computes how many requests occur at each time interval over a
	    specified number of days. The result is a dictionary mapping each time interval
	    to its corresponding request distribution.
	    """
	    self.request_distribution = {}
	    # Iterating over each time interval
	    for time in range(self.start_hour * 3600, self.end_hour * 3600, 60):
	        requests_at_time = []
	        # Counting requests for each day and time
	        for day in range(1, 21):
	            requests_on_day = 0
	            for request in self.unfiltered_requests[day][time]:
	                if (request.pickup in self.nodes_list) and (request.dropoff in self.nodes_list):
	                    requests_on_day += 1
	            requests_at_time.append(requests_on_day)
	        # Calculating and normalizing the distribution for each time interval
	        self.request_distribution[time] = {r: n / 20 for r, n in dict(Counter(requests_at_time)).items()}

	def get_popular_location_in_each_zone(self):
	    """
	    Identifies the most popular pickup location in each zone.

	    This function divides the area into zones, places nodes into these zones,
	    and then determines the most popular pickup location in each zone based on
	    the frequency of requests originating from these locations.
	    """
	    # Creating zones and placing nodes in them
	    zones = self.create_zones()
	    nodes_in_zones = self.place_nodes_in_zones(zones)
	    # Calculating the popularity of pickup locations in each zone
	    self.get_pickup_popularity(nodes_in_zones)

	def create_zones(self):
	    """
	    Creates a grid of zones based on the provided coordinates.

	    Divides the geographic area defined by the coordinates into a grid of zones.
	    The number of divisions is determined by the 'intervals' attribute. The function
	    generates pairs of latitude and longitude intervals, forming a grid of rectangular zones.

	    Returns:
	    list: A list of tuples representing the rectangular zones.
	    """
	    # Extracting the coordinates
	    lat_up, lat_down, long_left, long_right = self.coordinates
	    # Creating intervals for latitude and longitude
	    long_intervals = numpy.linspace(long_left, long_right, self.intervals + 1)
	    lat_intervals = numpy.linspace(lat_down, lat_up, self.intervals + 1)
	    # Generating pairs of intervals to form rectangular zones
	    long_pairs = [(long_intervals[i], long_intervals[i + 1]) for i in range(len(long_intervals) - 1)]
	    lat_pairs = [(lat_intervals[i], lat_intervals[i + 1]) for i in range(len(lat_intervals) - 1)]
	    # Returning the grid of zones
	    return [(lats, longs) for lats in lat_pairs for longs in long_pairs]

	def get_pickup_popularity(self, nodes_zones):
	    """
	    Determines the most popular pickup location in each zone.

	    For each zone, this function calculates the popularity of each node (pickup location)
	    based on the frequency of requests originating from it. The most popular node in each
	    zone is then identified and stored.

	    Args:
	    nodes_zones (dict): A dictionary mapping zones to their respective nodes.
	    """
	    self.top_locations = []
	    # Iterating over each zone and its nodes
	    for zone, nodes in nodes_zones.items():
	        if len(nodes) > 0:
	            top_node = None
	            top_node_value = 0
	            # Finding the node with the highest request frequency
	            for node in nodes:
	                prev_of_node = sum([value for request, value in self.prev_of_requests.items() if request[0] == node])
	                if prev_of_node > top_node_value:
	                    top_node_value = prev_of_node
	                    top_node = node
	            # Adding the most popular node to the top locations list
	            self.top_locations.append(top_node)

	def get_aggregated_zones(self):
	    """
	    Aggregates zones based on different levels of bisection.

	    This function slightly expands the geographic area defined by the coordinates
	    and then creates aggregated zones for each level of bisection specified in 'sections'.
	    The nodes are then placed into these aggregated zones.

	    Each level of bisection (e.g., 12, 8, 4) defines how finely the area is divided.
	    """
	    # Slightly expanding the coordinates
	    lat_up, lat_down, long_left, long_right = copy(self.coordinates)
	    lat_up += 0.009
	    lat_down -= 0.009
	    long_right += 0.009
	    long_left -= 0.009

	    nodes_list = list(self.locations_df[self.locations_df['Lat'].between(lat_down, lat_up) &
	                                        self.locations_df['Long'].between(long_left, long_right)].index)
	    zone_definitions = {}
	    # Creating aggregated zones for each level of bisection
	    for bisections in self.sections:
	        long_intervals = numpy.linspace(long_left, long_right, bisections + 1)
	        lat_intervals = numpy.linspace(lat_down, lat_up, bisections + 1)
	        long_pairs = [(long_intervals[i], long_intervals[i + 1]) for i in range(len(long_intervals) - 1)]
	        lat_pairs = [(lat_intervals[i], lat_intervals[i + 1]) for i in range(len(lat_intervals) - 1)]
	        zones = [(lats, longs) for lats in lat_pairs for longs in long_pairs]
	        zone_definitions[len(zones)] = zones
	    # Placing locations into these zones
	    self.place_locations_in_zones(zone_definitions, nodes_list)

	def place_nodes_in_zones(self, zones):
	    """
	    Places each pickup point node into the corresponding zone.

	    This function iterates over all pickup points and assigns each node to a zone
	    based on its geographical coordinates. It checks whether the latitude and longitude
	    of a node fall within the boundaries of a zone.

	    Args:
	    zones (list): A list of tuples representing the rectangular zones.

	    Returns:
	    dict: A dictionary mapping each zone to a list of nodes within that zone.
	    """
	    nodes_in_zones = {zone: [] for zone in range(len(zones))}
	    # Iterating over each pickup point
	    for node in self.pickup_points:
	        node_long, node_lat = self.locations_df.loc[node].values
	        # Assigning nodes to zones based on their coordinates
	        for zone_id, zone in enumerate(zones):
	            if (zone[0][0] <= node_lat <= zone[0][1]) and (zone[1][0] <= node_long <= zone[1][1]):
	                nodes_in_zones[zone_id].append(node)
	                break
	    return nodes_in_zones

	def place_locations_in_zones(self, zone_definitions, nodes_list):
	    """
	    Organizes nodes into zones based on various levels of bisection.

	    For each level of bisection defined in 'zone_definitions', this function places nodes
	    into the corresponding zones. It also creates a hierarchical structure of zones, with level 0
	    representing individual nodes and higher levels representing aggregated zones.

	    Args:
	    zone_definitions (dict): A dictionary of zone definitions at each level of bisection.
	    nodes_list (list): A list of nodes to be placed in zones.
	    """
	    zones = list(zone_definitions.keys())
	    self.final_zones = {i: {} for i in range(len(zones) + 1)}
	    self.final_zones[0] = {i: i for i in nodes_list}
	    # Placing nodes into zones for each level of bisection
	    for level in range(1, len(zones) + 1):
	        for node in nodes_list:
	            node_long, node_lat = self.locations_df.loc[node].values
	            # Assigning nodes to the appropriate zone
	            for zone, zone_dimensions in enumerate(zone_definitions[zones[level - 1]]):
	                if (zone_dimensions[0][0] <= node_lat <= zone_dimensions[0][1]) and (zone_dimensions[1][0] <= node_long <= zone_dimensions[1][1]):
	                    self.final_zones[level][node] = zone
	                    break

if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser()
    # Definitions of command line arguments
    parser.add_argument('-intervals', '--intervals', type=int, default=4)  # Number of intervals for zone parsing
    parser.add_argument('-num_nodes', '--num_nodes', type=int, default=150)  # Number of nodes to consider
    parser.add_argument('-start_hour', '--start_hour', type=int, default=11)  # Start hour for data consideration
    parser.add_argument('-end_hour', '--end_hour', type=int, default=12)  # End hour for data consideration
    parser.add_argument('-test_days', '--test_days', type=int, default=20)  # Number of days for testing
    parser.add_argument('-request_total', '--request_total', type=int, default=50)  # Total number of requests
    parser.add_argument('-sections', '--sections', type=list , default=[12, 8, 4])  # Sections for data aggregation
    parser.add_argument('-temporal', '--temporal', type=int , default=0)
    parser.add_argument('-seed', '--seed', type=int, default=5)
    args = parser.parse_args()

    # Random number generator initialization
    np = numpy.random.RandomState(args.seed)
    filename = f'{args.request_total}_{args.num_nodes}_{args.temporal}_{args.seed}'

    # Data preparation and generation
    Prep = DataPreperation(vars(args))
    Data = DataGenerator(start_hour=args.start_hour, end_hour=args.end_hour, ...)

    # Check for existing directory and create if necessary
    if not os.path.exists(f'generations/real_{filename}'):
        os.makedirs(f'generations/real_{filename}')

    # Save the generated data
    with open(f'generations/real_{filename}/data_{filename}.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)



