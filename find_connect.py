#!/usr/bin/env python3

'''
- Find connectivity in a cluster of points
- Dynamic connectivity problem
- Breadth-first search (BFS) method
- Use cdist in scipy to speed up finding distance between points
- Complexity performance of removing 
    - list.pop(0) is O(n) 
    - deque.popleft() is O(1)

- Rule 1: Threshold condition
    - Euclidean distance using cdist
- Rule 2: Path condition
    - Avoiding cycle within the cluster

Ref: https://en.wikipedia.org/wiki/Breadth-first_search
'''

import json
import pprint
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict as dfd
from collections import deque

# //////////////////////////////////////////////////
input_file = "in.json"
output_file = "my_out.json"
# //////////////////////////////////////////////////


class Connectivity:
    """Create instance of the cluster component 
    and use breadth-first search (BFS) algorithm
    to find the connectivity path for a given pair
    of points
    """

    def __init__(self, file):
        """Initialize variable and call 
        automated functions to extract and
        decorate the data.

        Arguments:
            file {str} -- Relative or full path of 
                          the input file
        """
        self.file = file

        # Use defaultdict for populating pairs
        # and avoid KeyError with regular dict
        self.cluster = dfd(list)

        self.extract_coord()
        self.calc_dist()

    def extract_coord(self):
        """Extract data from JSON input.
        - threshold - Cutoff distance
        - pairs - List of pair of index of two points
        - points - Index and coordinates of all points
        """
        with open(self.file, 'r') as f:
            data = f.read()
            json_data = json.loads(data)

        self.threshold = json_data['threshold']  # float
        self.pairs = json_data['pairs']  # list
        self.points = json_data['points']  # list

        # Remove the 'id' keys from dict
        self.coord = self.points
        for i in range(len(self.coord)):
            del self.coord[i]['id']

    def calc_dist(self):
        """Calculate distance between all points
        and screening out the pair of point whose
        distance is smaller than threshold.
        """
        self.coord = pd.DataFrame(self.coord).to_numpy()
        self.all_dist = cdist(self.coord, self.coord)
        self.thr_dist = np.clip(self.all_dist, 0, self.threshold)

    def create_cluster(self, index):
        """Create the component cluster of the point.

        Arguments:
            index {int} -- Index of a point of interest
        """
        self.survivor = np.where((self.thr_dist[index] != 0) &
                                 (self.thr_dist[index] != self.threshold)
                                 )[0]

        for i in range(len(self.survivor)):
            self.append_point(index, self.survivor[i])

    def append_point(self, a, b):
        """Append connected points to cluster.

        Arguments:
            a {int} -- Index of point i
            b {int} -- Index of point j
        """
        self.cluster[a].append(b)

    def path_search(self, a, b):
        """Search connectivity path in the cluster using
        Breadth-first search algorithm.

        Arguments:
            a {int} -- Index of starting point of path
            b {int} -- Index of ending point of path

        Returns:
            List -- If connectivity path is found, return list of point
            Bool -- If connectivity path is not found, return False
        """
        # If two points are the same 
        # just return one of them
        if a == b:
            return [a]

        # Initialize path
        checked_point = {a}

        spare = []
        lists = deque([(a, spare)])

        while lists:
            # Delete the point from the left end
            # and then get the path
            current_point, path = lists.popleft()
            # Add the path to the list points
            # that we have already gone
            checked_point.add(current_point)

            for near_point in self.cluster[current_point]:
                # Check if path is found
                if near_point == b:
                    return path + [current_point, near_point]
                if near_point in checked_point:
                    continue
                # pprint.pprint(path)
                lists.append((near_point, path + [current_point]))
                checked_point.add(near_point)

        return False


if __name__ == "__main__":

    my_cluster = Connectivity(input_file)
    # pprint.pprint(my_cluster.threshold)
    # pprint.pprint(my_cluster.pairs)
    # pprint.pprint(my_cluster.points)

    connectedRule1 = dict()
    totalPoints = len(my_cluster.points)

    for i in range(totalPoints):
        my_cluster.create_cluster(i)

    # pprint.pprint(my_cluster.cluster)

    results = []

    for i, pair in enumerate(my_cluster.pairs):
        found = my_cluster.path_search(pair[0], pair[1])
        if found:
            results.append(True)
        else:
            results.append(False)

    with open(output_file, 'w') as o:
        json.dump(results, o, indent=2)
