from collections import OrderedDict
import numpy as np


print("Function: Functions")
print("Release: 1.0.0")
print("Date: 2020-06-17")
print("Author: Brian Neely")
print()
print()
print("General Functions")
print()
print()


def dedupe_list(duplicate_list):
    # *****Dedup list*****
    return list(OrderedDict.fromkeys(duplicate_list))


def split_data(data, num_splits):
    # *****Split data for parallel processing*****
    # Calculate the split locations
    split_locations = np.linspace(0, len(data), num_splits)
    # Rounds up the  split_locations
    split_locations = np.ceil(split_locations)
    # Convert split_locations to int for splitting data
    split_locations = split_locations.astype(int)
    # Split data for parallel processing
    data_split = np.split(data, split_locations)

    return data_split
