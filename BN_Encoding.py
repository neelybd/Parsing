import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tkinter import Tk
import multiprocessing
from tkinter.filedialog import askopenfilename, asksaveasfilename

def main():
    print("Program: Parsing")
    print("Release: 1.1")
    print("Date: 2019-02-04")
    print("Author: Brian Neely")
    print()
    print()
    print("This program takes a csv given csv, parses, and encodes a given column of the data from a given deliminator.")
    print("The processing time varies exponentially with the number of encoding categories and rows.")
    print()
    print()


    # Hide Tkinter GUI
    Tk().withdraw()


    # Find input file
    file_in = askopenfilename(initialdir="../", title="Select file",
                              filetypes=(("Comma Separated Values", "*.csv"), ("all files", "*.*")))
    if not file_in:
        input("Program Terminated. Press Enter to continue...")
        exit()

    # Set ouput file
    file_out = asksaveasfilename(initialdir=file_in, title="Select file",
                                 filetypes=(("Comma Separated Values", "*.csv"), ("all files", "*.*")))
    if not file_out:
        input("Program Terminated. Press Enter to continue...")
        exit()

    # Create an empty output file
    open(file_out, 'a').close()

    # Read data
    data = pd.read_csv(file_in)

    # Create Column Header List
    headers = list(data.columns.values)

    # Select Column
    column = column_selection(headers)

    print()
    # Select Encoding
    deliminator = input("Enter deliminators separated by spaces: ")
    print()
    print("Processing File: " + file_in)

    # Parsed array
    print("Parsing column: [" + str(column) + "]...")
    parsed_array = [i.split(deliminator) for i in data[column]]
    print("Column: [" + str(column) + "] Parsed!")
    print()

    # Unpivot array
    parse_list = list()
    for i in range(len(parsed_array)):
        parse_list.extend(parsed_array[i])

    # Dedupe list
    print("Removing Duplicates for Parsed Field...")
    deduped_list = list()
    for index, i in enumerate(parse_list):
        if i not in deduped_list:
            deduped_list.append(i)

    # Remove None for Deduped List
    deduped_list = [x for x in deduped_list if x is not None]
    print("Duplicates Removed!")
    print()

    print("Number of Unique words: " + str(len(deduped_list)))
    print()

    # *****Split data for parallel processing*****
    print("Calculating Splits...")
    # Find number of CPUs and multiply by 16 for number of parallel threads
    num_splits = multiprocessing.cpu_count() * 16
    # Calculate the split locations
    split_locations = np.linspace(0,len(data),num_splits)
    # Rounds up the  split_locations
    split_locations = np.ceil(split_locations)
    # Convert split_locations to int for splitting data
    split_locations = split_locations.astype(int)
    # Split data for parallel processing
    data_split = np.split(data, split_locations)
    print("Splits Calculated!")
    print()
    # *****End Split*****

    # Parse Data using parallel process
    print("Encoding Parsed Data...")
    data_split_parsed = Parallel(n_jobs=-2)(delayed(encoding_data)(par_index + 1, len(data_split), i, column, deduped_list) for par_index, i in enumerate(data_split))
    print("Encoding Complete!")
    print()

    # Union split data frames
    data_out = pd.concat(data_split_parsed)

    #Write CSV
    print("Writing CSV File...")
    data_out.to_csv(file_out)
    print("Wrote CSV File!")
    print()

    print("Encoding Completed on column: [" + column + "]")
    print("File written to: " + file_out)
    input("Press Enter to close...")

def encoding_data(par_index, par_len, data, column, deduped_list):
    pd.options.mode.chained_assignment = None  # default='warn'
    for index, i in enumerate(deduped_list):
        # Add Columns
        data[i] = ""
        # Encode Columns
        data[i][data[column].str.find(i) != -1] = 1

    print("Completed: " + str(par_index) + " out of " + str(par_len))
    return data

def column_selection(headers):
    while True:
        try:
            print("Select column to parse and encode.")
            for j, i in enumerate(headers):
                print(str(j) + ": to parse and encode column [" + str(i) + "]")
            column = headers[int(input("Enter Selection: "))]
        except ValueError:
                print("Input must be integer between 0 and " + str(len(headers)))
                continue
        else:
            break
    return column

if __name__ =='__main__':
    main()