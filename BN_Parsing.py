import numpy as np
import pandas as pd
from joblib import delayed, Parallel
import multiprocessing
from file_handling import *
from selection import *
from functions import *
import time
import operator
from sklearn.feature_extraction.text import CountVectorizer
import csv


def main():
    print("Program: Parsing")
    print("Release: 1.11.3")
    print("Date: 2020-06-26")
    print("Author: Brian Neely")
    print()
    print()
    print(
        "This program takes a csv given csv, parses, and encodes a given column of the data from a given deliminator.")
    print("The processing time varies exponentially with the number of encoding categories and rows.")
    print()
    print()

    # Find input file
    file_in = select_file_in()

    # Set output file
    file_out = select_file_out_csv(file_in)

    # Ask for delimination
    delimination = input("Enter Deliminator: ")

    # Open input csv using the unknown encoder function
    data = open_unknown_csv(file_in, delimination)

    # Create an empty output file
    open(file_out, 'a').close()

    # Create Column Header List
    headers = list(data.columns.values)

    # Select Column
    column = column_selection(headers, "parsing")

    # Export list of parsed words
    if y_n_question("Export list of parsed words (y/n): "):
        # Set flag for export parse list
        export_parsed_list = True

        # Select second file out
        file_out_parse_list = select_file_out_csv(file_out)

    print()

    if y_n_question("Split data using spaces? Note: This will speed up processing time significantly. (y/n):"):
        # Start Timer
        start_time = time.time()

        # Parse Data
        data_out, new_headers = vectorize_text(data, column)

        # Print Time
        print("Parsing completed in " + str(round(time.time() - start_time, 2)) + " s")

    else:
        # Select Encoding Delimination
        deliminator = input("Enter deliminators separated by spaces: ")
        while deliminator is None:
            deliminator = input("No deliminator selected! Enter deliminators separated by spaces: ")
        print()
        print("Processing File: " + file_in)

        # Get name to append
        encode_concate = input("Append string to encoded column name: ")

        # Set parallel to true
        parallel = True

        data_out, new_headers = parse_and_encode_data(data, column, deliminator, encode_concate, parallel)

    # Write CSV
    print("Writing CSV File...")
    data_out.to_csv(file_out, index=False)
    print("Wrote CSV File!")
    print()

    # If parse list, write CSV
    if export_parsed_list:
        # Write list
        with open(file_out_parse_list, 'w') as write_file:
            writer = csv.writer(write_file, dialect='excel')
            writer.writerow(new_headers)

    print("Encoding Completed on column: [" + column + "]")
    print("File written to: " + file_out)
    input("Press Enter to close...")


def vectorize_text(data, column):
    print()
    print("Creating Vectorizer...")
    # Set vectorizer from CountVectorizer
    vectorizer = CountVectorizer()
    print("Vectorizer Created!")

    # Fill NaN
    print()
    print("Dropping NAs...")
    data[column] = data[column].fillna("empty_text")
    print("NAs Dropped!")

    # Create sparse matrix of parsed text
    print()
    print("Creating Sparse Matrix of Vectorized Data...")
    X = vectorizer.fit_transform(data[column])
    print("Sparse matrix created!")

    # Convert sparse matrix to DataFrame
    print()
    print("Converting sparse matrix to dense...")
    parsed = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    print("Conversion completed!")

    # Look for columns in the original data that matches new columns
    print()
    print("Creating list of new columns...")
    any_match_found = True
    while any_match_found:
        any_match_found = False

        for i in parsed:
            column_match_found = False
            for j in column:
                # Set flag for found match
                column_match_found = True

            # If match found, add string into column name
            parsed.rename(columns={i: i + "_parsed"}, inplace=True)
            any_match_found = True

    # Get new headers
    new_headers = list(parsed.columns.values)
    print("List created!")

    # Append original dataset to parsed dataset
    print()
    print("Appending new matrix to original data...")
    data_out = pd.concat([data, parsed], axis=1, sort=False)
    print("Append complete!")

    # Return parsed data
    return data_out, new_headers


def parse_and_encode_data(data, column, deliminator, encode_concate, parallel=False):
    # Replace first deliminator if it is in the first character position
    column_list = []
    for i in data[column]:
        if type(i) != float:
            if i.find(deliminator) == 0:
                i = i[1:]
            column_list.append(i)

    # Parsed array
    print("Parsing column: [" + str(column) + "]...")
    parsed_array = [i.split(deliminator) for i in column_list]
    print("Column: [" + str(column) + "] Parsed!")
    print()

    # Unpivot array
    parse_list = list()
    for i in range(len(parsed_array)):
        parse_list.extend(parsed_array[i])

    # Lowercase parse_list
    parse_list_lower = list()
    for i in parse_list:
        parse_list_lower.append(i.lower())

    # Dedupe list
    print("Removing Duplicates for Parsed Field...")
    deduped_list = dedupe_list(parse_list_lower)

    # Remove None for Deduped List
    deduped_list = [x for x in deduped_list if x is not None]
    print("Duplicates Removed!")
    print()

    # Add _encoder string
    if encode_concate == "":
        deduped_list_concat = deduped_list
    else:
        deduped_list_concat = list()
        for i in deduped_list:
            deduped_list_concat.append(i + encode_concate)

    print("Number of Unique words: " + str(len(deduped_list)))
    print()

    # Create dataframe of rows with empty strings
    empty_list = data[column].isnull()
    data_empty = data[empty_list]
    data_filled = data[empty_list == False]

    # Start time
    if parallel:
        # *****Find optimum split*****
        if len(data_filled) > 8192:
            # Create sample dataset
            data_filled_sample = data_filled.head(1024)

            # List of number of splits to test
            num_splits_test = [8, 16, 32, 64, 128, 256, 512]

            # Test splits
            old_time = 99999
            time_dict = dict()
            for splits in num_splits_test:
                # Start Time
                start_time = time.time()

                # Print statement
                print("Testing " + str(splits) + " splits")

                # Split Data
                data_split = split_data(data_filled_sample, splits)

                # Test Speed
                Parallel(n_jobs=-1)(delayed(encoding_data)
                                    (par_index + 1, len(data_split), i, column, deduped_list_concat, encode_concate)
                                    for par_index, i in enumerate(data_split))

                # Record time and print results
                time_dict[splits] = time.time() - start_time
                print(str(splits) + " Splits: " + str(round(time_dict[splits], 2)) + " s")

                # If time is increasing, stop and use current. More data tends to benefit from a slight increase in
                # number of splits.
                if time_dict[splits] > old_time:
                    break
                else:
                    old_time = time_dict[splits]

            # Lookup the optimum split
            optimum_num_splits = min(time_dict.items(), key=operator.itemgetter(1))[0]
        else:
            optimum_num_splits = 16

        # Split data for optimum number of splits
        data_split = split_data(data_filled, optimum_num_splits)

        # Parse Data using parallel process
        start_time = time.time()
        print("Encoding Parsed Data on full dataset...")
        data_split_parsed = Parallel(n_jobs=-1) \
            (delayed(encoding_data)(par_index + 1, len(data_split), i, column, deduped_list_concat, encode_concate)
             for par_index, i in enumerate(data_split))
        print("Encoding Complete!")
        print()

        # Union split data frames
        data_encoded = pd.concat(data_split_parsed)

    else:
        # Start timer
        start_time = time.time()

        # Single Thread
        data_encoded = encoding_data(1, 1, data_filled, column, deduped_list_concat, encode_concate)

    # Bring back data_empty with 0's in new columns
    for i in deduped_list_concat:
        data_empty[i] = 0

    # Union dataframes encoded and empty
    data_out = pd.concat([data_encoded, data_empty])

    # End time
    print("Parsing completed in " + str(round(time.time() - start_time, 2)) + " s")

    # Look for columns in the original data that matches new columns
    any_match_found = True
    while any_match_found:
        any_match_found = False

        for i in data_out:
            column_match_found = False
            for j in column:
                # Set flag for found match
                column_match_found = True

            # If match found, add string into column name
            data_out.rename(columns={i: i + "_parsed"}, inplace=True)
            any_match_found = True

    # Get original columns
    headers_original = list(data.columns.values)

    # Get output headers
    headers_new = list(data_out.columns.values)

    # Add print statement and timer
    print("Extracting new columns added...")
    start_time = time.time()

    # Look for differences between original headers and new
    new_headers = list_diff(headers_original, headers_new)

    # Print time and results
    print(str(len(new_headers)) + " new columns found in " + str(round(time.time() - start_time, 2)) + " s")

    # Return data
    return data_out, new_headers


def encoding_data(par_index, par_len, data, column, deduped_list, encode_concate):
    pd.options.mode.chained_assignment = None  # default='warn'
    for index, i in enumerate(deduped_list):
        # Add 0 for new column
        data[i] = "0"
        # Encode Columns
        data[i][data[column].str.find(i.replace(encode_concate, "")) != -1] = 1

    print("Completed: " + str(par_index) + " out of " + str(par_len))
    return data


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
