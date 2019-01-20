import numpy as np
import pandas as pd
from joblib import delayed, parallel
from tkinter import Tk

from tkinter.filedialog import askopenfilename, asksaveasfilename

def main():
    print("Program: Parsing")
    print("Release: 1.0")
    print("Date: 2019-01-19")
    print("Author: Brian Neely")
    print()
    print()
    print("-----")
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

    # Parsed array
    parsed_array = [i.split(deliminator) for i in data[column]]

    # Unpivot array
    parsed_list = list()
    for i in parsed_array:
        parsed_list = parsed_list + i

    # Dedupe list
    deduped_list = list()
    for i in parsed_list:
        if i not in deduped_list:
            deduped_list.append(i)

    # Add columns to data out
    data_out = data
    for i in deduped_list:
        data_out[i] = ""
        data_out[i][data_out[column].str.find(i) != -1] = 1

    #Write CSV
    data_out.to_csv(file_out)

    print("Encoding Completed on column: [" + column + "]")
    print("File written to: " + file_out)
    input("Press Enter to close...")

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