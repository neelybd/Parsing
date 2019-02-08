import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tkinter import Tk
import multiprocessing
from tkinter.filedialog import askopenfilename, asksaveasfilename

def main():
    print("Program: Parsing")
    print("Release: 1.5")
    print("Date: 2019-02-08")
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

    # Ask for Encoding
    encoding = encoding_selection()

    # Read data
    data = pd.read_csv(file_in, low_memory=False, encoding=encoding)

    # Create an empty output file
    open(file_out, 'a').close()

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

    # Lowercase parse_list
    parse_list_lower = list()
    for i in parse_list:
        parse_list_lower.append(i.lower())

    # Dedupe list
    print("Removing Duplicates for Parsed Field...")
    deduped_list = list()
    for index, i in enumerate(parse_list_lower):
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
        data[i] = "0"
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

def encoding_selection():
    basic_encoders = ['utf_8','latin1','utf_16','See All Encoders']
    advanced_encoders = ['ascii','big5','big5hkscs','cp037','cp424',
                         'cp437','cp500','cp720','cp737','cp775',
                         'cp850','cp852','cp855','cp856','cp857',
                         'cp858','cp860','cp861','cp862','cp863',
                         'cp864','cp865','cp866','cp869','cp874',
                         'cp875','cp932','cp949','cp950','cp1006',
                         'cp1026','cp1140','cp1250','cp1251','cp1252',
                         'cp1253','cp1254','cp1255','cp1256','cp1257',
                         'cp1258','euc_jp','euc_jis_2004','euc_jisx0213','euc_kr',
                         'gb2312','gbk','gb18030','hz','iso2022_jp',
                         'iso2022_jp_1','iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext',
                         'iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4',
                         'iso8859_5','iso8859_6','iso8859_7','iso8859_8','iso8859_9',
                         'iso8859_10','iso8859_11','iso8859_13','iso8859_14','iso8859_15',
                         'iso8859_16','johab','koi8_r','koi8_u','mac_cyrillic',
                         'mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish',
                         'ptcp154','shift_jis','shift_jis_2004','shift_jisx0213','utf_32',
                         'utf_32_be','utf_32_le','utf_16','utf_16_be','utf_16_le',
                         'utf_7','utf_8','utf_8_sig']
    while True:
        try:
            print("Select encoder.")
            for j, i in enumerate(basic_encoders):
                if j != len(basic_encoders) - 1:
                    print(str(j) + ": to use " + str(i) + "")
                else:
                    print(str(j) + ": to see all possible encoders.")
            encoder = basic_encoders[int(input("Enter Selection: "))]
        except ValueError:
                print("Input must be integer between 0 and " + str(len(basic_encoders)))
                continue
        else:
            break

    if encoder == 'See All Encoders':
        while True:
            try:
                print("Select encoder.")
                for j, i in enumerate(advanced_encoders):
                    print(str(j) + ": to use " + str(i) + "")
                encoder = advanced_encoders[int(input("Enter Selection: "))]
            except ValueError:
                print("Input must be integer between 0 and " + str(len(advanced_encoders)))
                continue
            else:
                break
    return encoder

if __name__ =='__main__':
    main()