import csv

def insert_newline_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    modified_rows = []
    for row in rows:
        modified_row = []
        for item in row:
            modified_item = ""
            i = 0
            while i < len(item) - 1:
                if item[i].islower() and item[i + 1].isupper():
                    modified_item += item[i] + ' '
                else:
                    modified_item += item[i]
                i += 1
            modified_item += item[-1]
            modified_row.append(modified_item)
        modified_rows.append(modified_row)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(modified_rows)

import csv
import re

def remove_square_brackets_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    modified_rows = []
    for row in rows:
        modified_row = []
        for item in row:
            # Use regular expression to remove content between square brackets
            modified_item = re.sub(r'\[.*?\]', '', item)
            modified_row.append(modified_item)
        modified_rows.append(modified_row)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(modified_rows)

input_file = "input.csv"  # Replace with your input file path
output_file = "output.csv"  # Replace with your output file path
remove_square_brackets_csv(input_file, output_file)


input_file = "output.csv"  # Replace with your input file path
output_file = "dataset.csv"  # Replace with your output file path
remove_square_brackets_csv(input_file, output_file)
