import csv

input_file = 'swissmetro.dat'
output_file = 'swissmetro.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)

    for line in infile:
        row = line.strip().split('\t')
        writer.writerow(row)
