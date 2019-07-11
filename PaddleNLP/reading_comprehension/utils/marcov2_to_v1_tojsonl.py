import sys
import json
import pandas as pd 

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: tojson.py <input_path> <output_path>')
        exit()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    df = pd.read_json(infile)
    with open(outfile, 'w') as f:
        for row in df.iterrows():
            f.write(row[1].to_json() + '\n')