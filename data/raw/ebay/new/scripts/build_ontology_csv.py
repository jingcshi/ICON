import pandas as pd
import owlready2 as o2
import argparse
import os
import re
import types

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert taxonomy to ontology.")

    # Required input argument
    parser.add_argument(
        'input',
        type=str,
        help='The path of taxonomy to read from, expected csv'
    )

    # Optional output argument
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='The path of output ontology to write to. If not specified, will be constructed from the input file path.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # If output is not provided, construct it from the input path
    if not args.output:
        input_path = args.input
        base_name, _ = os.path.splitext(input_path)
        args.output = f"{base_name}_onto.owl"

    return args

# Add explicit catgeory name and parent columns to the data 
def annotate_columns(df):
    current_breadcrumb = [0]
    names = []
    parents = []
    notna = df.notna()
    re_level = re.compile(r'^l(\d+)$')
    match = lambda s: re.findall(re_level, s)
    max_level = max([int(col[0]) for col in map(match, df.columns) if col])
    for i,row in df.iterrows():
        for lv in range(1,max_level+1):
            if notna.at[i, f'l{lv}']:
                break
        name = row[f'l{lv}']
        names.append(name)
        current_breadcrumb = current_breadcrumb[:lv] + [row['CategoryID']]
        parents.append(current_breadcrumb[-2])
    df['CategoryName'] = names
    df['Parent'] = parents
            
if __name__ == "__main__":
    
    args = parse_arguments()
    data = pd.read_csv(args.input)
    annotate_columns(data)
    site = data.at[0, 'siteID']
    
    onto = o2.get_ontology(f"https://ebay.com/ontology/{site}")
    with onto:
        root = types.new_class("0", (o2.Thing,))
        root.label = "Everything"
        for _,row in data.iterrows():
            id = row['CategoryID']
            name = row['CategoryName']
            parent = row['Parent']
            newclass = types.new_class(str(id), (onto[str(parent)],))
            newclass.label = name
            
    onto.save(file = args.output, format = "rdfxml")