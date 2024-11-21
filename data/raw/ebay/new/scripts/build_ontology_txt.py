import pandas as pd
import owlready2 as o2
import argparse
import os
import re
import types
import math
from copy import deepcopy
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert taxonomy to ontology.")

    # Required input argument
    parser.add_argument(
        'input',
        type=str,
        help='The path of taxonomy to read from, expected txt'
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

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{size_names[i]}"
            
if __name__ == "__main__":
    
    args = parse_arguments()
    data = pd.read_csv(args.input)

    # Verify the existence of Category Friendly Name (CFN) for each category
    data["HasCFN"] = data["CFN"].notna()
    # Map site codes to conventional region abbreviations
    site_codes = {0: 'US', 2: 'CA', 3: 'UK', 15: 'AU', 16: 'AT', 23: 'BEFR', 71: 'FR', 77: 'DE', 100: 'EM', 101: 'IT', 123: 'BENL', 146: 'NL', 186: 'ES', 193: 'CH', 201: 'HK', 205: 'IE', 207: 'MY', 210: 'FRCA', 211: 'PH', 212: 'PL', 216: 'SG'}
    
    onto = o2.get_ontology(f"https://ebay.com/ontology")

    # Define the annotation properties
    with onto:

        class catid(o2.AnnotationProperty):
            pass
        class site(o2.AnnotationProperty):
            pass
        class level(o2.AnnotationProperty):
            pass
        class isLeaf(o2.AnnotationProperty):
            pass
        class breadcrumb(o2.AnnotationProperty):
            pass
        class meta(o2.AnnotationProperty):
            pass
        class isTest(o2.AnnotationProperty):
            pass
        class isOther(o2.AnnotationProperty):
            pass
    
    sites = data["Site ID"].unique()
    with tqdm(total=len(sites), desc="Total progress", position=0) as outer:
        for site in sites:

            namespace = onto.get_namespace(f"https://ebay.com/ontology/{site}")
            site_data = deepcopy(data[data['Site ID'] == site])

            with onto:
                site_root = types.new_class(str(site), (o2.Thing,))
                site_root.label = f'{site} - {site_codes[int(site)]}'

            with namespace:
                with tqdm(total=len(site_data), desc=f"site {site} - {site_codes[int(site)]}", position=1, leave=False) as inner:
                    for _,row in site_data.iterrows():
                        catid = str(row['Cat. ID'])
                        name = row['CFN'] if row['HasCFN'] else row['Cat. Name']
                        breadcrumb = row['ID Breadcrumb'].split(' > ')
                        level = len(breadcrumb)
                        parent = breadcrumb[-2] if level > 1 else '0'
                        newclass = types.new_class(catid, (namespace[parent] if parent != '0' else site_root,))
                        newclass.label = name
                        newclass.catid = int(catid)
                        newclass.site = int(site)
                        newclass.level = level
                        newclass.isLeaf = True if row['Parent/Leaf'] == 'Leaf' else False
                        newclass.breadcrumb = row['Breadcrumb']
                        newclass.meta = row['Meta']
                        newclass.isTest = row['Test Cat.']
                        newclass.isOther = row['Other']
                        inner.update()
            outer.update()

    onto.save(file = args.output, format = "rdfxml")
    size = convert_size(os.path.getsize(args.output))
    print(f'Success. Ontology saved at {args.output} ({size}, {len(data) + len(sites) + 1} classes)')