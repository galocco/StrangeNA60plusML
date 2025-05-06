import yaml
import sys
import ROOT
import os

def update_nevents(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        content = file.read()

    # Remove python-specific tag for tuples
    content = content.replace("!!python/tuple", "")
    data = yaml.safe_load(content)

    root_file = ROOT.TFile(data["INFO_PATH"])
    nev = root_file.Get("hNevents").GetBinContent(1)
    root_file.Close()
    print()
    # Modify the NEVENTS value
    if 'NEVENTS' in data:
        print(f"Old NEVENTS value: {data['NEVENTS']}")
        data['NEVENTS'] = nev
    else:
        print("NEVENTS key not found. Adding it.")
        data['NEVENTS'] = nev

    # Write the updated data back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)



def find_yaml_files(directory, recursive=True):
    yaml_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    yaml_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(directory, file))

    return yaml_files

# Usage: python script.py config.yaml 1000
if __name__ == "__main__":
    directory = "../Config"
    yaml_files = find_yaml_files(directory, recursive=True)
    print(f"Found {len(yaml_files)} YAML files in {directory}.")
    for yaml_file in yaml_files:
        print(f"Processing {yaml_file}")
        update_nevents(yaml_file)