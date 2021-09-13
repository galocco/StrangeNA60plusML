#!/usr/bin/env python3
import argparse
import os
import time
import warnings
import sys
import ruamel.yaml

import numpy as np
import uproot
import yaml

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config)) as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define analysis global variables
FILE_PREFIX = params['FILE_PREFIX']
COLUMNS = params['TRAINING_COLUMNS']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
###############################################################################
start_time = time.time()                          # for performances evaluation

df_sig = uproot.open(signal_path)['ntcand'].arrays(library='pd')

nbins = 200
eff_min = 0.99
preselection_cut = ''
for feature in COLUMNS:
    max_val = df_sig[feature].max()
    min_val = df_sig[feature].min()
    cut_min = min_val
    cut_max = max_val
    counts_sig, bin_edges = np.histogram(df_sig[feature], nbins, range=[min_val,max_val])
    #inverted list
    rev_counts_sig = counts_sig[::-1]
    sum_counts = sum(counts_sig)
    if feature == "dist":
        print(counts_sig)
        print(bin_edges)

    for index in range(len(counts_sig)-2, 0, -1):
        rev_counts_sig[index] += rev_counts_sig[index+1]
        print(rev_counts_sig[index]/sum_counts)
        #print(feature," < ",bin_edges[index+1]," eff: ", rev_counts_sig[index]/sum_counts)
        if (rev_counts_sig[index]/sum_counts) < eff_min:
            cut_max = bin_edges[index+1]
            break

    for index in range(2, len(counts_sig)):
        counts_sig[index] += counts_sig[index-1]
        print(rev_counts_sig[index]/sum_counts)        
        #print(feature," > ",bin_edges[index+1]," eff: ", rev_counts_sig[index]/sum_counts)
        if counts_sig[index]/sum_counts < eff_min:
            cut_min = bin_edges[index-1]
            break
    
    preselection_cut += feature + f'< {cut_max}  and '+feature+f' > {cut_min}'
    if feature is not COLUMNS[-1]:
        preselection_cut += ' and '
    print(cut_min, " < ", feature," < ", cut_max)

yaml = ruamel.yaml.YAML()
params['PRESELECTION'] = preselection_cut
with open(os.path.expandvars(args.config), "w") as stream:
    yaml.dump(params, stream)

print('')
print(f'--- features comparison in {((time.time() - start_time) / 60):.2f} minutes ---')