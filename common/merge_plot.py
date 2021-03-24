#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import time
import warnings

import numpy as np
import yaml

from ROOT import TH1D, TFile, gROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
###############################################################################

###############################################################################
# define analysis global variables
N_BODY = params['NBODY']
FILE_PREFIX = params['FILE_PREFIX']
DATA_LIST = params['DATA_BKG_PATH']
CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)


###############################################################################
# define paths for loading results
results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]
input_file_list = []
for index in range(0,len(DATA_LIST)):
    input_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_{index}'+'.root'
    input_file_list.append(TFile(input_file_name, 'read'))

output_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_merged.root'
output_file = TFile(output_file_name, 'recreate')


###############################################################################
for cclass in CENT_CLASSES:
    cent_dir_name = f'{cclass[0]}-{cclass[1]}'
    cent_dir = output_file.mkdir(cent_dir_name)
    cent_dir.cd()

    #all file should have the same number of events
    PreselEff = 0
    for input_file in input_file_list:
        if PreselEff==0:
            PreselEff = input_file.Get(f'{cent_dir_name}/PreselEff')
        else:
            PartialEff = input_file.Get(f'{cent_dir_name}/PreselEff')
            PreselEff.Add(PartialEff)
    PreselEff.Scale(1./len(DATA_LIST))
    PreselEff.Write()
    
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
            # get the dir where the inv mass histo are
            input_subdir_list = []
            for input_file in input_file_list:
                subdir_name = f'pt_{ptbin[0]}{ptbin[1]}'
                input_subdir_list.append(input_file.Get(f'{cent_dir_name}/{subdir_name}'))
            
            #print(input_subdir_list)
            # create the subdir in the output file
            output_subdir = cent_dir.mkdir(subdir_name)
            output_subdir.cd()
            # loop over all the histo in the dir
            for key_m in input_subdir_list[0].GetListOfKeys():
                hist = TH1D(key_m.ReadObj())
                print(hist.GetName())
                for input_subdir in input_subdir_list[1:len(DATA_LIST)]:
                    hist_part = input_subdir.Get(hist.GetName())
                    hist.Add(hist_part)
                hist.Write()
output_file.Close()