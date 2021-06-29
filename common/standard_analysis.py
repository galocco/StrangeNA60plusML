#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array

import numpy as np
import yaml
import math
import analysis_utils as au
import plot_utils as pu
import pandas as pd

import ROOT
from ROOT import TFile, gROOT, TDatabasePDG, TF1, TH1D
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
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
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
GAUSS = params['GAUSS']
MASS_WINDOW = params['MASS_WINDOW']
PT_BINS = params['PT_BINS']

PRESELECTION = params['PRESELECTION']

PEAK_MODE = args.peak


for index in range(0,len(params['EVENT_PATH'])):

    ###############################################################################
    # define paths for loading data
    data_path = os.path.expandvars(params['DATA_PATH'][index])
    event_path = os.path.expandvars(params['EVENT_PATH'][index])
    BKG_MODELS = params['BKG_MODELS']

    results_dir = f"../Results/2Body"

    ###############################################################################
    start_time = time.time()                          # for performances evaluation

    resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

    file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_std_results.root'
    results_file = TFile(file_name, 'recreate')

    mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
    cv = ROOT.TCanvas("cv","cv")


    hnsparse = au.get_skimmed_large_data_std_hsp(mass, data_path, PT_BINS, PRESELECTION, MASS_WINDOW)
    results_file.cd()
    hnsparse.Write()
    cent_dir = results_file.mkdir('0-5')
    cent_dir.cd()

    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        sub_dir = cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
        sub_dir.cd()
        h1_minv = au.h1_from_sparse_std(hnsparse, ptbin, f'pt_{ptbin[0]}{ptbin[1]}')
        h1_minv.Write()

    results_file.Close()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

