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
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-f', '--full', help='Run with the full simulation', action='store_true')
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
N_BODY = params['NBODY']
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
MULTIPLICITY = params['MULTIPLICITY']
BRATIO = params['BRATIO']
EINT = pu.get_sNN(params['EINT'])
T = params['T']
EFF = params['EFF']
SIGMA = params['SIGMA']
GAUSS = params['GAUSS']

PT_BINS = params['PT_BINS']
COLUMNS = params['TRAINING_COLUMNS']

PRESELECTION = params['PRESELECTION']
SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

FULL_SIM = args.full
PEAK_MODE = args.peak


for index in range(0,len(params['EVENT_PATH']) if FULL_SIM else len(params['EVENT_PATH'])):

    ###############################################################################
    # define paths for loading data
    signal_path = os.path.expandvars(params['MC_PATH'])
    if FULL_SIM:
        bkg_path = os.path.expandvars(params['MC_PATH_FULL'])
    else:
        bkg_path = os.path.expandvars(params['BKG_PATH'])
    data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'][index])
    if FULL_SIM:
        data_sig_path = os.path.expandvars(params['DATA_PATH'][index])
        event_path = os.path.expandvars(params['EVENT_PATH_FULL'][index])
    else:
        data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'][index])
        event_path = os.path.expandvars(params['EVENT_PATH'][index])
    BKG_MODELS = params['BKG_MODELS']

    results_dir = f"../Results/2Body"

    ###############################################################################
    start_time = time.time()                          # for performances evaluation

    resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

    file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_std_results.root'
    results_file = TFile(file_name, 'recreate')

    standard_selection = 'pt > 0'
    application_columns = ['pt','m','ct','centrality','score','y','cosp']
    mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

    background_file = ROOT.TFile(event_path)
    hist_ev = background_file.Get('hNevents')
    n_ev = hist_ev.GetBinContent(1)
    background_file.Close()
    cv = ROOT.TCanvas("cv","cv")

    for split in SPLIT_LIST:

        if FULL_SIM:
            hnsparse = au.get_skimmed_large_data_std_full(mass, data_sig_path, PT_BINS, COLUMNS, N_BODY, split, FILE_PREFIX, PRESELECTION)
        else:
            hnsparse = au.get_skimmed_large_data_std(mass, MULTIPLICITY, BRATIO, EFF, data_sig_path, data_bkg_path, event_path, PT_BINS, COLUMNS, N_BODY, split, FILE_PREFIX, PRESELECTION)

        #ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, split, FULL_SIM, PRESELECTION, hnsparse)
        results_file.cd()
        hnsparse.Write()
        cent_dir = results_file.mkdir(f'0-5{split}')
        cent_dir.cd()

        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            mass_bins = 40
            sub_dir = cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
            sub_dir.cd()
            h1_minv = au.h1_from_sparse_std(hnsparse, ptbin, f'pt_{ptbin[0]}{ptbin[1]}')
            h1_minv.Write()

    results_file.Close()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

