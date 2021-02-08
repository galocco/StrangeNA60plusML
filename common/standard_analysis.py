#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array

import numpy as np
import yaml

import analysis_utils as au
import plot_utils as pu
import pandas as pd

import ROOT
from ROOT import TFile, gROOT, TDatabasePDG
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
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

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
COLUMNS = params['TRAINING_COLUMNS']

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])

BKG_MODELS = params['BKG_MODELS']

results_dir = f"../Results/2Body"

###############################################################################
start_time = time.time()                          # for performances evaluation

file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
results_file = TFile(file_name, 'recreate')

file_name = results_dir + f'/{FILE_PREFIX}_mass_res.root'
shift_file = TFile(file_name, 'read')

standard_selection = 'cosp > 0.999999 and dist > 0.3'
application_columns = ['cosp', 'dist', 'dca', 'rapidity', 'd0prod', 'd01', 'd02', 'ptMin']

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
for split in SPLIT_LIST:

    ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split)
    #get the histogram with the mass shift
    shift_hist = shift_file.Get("fit_mean"+split)
    #initialize the histogram with the mass pol0 fit
    iBin = 0
    for cclass in CENT_CLASSES:
        cent_dir = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        df_applied = ml_application.df_data.query(standard_selection)
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                mass_bins = 40
                sub_dir = cent_dir.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir.cd()
                mass_array = np.array(df_applied.query("ct<@ctbin[1] and ct>@ctbin[0] and pt<@ptbin[1] and pt>@ptbin[0]")['m'].values, dtype=np.float64)
                counts, _ = np.histogram(mass_array, bins=mass_bins, range=[mass*0.97, mass*1.03])
                h1_minv = au.h1_invmass_ov(counts, cclass, ptbin, ctbin, hist_range = [mass*0.97, mass*1.03])

                for bkgmodel in BKG_MODELS:
                    # create dirs for models
                    fit_dir = sub_dir.mkdir(bkgmodel)
                    fit_dir.cd()
                    rawcounts, err_rawcounts, significance, err_significance, mu, mu_err, _, _ = au.fit_hist(h1_minv, cclass, ptbin, ctbin, mass, model=bkgmodel, mode=N_BODY, split=split, Eint = 17.3)
    results_file.cd()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')