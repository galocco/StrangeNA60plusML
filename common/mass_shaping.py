#!/usr/bin/env python3
#macro to compute the shift of the gaussian mu parameter due to reconstruction and BDT selection
import argparse
import os
import time
import warnings
import numpy as np
import yaml

import analysis_utils as au
import pandas as pd
import xgboost as xgb
from analysis_classes import (ModelApplication)
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT, TDatabasePDG

from array import*

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
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
MULTIPLICITY = params['MULTIPLICITY']
BRATIO = params['BRATIO']
MASS_WINDOW = params['MASS_WINDOW']
PT_BINS = params['PT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
PRESELECTION = params['PRESELECTION']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

###############################################################################
# define paths for loading data

data_path = os.path.expandvars(params['DATA_PATH'][0])
event_path = os.path.expandvars(params['EVENT_PATH'][0])

handlers_path = '../Models/handlers'
###############################################################################

resultsSysDir = os.environ['HYPERML_RESULTS']
file_name =  '../Results/' + FILE_PREFIX + '/' + FILE_PREFIX + '_mass_shaping.root'
results_file = TFile(file_name,"recreate")

file_name = '../Results/' + FILE_PREFIX + '/' + FILE_PREFIX + '_results_fit.root'
eff_file = TFile(file_name, 'read')

results_file.cd()
#efficiency from the significance scan

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

hnsparse_bkg = au.get_skimmed_large_data_hsp(mass, data_path, PT_BINS, COLUMNS, FILE_PREFIX, PRESELECTION + " and true < 0.5", MASS_WINDOW)
ml_application_bkg = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, event_path, hnsparse_bkg)

hnsparse_sig = au.get_skimmed_large_data_hsp(mass, data_path, PT_BINS, COLUMNS, FILE_PREFIX, PRESELECTION + " and true > 0.5", MASS_WINDOW)
ml_application_sig = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, event_path, hnsparse_sig)

shift_bin = 1
eff_index=0
histo_split = []
cent_dir_histos = results_file.mkdir('0-5')
for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test

    data_range = f'{ptbin[0]}<pt<{ptbin[1]}'

    input_model = xgb.XGBClassifier()
    model_handler = ModelHandler(input_model)
    
    info_string = f'_{ptbin[0]}{ptbin[1]}'
    filename_handler = handlers_path + '/model_handler_' +FILE_PREFIX+ info_string + '.pkl'
    model_handler.load_model_handler(filename_handler)
    eff_score_array, model_handler = ml_application_bkg.load_ML_analysis(ptbin, FILE_PREFIX)
    mass_bins = 40
    # define subdir for saving invariant mass histograms
    sub_dir_histos = cent_dir_histos.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
    sub_dir_histos.cd()
    sig_dir_histos = sub_dir_histos.mkdir('sig')
    bkg_dir_histos = sub_dir_histos.mkdir('bkg')

    for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
        #after selection
        bkg_dir_histos.cd()
        histo_name = f"eff{eff:.3f}"
        hbkg_sel = au.h1_from_sparse(hnsparse_bkg, ptbin, tsd, name=histo_name)
        hbkg_sel.Write()
        
        sig_dir_histos.cd()
        histo_name = f"eff{eff:.3f}"
        hsig_sel = au.h1_from_sparse(hnsparse_sig, ptbin, tsd, name=histo_name)
        hsig_sel.Write()
            
results_file.Close()