#!/usr/bin/env python3
#macro to compute the shift of the gaussian mu parameter due to reconstruction and BDT selection
import argparse
import os
import time
import warnings
import math
import numpy as np
import yaml
import uproot

import analysis_utils as au
import pandas as pd
import xgboost as xgb
from analysis_classes import (ModelApplication, TrainingAnalysis)
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT, TF1, TH1D, TH2D, TCanvas, TLegend, TDatabasePDG

from array import*

hyp3mass = 2.99131

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-f', '--full', help='Run with the full simulation', action='store_true')
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
EINT = params['EINT']
EFF = params['EFF']

CENT_CLASSES = params['CENTRALITY_CLASS']
CT_BINS = params['CT_BINS']
PT_BINS = params['PT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']
PRESELECTION = params['PRESELECTION']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)
LARGE_DATA = params['LARGE_DATA']
FULL_SIM = args.full
SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data

signal_path = os.path.expandvars(params['MC_PATH'])
if FULL_SIM:
    bkg_path = os.path.expandvars(params['BKG_PATH'])
else:
    bkg_path = os.path.expandvars(params['BKG_PATH'])

data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'][0])
if FULL_SIM:
    data_sig_path = os.path.expandvars(params['DATA_PATH'][0])
    event_path = os.path.expandvars(params['EVENT_PATH_FULL'][0])
else:
    data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'][0])
    event_path = os.path.expandvars(params['EVENT_PATH'][0])

handlers_path = '../Models/2Body/handlers'
###############################################################################

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name =  '../Results/2Body/' + FILE_PREFIX + '/' + FILE_PREFIX + '_mass_shaping.root'
results_file = TFile(file_name,"recreate")

file_name = '../Results/2Body/' + FILE_PREFIX + '/' + FILE_PREFIX + '_results_fit.root'
eff_file = TFile(file_name, 'read')

results_file.cd()
#efficiency from the significance scan

SEL_EFF = []
gauss = TF1('gauss','gaus')
histos = []
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

for split in SPLIT_LIST:

    application_columns = ['score', 'm', 'y', 'pt', 'ct','centrality']
    if LARGE_DATA:
        df_skimmed = au.get_skimmed_large_data(MULTIPLICITY, BRATIO, EFF, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split, FILE_PREFIX, PRESELECTION)
        ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, False, PRESELECTION, df_skimmed)
    else:
        ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, False, PRESELECTION)
    
    shift_bin = 1
    eff_index=0
    histo_split = []

    for cclass in CENT_CLASSES:
        cent_dir_histos = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):

                if LARGE_DATA:
                    df_applied = ml_application.get_data_slice(cclass, ptbin, ctbin, application_columns)
                else: 
                    df_applied = ml_application.apply_BDT_to_data(model_handler, cclass, ptbin, ctbin, model_handler.get_training_columns(), application_columns)

                # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                #data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)

                data_range = f'{ptbin[0]}<pt<{ptbin[1]}'#' and {cent_class[0]}<=centrality<{cent_class[1]}'

                input_model = xgb.XGBClassifier()
                model_handler = ModelHandler(input_model)
                
                info_string = f'_{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'
                filename_handler = handlers_path + '/model_handler_' +FILE_PREFIX+ info_string + '.pkl'
                model_handler.load_model_handler(filename_handler)
                eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split, FILE_PREFIX)
                mass_bins = 40
                # define subdir for saving invariant mass histograms
                sub_dir_histos = cent_dir_histos.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir_histos.cd()
                sig_dir_histos = sub_dir_histos.mkdir('sig')
                bkg_dir_histos = sub_dir_histos.mkdir('bkg')

                for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                    #after selection
                    mass_array = np.array(df_applied.query('score>@tsd and y<0.5')['m'].values, dtype=np.float64)
                    counts, bin = np.histogram(mass_array, bins=mass_bins, range=[mass*0.97, mass*1.03])
                    
                    bkg_dir_histos.cd()
                    histo_name = f"eff{eff:.2f}"+split
                    hbkg_sel = au.h1_invmass_ov(counts, cclass, ptbin, ctbin, hist_range=[mass*0.97, mass*1.03], bins=mass_bins, name=histo_name)
                    hbkg_sel.SetTitle(";m (GeV/#it{c}^{2});counts")
                    hbkg_sel.SetName(hbkg_sel.GetName()+"_bkg")
                    hbkg_sel.Write()

                    mass_array = np.array(df_applied.query('score>@tsd and y>0.5')['m'].values, dtype=np.float64)
                    counts, bin = np.histogram(mass_array, bins=mass_bins, range=[mass*0.97, mass*1.03])
                    
                    sig_dir_histos.cd()
                    histo_name = f"eff{eff:.2f}"+split
                    hsig_sel = au.h1_invmass_ov(counts, cclass, ptbin, ctbin, hist_range=[mass*0.97, mass*1.03], bins=mass_bins, name=histo_name)
                    hsig_sel.SetTitle(";m (GeV/#it{c}^{2});counts")
                    hsig_sel.SetName(hsig_sel.GetName()+"_sig")
                    hsig_sel.Write()
                
results_file.Close()