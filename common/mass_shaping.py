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

import analysis_utils as hau
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

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

LARGE_DATA = params['LARGE_DATA']

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']

handlers_path = '../Models/2Body/handlers'
###############################################################################

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]
file_name =  '../Results/2Body/' + params['FILE_PREFIX'] + '_mass_shaping.root'
results_file = TFile(file_name,"recreate")

file_name = '../Results/2Body/' + params['FILE_PREFIX'] + '_results_fit.root'
eff_file = TFile(file_name, 'read')

results_file.cd()
#efficiency from the significance scan

SEL_EFF = []
gauss = TF1('gauss','gaus')
histos = []
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

for split in SPLIT_LIST:
    if LARGE_DATA:
        if split== '_antimatter':
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df(entrystop=5000000).query("ArmenterosAlpha<0")[COLUMNS+['m']]
        elif split == '_matter':
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df(entrystop=5000000).query("ArmenterosAlpha>0")[COLUMNS+['m']]
        else:
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df(entrystop=5000000)[COLUMNS+['m']]
    else:
        if split== '_antimatter':
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df().query("ArmenterosAlpha<0")[COLUMNS+['m']]
        elif split == '_matter':
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df().query("ArmenterosAlpha>0")[COLUMNS+['m']]
        else:
            df_LS = uproot.open(data_bkg_path)["ntcand"].pandas.df()[COLUMNS+['m']]
        
    if 'ct' in params['FILE_PREFIX']:
        BINS = params['CT_BINS']
    else:
        BINS = params['PT_BINS']

    binning = array('d',BINS)
    ml_analysis = TrainingAnalysis(PDG_CODE, N_BODY, signal_path, bkg_path, split)
                    
    application_columns = ['score', 'm', 'ct', 'pt', 'centrality','ArmenterosAlpha']
    ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, False)

    shift_bin = 1
    eff_index=0
    histo_split = []

    for cclass in CENT_CLASSES:
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):

                # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)

                input_model = xgb.XGBClassifier()
                model_handler = ModelHandler(input_model)
                
                info_string = f'_{cclass[0]}{cclass[1]}_{ptbin[0]}{ptbin[1]}_{ctbin[0]}{ctbin[1]}{split}'
                filename_handler = handlers_path + '/model_handler' + info_string + '.pkl'
                model_handler.load_model_handler(filename_handler)
                eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split)
                del ml_application
                del ml_analysis
                y_pred = model_handler.predict(df_LS)
                df_LS.insert(0, 'score', y_pred)

                mass_bins = 40 if ctbin[1] < 16 else 36
        
                
                for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                    #after selection
                    mass_array = np.array(df_LS.query('score>@tsd')['m'].values, dtype=np.float64)
                    counts, roba = np.histogram(mass_array, bins=mass_bins, range=[mass*0.97, mass*1.03])
                    
                    histo_name = f"eff{eff:.2f}"+split
                    h1_sel = hau.h1_invmass_ov(counts, cclass, ptbin, ctbin, hist_range=[mass*0.97, mass*1.03], bins=mass_bins, name=histo_name)
                    h1_sel.SetTitle(";m (GeV/#it{c}^{2});counts")
                    histo_split.append(h1_sel)
    histos.append(histo_split)
                
results_file.cd()
for i in range(0,len(histos[0])):
    #histos[0][i].Add(histos[1][i])
    histos[0][i].SetName(histos[0][i].GetName().replace('_matter',''))
    histos[0][i].Write()