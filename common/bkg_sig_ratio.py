#!/usr/bin/env python3
import argparse
import math
import os
import time
import warnings
from array import array
import analysis_utils as au
import plot_utils as pu
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import yaml
from analysis_classes import ModelApplication, TrainingAnalysis
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT, TF1, TDatabasePDG, TH1D, TCanvas

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--custom', help='Run with customized bdt efficiencies', action='store_true')
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

PT_BINS = params['PT_BINS']

COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

CUSTOM = args.custom

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']

###############################################################################
file_name = f"../Results/2Body/{FILE_PREFIX}_results_fit.root"
inDirName = '0-5'
input_file = TFile(file_name)
h1BDTEff = input_file.Get(f'{inDirName}/BDTeff')
if CUSTOM:
    print("insert custom bdt efficiencies:")
    best_sig_eff = []
    for index in range(0,len(PT_BINS)):
        best_sig_eff.append(input())
else:
    best_sig_eff = np.round(np.array(h1BDTEff)[1:-1], 2)
n_bkgpars = 3
nsigma = 3

binning = array('d',PT_BINS)
output_file = TFile(results_dir + '/' + FILE_PREFIX + '/' + FILE_PREFIX+'_signal_results.root',"RECREATE")
for model in BKG_MODELS:
    if 'pol' in str(model):
        n_bkgpars = int(model[3]) + 1
    elif 'expo' in str(model):
        n_bkgpars = 2
    else:
        print(f'Unsupported model {model}')

    hist_sig = TH1D(f'hist_sig_{model}', ';#it{p}_{T} (GeV/#it{c}); Significance',len(PT_BINS)-1,binning)
    hist_BS = TH1D(f'hist_BS_{model}', ';#it{p}_{T} (GeV/#it{c}); B/S',len(PT_BINS)-1,binning)
    hist_sigma = TH1D(f'hist_sigma_{model}', ';#it{p}_{T} (GeV/#it{c}); #sigma (GeV/#it{c}^{2})',len(PT_BINS)-1,binning)

    for index in range(0,len(best_sig_eff)):

        dir_name = f'{inDirName}/pt_{PT_BINS[index]}{PT_BINS[index+1]}/'
        obj_name = f'pT{PT_BINS[index]}{PT_BINS[index+1]}_eff{best_sig_eff[index]:.2f}'

        histo = input_file.Get(dir_name+f'{model}/'+obj_name)
        lineshape = histo.GetFunction("fitTpl")
        bkg_tpl = TF1('bkgTpl', f'{model}(0)', 0, 5)
        for parameter in range(0,n_bkgpars):
            bkg_tpl.SetParameter(parameter,lineshape.GetParameter(parameter))
        # get the fit parameters
        mu = lineshape.GetParameter(n_bkgpars+1)
        muErr = lineshape.GetParError(n_bkgpars+1)
        sigma = lineshape.GetParameter(n_bkgpars+2)
        sigmaErr = lineshape.GetParError(n_bkgpars+2)
        signal = lineshape.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
        errsignal = lineshape.GetParError(n_bkgpars) / histo.GetBinWidth(1)
        bkg = bkg_tpl.Integral(mu - nsigma * sigma, mu +
                                nsigma * sigma) / histo.GetBinWidth(1)

        if bkg > 0:
            errbkg = math.sqrt(bkg)
        else:
            errbkg = 0
        if signal > 0:
            errsignal = math.sqrt(signal)
        else:
            errsignal = 0
        # compute the significance
        if signal+bkg > 0:
            signif = signal/math.sqrt(signal+bkg)
            deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
            deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
            errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
        else:
            signif = 0
            errsignif = 0

        if signal > 0:
            ratio = bkg/signal
            ratioErr = math.sqrt((errbkg/bkg)**2+(errsignal/signal)**2)*ratio
        else:
            ratio = 0
            ratioErr = 0

        if math.isnan(sigmaErr):
            sigmaErr = 0
        hist_sig.SetBinContent(index+1, signif)
        hist_sig.SetBinError(index+1, errsignif)
        hist_sigma.SetBinContent(index+1, sigma)
        hist_sigma.SetBinError(index+1, sigmaErr)
        
        hist_BS.SetBinContent(index+1, ratio)
        hist_BS.SetBinError(index+1, ratioErr)
        
    output_file.cd()
    hist_sig.Write()
    hist_BS.Write()
    hist_sigma.Write()
output_file.Close()