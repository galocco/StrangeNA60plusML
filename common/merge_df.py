#!/usr/bin/env python3
import argparse
import math
import os
import time
import warnings
import uproot
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
from ROOT import TFile, gROOT, TF1, TDatabasePDG, TH1D, TCanvas, RDF

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
MULTIPLICITY = params['MULTIPLICITY']
BRATIO = params['BRATIO']
EFF = params['EFF']
FILE_PREFIX = params['FILE_PREFIX']

###############################################################################
# define paths for loading data
data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])

###############################################################################
print("start merging signal and background dataframe")

background_file = TFile(event_path)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
nsig = int(MULTIPLICITY*EFF*n_ev*BRATIO)

df_sig = uproot.open(data_sig_path)["ntcand"].pandas.df(entrystop=nsig)
df_bkg = uproot.open(data_bkg_path)["ntcand"].pandas.df()
df_sig['y'] = 1
df_bkg['y'] = 0
df_data = pd.concat([df_sig, df_bkg])
FILE_PREFIX.replace("_analysis","")
rdf = RDF.MakeNumpyDataFrame(df_data)
results_file = TFile(f"../Data/{FILE_PREFIX}/merged_df.root")
rdf.Write()
results_file.Close()
print("dataframe merged")