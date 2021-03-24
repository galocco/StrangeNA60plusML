#!/usr/bin/env python3
import argparse
import math
import os
import time
import warnings

import plot_utils as pu
import numpy as np
import uproot
import yaml
from ROOT import TFile, gROOT, TF1, TDatabasePDG, TH1D, TCanvas, gStyle, gSystem, TH1D
import ROOT

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--full', help='Run with the full simulation', action='store_true')
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

LARGE_DATA = params['LARGE_DATA']

PRESELECTION = params['PRESELECTION']

SPLIT_MODE = args.split
FULL_SIM = args.full
if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
if FULL_SIM:
    bkg_path = os.path.expandvars(params['BKG_PATH'])
else:
    bkg_path = os.path.expandvars(params['MC_PATH_FULL'])
    
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']+"/"+FILE_PREFIX

###############################################################################
start_time = time.time()                          # for performances evaluation
file_name = results_dir + f'/{FILE_PREFIX}_features.root'
results_file = TFile(file_name, 'recreate')

df_bkg = uproot.open(bkg_path)['ntcand'].arrays(library='pd',entry_stop=4000000)
if FULL_SIM:
    df_bkg = df_bkg.query("true < 0.5")
df_sig = uproot.open(signal_path)['ntcand'].arrays(library='pd',entry_stop=4000000)

nbins = 500
cv = TCanvas("cv","cv")
for item1 in df_bkg.columns.tolist():
    for item2 in df_sig.columns.tolist():
        if item1==item2:
            if item1 == "centrality":
                continue

            legend = ROOT.TLegend(0.6,0.6,0.9,0.9)
            features_dir = results_file.mkdir(item1)
            features_dir.cd()
            bkg_max_val = df_bkg[item1].max()  
            bkg_min_val = df_bkg[item1].min()
            sig_max_val = df_sig[item1].max()
            sig_min_val = df_sig[item1].min()
            min_val = sig_min_val
            max_val = sig_max_val
            
            counts_sig, _ = np.histogram(df_sig[item1], nbins, range=[min_val,max_val])
            hist_sig = TH1D('hist_sig'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            counts_bkg, _ = np.histogram(df_bkg[item1], nbins, range=[min_val,max_val])
            hist_bkg = TH1D('hist_bkg'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            for index in range(0, nbins):
                hist_sig.SetBinContent(index + 1, counts_sig[index]/sum(counts_sig))
                hist_sig.SetBinError(index + 1, math.sqrt(counts_sig[index])/sum(counts_sig))
                hist_bkg.SetBinContent(index + 1, counts_bkg[index]/sum(counts_bkg))
                hist_bkg.SetBinError(index + 1, math.sqrt(counts_bkg[index])/sum(counts_bkg))

            max_hist = max(max(counts_bkg)/sum(counts_bkg), max(counts_sig)/sum(counts_sig))*1.5

            hist_bkg.SetLineColor(ROOT.kRed)
            hist_sig.SetLineColor(ROOT.kBlue)
            legend.AddEntry(hist_sig,"signal","l")
            legend.AddEntry(hist_bkg,"background","l")
            hist_bkg.Write()
            hist_sig.Write()
            cv.SetName("cv_"+item1)
            cv.SetLogy(0)
            hist_bkg.GetYaxis().SetRangeUser(0.0001,max_hist)
            hist_bkg.Draw("")
            hist_sig.Draw("SAME")
            cv.Write()
            cv.SetName("cv_"+item1+"_log")
            cv.SetLogy()
            cv.Write()

            
        
            
print('')
print(f'--- features comparison in {((time.time() - start_time) / 60):.2f} minutes ---')