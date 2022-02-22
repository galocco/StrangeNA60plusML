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
from ROOT import TFile, gROOT, TDatabasePDG, TH1D, TCanvas, TH1D
import ROOT
import pandas as pd

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-p', '--presel', help='Apply preselection efficiency', action='store_true')
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
PRESELECTION = params['PRESELECTION']
PRESEL = args.presel
###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_path = os.path.expandvars(params['DATA_PATH'])
    
results_dir = os.environ['RESULTS']+"/"+FILE_PREFIX

###############################################################################
start_time = time.time()                          # for performances evaluation
file_name = results_dir + f'/{FILE_PREFIX}_features.root'
results_file = TFile(file_name, 'recreate')


pd.set_option("display.precision", 10)

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
df_bkg = uproot.open(bkg_path)['ntcand'].arrays(library='pd',entry_stop=400000).query("true < 0.5 and cosp > 0.9999 and mD > 1.110683 and mD < 1.120683")
df_skg = uproot.open(data_path)['ntcand'].arrays(library='pd',entry_stop=20000000).query("true > 0.5 and cosp > 0.9999 and mD > 1.110683 and mD < 1.120683")
df_sig = uproot.open(signal_path)['ntcand'].arrays(library='pd',entry_stop=400000).query("cosp > 0.9999 and mD > 1.110683 and mD < 1.120683")
if PRESEL:
    df_bkg = df_bkg.query(PRESELECTION)
    df_skg = df_skg.query(PRESELECTION)
    df_sig = df_sig.query(PRESELECTION)
if "thetad" in df_bkg.columns:
    df_sig.rename(columns={"thetad": "costhetad"}, inplace=True)
    df_bkg.rename(columns={"thetad": "costhetad"}, inplace=True)
    df_sig = df_sig.query("costhetad > -1 and costhetad < 1")
    df_bkg = df_bkg.query("costhetad > -1 and costhetad < 1")
    df_sig['thetad'] = df_sig.apply(lambda x: math.acos(x['costhetad'] if x['costhetad']<1 or x['costhetad']>-1 else -1), axis=1)
    df_bkg['thetad'] = df_bkg.apply(lambda x: math.acos(x['costhetad'] if x['costhetad']<1 or x['costhetad']>-1 else -1), axis=1)

nbins = 4000
cv = TCanvas("cv","cv")
ROOT.gStyle.SetOptStat(0)
for item1 in df_bkg.columns.tolist():
    for item2 in df_sig.columns.tolist():
        if item1==item2:

            legend = ROOT.TLegend(0.8,0.8,1,1)
            features_dir = results_file.mkdir(item1)
            features_dir.cd()
            sig_max_val = df_sig[item1].max()
            sig_min_val = df_sig[item1].min()
            min_val = sig_min_val
            max_val = sig_max_val
            if 'cosp' in item1:
                min_val = 0.9999
                max_val = 1
                nbins = 5000
            counts_sig, _ = np.histogram(df_sig[item1], nbins, range=[min_val,max_val])
            hist_sig = TH1D('hist_sig'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            counts_bkg, _ = np.histogram(df_bkg[item1], nbins, range=[min_val,max_val])
            hist_bkg = TH1D('hist_bkg_'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            counts_skg, _ = np.histogram(df_skg[item1], nbins, range=[min_val,max_val])
            hist_skg = TH1D('hist_skg_'+item1, ';'+item1+';pdf', nbins, min_val, max_val)
            for index in range(0, nbins):
                hist_sig.SetBinContent(index + 1, counts_sig[index]/sum(counts_sig))
                hist_sig.SetBinError(index + 1, math.sqrt(counts_sig[index])/sum(counts_sig))
                hist_bkg.SetBinContent(index + 1, counts_bkg[index]/sum(counts_bkg))
                hist_bkg.SetBinError(index + 1, math.sqrt(counts_bkg[index])/sum(counts_bkg))
                hist_skg.SetBinContent(index + 1, counts_skg[index]/sum(counts_skg))
                hist_skg.SetBinError(index + 1, math.sqrt(counts_skg[index])/sum(counts_skg))

            max_hist = max(max(counts_bkg)/sum(counts_bkg), max(counts_sig)/sum(counts_sig))*1.5

            hist_bkg.SetLineColor(ROOT.kRed)
            hist_sig.SetLineColor(ROOT.kBlue)
            hist_skg.SetLineColor(ROOT.kGreen)
            legend.AddEntry(hist_sig,"signal","l")
            legend.AddEntry(hist_skg,"signal from bkg","l")
            legend.AddEntry(hist_bkg,"background","l")
            hist_bkg.Write()
            hist_sig.Write()
            hist_skg.Write()
            cv.SetName("cv_"+item1)
            cv.SetLogy(0)
            hist_bkg.GetYaxis().SetRangeUser(0.00001,max_hist)
            hist_bkg.Draw("")
            hist_sig.Draw("SAME")
            hist_skg.Draw("SAME")
            legend.Draw()
            cv.Write()
            cv.SetName("cv_"+item1+"_log")
            cv.SetLogy()
            cv.Write()

            
        
            
print('')
print(f'--- features comparison in {((time.time() - start_time) / 60):.2f} minutes ---')