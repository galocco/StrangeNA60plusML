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

LARGE_DATA = params['LARGE_DATA']

PRESELECTION = params['PRESELECTION']

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
#if BOTH:
bkg_path_full = os.path.expandvars(params['MC_PATH_FULL'])
bkg_path_prompt = os.path.expandvars(params['DATA_BKG_PATH'][0])
    
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']+"/"+FILE_PREFIX

###############################################################################

def counts_sum(counts):
    tot = 0
    summed_list = []
    for i in reversed(counts):
        tot += i
        summed_list.insert(0, tot)
    return summed_list

###############################################################################
start_time = time.time()                          # for performances evaluation
file_name = results_dir + f'/{FILE_PREFIX}_effciency.root'
results_file = TFile(file_name, 'recreate')

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
df_bkg_full = uproot.open(bkg_path_full)['ntcand'].arrays(library='pd',entry_stop=4000000).query("true < 0.5")
df_bkg_prompt = uproot.open(bkg_path_prompt)['ntcand'].arrays(library='pd',entry_stop=4000000)
df_sig = uproot.open(signal_path)['ntcand'].arrays(library='pd',entry_stop=4000000)
nbins = 200
cv = TCanvas("cv","cv")
ROOT.gStyle.SetOptStat(0)
for item1 in df_bkg_prompt.columns.tolist():
    for item2 in df_sig.columns.tolist():
        if item1==item2:
            if item1 == "centrality":
                continue

            legend = ROOT.TLegend(0.8,0.8,1,1)
            features_dir = results_file.mkdir(item1)
            features_dir.cd()
            sig_max_val = df_sig[item1].max()
            sig_min_val = df_sig[item1].min()
            min_val = sig_min_val
            max_val = sig_max_val
            
            counts_gen_sig, _ = np.histogram(df_sig[item1], nbins, range=[min_val,max_val])
            counts_gen_bkg_full, _ = np.histogram(df_bkg_full[item1], nbins, range=[min_val,max_val])
            counts_gen_bkg_prompt, _ = np.histogram(df_bkg_prompt[item1], nbins, range=[min_val,max_val])

            counts_rec_sig, _ = np.histogram(df_sig[item1], nbins, range=[min_val,max_val])            
            hist_eff_sig = TH1D('hist_eff_sig'+item1, ';'+item1+' selection ;efficiency', nbins, min_val, max_val)

            counts_rec_bkg_full, _ = np.histogram(df_bkg_full[item1], nbins, range=[min_val,max_val])
            hist_eff_bkg_full = TH1D('hist_eff_bkg_full_'+item1, ';'+item1+' selection ;efficiency', nbins, min_val, max_val)

            counts_rec_bkg_prompt, _ = np.histogram(df_bkg_prompt[item1], nbins, range=[min_val,max_val])
            hist_eff_bkg_prompt = TH1D('hist_eff_bkg_prompt_'+item1, ';'+item1+' selection ;efficiency', nbins, min_val, max_val)

            counts_rec_sig = counts_sum(counts_rec_sig)
            counts_rec_bkg_full = counts_sum(counts_rec_bkg_full)
            counts_rec_bkg_prompt = counts_sum(counts_rec_bkg_prompt)

            for index in range(0, nbins):
                eff_sig = counts_rec_sig[index]/sum(counts_gen_sig)
                hist_eff_sig.SetBinContent(index + 1, eff_sig)
                hist_eff_sig.SetBinError(index + 1, math.sqrt(eff_sig*(1-eff_sig)/sum(counts_gen_sig)))

                eff_bkg_prompt = counts_rec_bkg_prompt[index]/sum(counts_gen_bkg_prompt)

                hist_eff_bkg_prompt.SetBinContent(index + 1, eff_bkg_prompt)
                hist_eff_bkg_prompt.SetBinError(index + 1, math.sqrt(eff_bkg_prompt*(1-eff_bkg_prompt)/sum(counts_gen_bkg_prompt)))

                eff_bkg_full = counts_rec_bkg_full[index]/sum(counts_gen_bkg_full)
                hist_eff_bkg_full.SetBinContent(index + 1, eff_bkg_full)
                hist_eff_bkg_full.SetBinError(index + 1, math.sqrt(eff_bkg_full*(1-eff_bkg_full)/sum(counts_gen_bkg_full)))

            hist_eff_bkg_full.SetLineColor(ROOT.kRed)
            hist_eff_sig.SetLineColor(ROOT.kBlue)
            hist_eff_bkg_prompt.SetLineColor(ROOT.kGreen)
            legend.AddEntry(hist_eff_sig,"signal","l")
            legend.AddEntry(hist_eff_bkg_prompt,"background prompt","l")
            legend.AddEntry(hist_eff_bkg_full,"background full","l")
            hist_eff_bkg_full.Write()
            hist_eff_bkg_prompt.Write()
            hist_eff_sig.Write()
            cv.SetName("cv_"+item1)
            cv.SetLogy(0)
            hist_eff_bkg_full.GetYaxis().SetRangeUser(0,1.1)
            hist_eff_bkg_full.Draw("")
            hist_eff_sig.Draw("SAME")
            hist_eff_bkg_prompt.Draw("SAME")
            legend.Draw()
            cv.Write()
            
        
            
print('')
print(f'--- features comparison in {((time.time() - start_time) / 60):.2f} minutes ---')