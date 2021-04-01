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
parser.add_argument('-b', '--both', help='Features comparison with prompt and full background', action='store_true')
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
BOTH = args.both

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
#elif FULL_SIM:
#    bkg_path = os.path.expandvars(params['MC_PATH_FULL'])
    #bkg_path = os.path.expandvars()
#else:
#    bkg_path = os.path.expandvars(params['BKG_PATH'])
    
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']+"/"+FILE_PREFIX

###############################################################################
start_time = time.time()                          # for performances evaluation
file_name = results_dir + f'/{FILE_PREFIX}_features.root'
results_file = TFile(file_name, 'recreate')

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
df_bkg_full = uproot.open(bkg_path_full)['ntcand'].arrays(library='pd',entry_stop=4000000).query("true < 0.5")
df_bkg_prompt = uproot.open(bkg_path_prompt)['ntcand'].arrays(library='pd',entry_stop=4000000)

#    df_bkg = uproot.open(bkg_path)['ntcand'].arrays(library='pd',entry_stop=20000000)
#    if FULL_SIM:
#        df_bkg = df_bkg.query("true < 0.5")
        #df_bkg.drop('ct', axis='columns', inplace=True)
        #df_bkg['ct'] = df_bkg.apply(lambda x: x['dist']*mass/(x['pt']*math.sqrt(1+math.sinh(x['rapidity'])**2)), axis=1)
df_sig = uproot.open(signal_path)['ntcand'].arrays(library='pd',entry_stop=4000000)
    #df_sig.drop('ct', axis='columns', inplace=True)
    #df_sig['ct'] = df_sig.apply(lambda x: x['dist']*mass/(x['pt']*math.sqrt(1+math.sinh(x['rapidity'])**2)), axis=1)

#df_sig['ct_rem'] = df_sig.apply(lambda x: x['dist']*mass/(math.sqrt(x['pt']**2+(x['pt']**2+mass**2)*math.sinh(x['rapidity'])**2)), axis=1)
#df_bkg_full['ct_rem'] = df_bkg_full.apply(lambda x: x['dist']*mass/(math.sqrt(x['pt']**2+(x['pt']**2+mass**2)*math.sinh(x['rapidity'])**2)), axis=1)
#df_bkg_prompt['ct_rem'] = df_bkg_prompt.apply(lambda x: x['dist']*mass/(math.sqrt(x['pt']**2+(x['pt']**2+mass**2)*math.sinh(x['rapidity'])**2)), axis=1)

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
            
            counts_sig, _ = np.histogram(df_sig[item1], nbins, range=[min_val,max_val])
            hist_sig = TH1D('hist_sig'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            counts_bkg_full, _ = np.histogram(df_bkg_full[item1], nbins, range=[min_val,max_val])
            hist_bkg_full = TH1D('hist_bkg_full_'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            counts_bkg_prompt, _ = np.histogram(df_bkg_prompt[item1], nbins, range=[min_val,max_val])
            hist_bkg_prompt = TH1D('hist_bkg_prompt_'+item1, ';'+item1+';pdf', nbins, min_val, max_val)

            for index in range(0, nbins):
                hist_sig.SetBinContent(index + 1, counts_sig[index]/sum(counts_sig))
                hist_sig.SetBinError(index + 1, math.sqrt(counts_sig[index])/sum(counts_sig))
                hist_bkg_prompt.SetBinContent(index + 1, counts_bkg_prompt[index]/sum(counts_bkg_prompt))
                hist_bkg_prompt.SetBinError(index + 1, math.sqrt(counts_bkg_prompt[index])/sum(counts_bkg_prompt))
                hist_bkg_full.SetBinContent(index + 1, counts_bkg_full[index]/sum(counts_bkg_full))
                hist_bkg_full.SetBinError(index + 1, math.sqrt(counts_bkg_full[index])/sum(counts_bkg_full))

            max_hist = max(max(counts_bkg_full)/sum(counts_bkg_full), max(counts_sig)/sum(counts_sig))*1.5

            hist_bkg_full.SetLineColor(ROOT.kRed)
            hist_sig.SetLineColor(ROOT.kBlue)
            hist_bkg_prompt.SetLineColor(ROOT.kGreen)
            legend.AddEntry(hist_sig,"signal","l")
            legend.AddEntry(hist_bkg_prompt,"background prompt","l")
            legend.AddEntry(hist_bkg_full,"background full","l")
            hist_bkg_full.Write()
            hist_bkg_prompt.Write()
            hist_sig.Write()
            cv.SetName("cv_"+item1)
            cv.SetLogy(0)
            hist_bkg_full.GetYaxis().SetRangeUser(0.00001,max_hist)
            hist_bkg_full.Draw("")
            hist_sig.Draw("SAME")
            hist_bkg_prompt.Draw("SAME")
            legend.Draw()
            #cv.SaveAs(cv.GetName()+".png")
            cv.Write()
            cv.SetName("cv_"+item1+"_log")
            cv.SetLogy()
            #cv.SaveAs(cv.GetName()+".png")
            cv.Write()

            
        
            
print('')
print(f'--- features comparison in {((time.time() - start_time) / 60):.2f} minutes ---')