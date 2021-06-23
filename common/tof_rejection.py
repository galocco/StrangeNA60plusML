#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array

import numpy as np
import yaml
import math
import analysis_utils as au
import plot_utils as pu
import pandas as pd

import ROOT
from ROOT import TFile, gROOT, TDatabasePDG, TF1, TH1D
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-f', '--full', help='Run with the full simulation', action='store_true')
parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
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
GAUSS = params['GAUSS']

PT_BINS = params['PT_BINS']
COLUMNS = params['TRAINING_COLUMNS']

PRESELECTION = params['PRESELECTION']
SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

FULL_SIM = args.full
PEAK_MODE = args.peak

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

pt_spectrum = TF1("fpt","x*exp(-TMath::Sqrt(x**2+[0]**2)/[1])",0,100)
pt_spectrum.FixParameter(0,mass)
pt_spectrum.FixParameter(1,T)
pt_integral = pt_spectrum.Integral(0, 100, 1e-8)
###############################################################################
start_time = time.time()                          # for performances evaluation

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

file_name = resultsSysDir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_std_results.root'
input_file = TFile(file_name)

file_name = resultsSysDir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_BoverS.root'
results_file = TFile(file_name, 'recreate')


if FULL_SIM:
    data_sig_path = os.path.expandvars(params['DATA_PATH'][0])
    event_path = os.path.expandvars(params['EVENT_PATH_FULL'][0])
else:
    data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'][0])
    event_path = os.path.expandvars(params['EVENT_PATH'][0])

background_file = ROOT.TFile(event_path)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
background_file.Close()
bkg_func = TF1("bkg_func","pol2",1.,1.04)

binning = array('d', PT_BINS)

for split in SPLIT_LIST:
    hist_BS = TH1D("hist_BS"+split,";#it{p}_{T} (GeV/#it{c}); B/S", len(binning)-1, binning)
    hist_sgn = TH1D("hist_sgn"+split,";#it{p}_{T} (GeV/#it{c}); Sign/#sqrt{N_{ev}}", len(binning)-1, binning)
    results_file.cd()
    cent_dir = results_file.mkdir(f'0-5{split}')
    cent_dir.cd()
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        h1_minv = input_file.Get(f'0-5{split}/pt_{ptbin[0]}{ptbin[1]}/pt_{ptbin[0]}{ptbin[1]}')
        sig = MULTIPLICITY*n_ev*EFF*BRATIO*pt_spectrum.Integral(ptbin[0], ptbin[1], 1e-8)/pt_integral
        h1_minv.Fit(bkg_func,"MR+")
        bkg = bkg_func.Integral(mass-3*SIGMA,mass+3*SIGMA, 1e-8)/h1_minv.GetBinWidth(1)
        print('bkg: ', bkg, 'sig: ',sig)
        err_BoverS = bkg/sig*math.sqrt(1./bkg)
        index = PT_BINS.index(ptbin[0])+1
        hist_BS.SetBinContent(index, bkg/sig)
        hist_BS.SetBinError(index, err_BoverS)
        hist_sgn.SetBinContent(index, sig/math.sqrt((sig+bkg)*n_ev))
        hist_sgn.SetBinError(index, 0)
    hist_BS.Write()
    hist_sgn.Write()
input_file.Close()
results_file.Close()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

