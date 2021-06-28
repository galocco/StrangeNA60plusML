#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import time
import warnings
import math
import numpy as np
import yaml

import analysis_utils as au
import plot_utils as pu

from ROOT import TH1D, TFile, gROOT, TDatabasePDG, TF1

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-m', '--merged', help='Run on the merged histograms', action='store_true')
parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
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
SIGMA = params['SIGMA']
EFF = params['EFF']
EINT = pu.get_sNN(params['EINT'])
GAUSS = params['GAUSS']
PT_BINS = params['PT_BINS']
T = params['T']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)
BKG_MODELS = params['BKG_MODELS']

SPLIT_MODE = args.split
PEAK_MODE = args.peak
MERGED = args.merged
if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

LABELS = [f'{x:.2f}_{y}' for x in FIX_EFF_ARRAY for y in BKG_MODELS]

###############################################################################
# define paths for loading results
results_dir = os.environ['HYPERML_RESULTS_{}'.format(N_BODY)]

input_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_merged.root' if MERGED else results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results.root'
input_file = TFile(input_file_name, 'read')

output_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_BS.root'
output_file = TFile(output_file_name, 'recreate')

###############################################################################

###############################################################################
# start the actual signal extraction
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
pt_spectrum = TF1("fpt", "x*exp(-TMath::Sqrt(x**2+[0]**2)/[1])", 0, 100)
pt_spectrum.FixParameter(0, mass)
pt_spectrum.FixParameter(1, T)
n_events = 0
for index in range(0, len(params['EVENT_PATH'])):
    event_path = os.path.expandvars(params['EVENT_PATH'][index])
    background_file = TFile(event_path)
    hist_ev = background_file.Get('hNevents')
    n_events += hist_ev.GetBinContent(1)
nsigma = 3
print("n_events: ",n_events)
for split in SPLIT_LIST:
    cent_dir_name = f'0-5{split}'
    #cent_dir = output_file.mkdir(cent_dir_name)
    #cent_dir.cd()

    h1_eff = input_file.Get(cent_dir_name + '/PreselEff')
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        ptbin_index = au.get_ptbin_index(h1_eff, ptbin)
        # get the dir where the inv mass histo are
        subdir_name = f'pt_{ptbin[0]}{ptbin[1]}'
        input_subdir = input_file.Get(f'{cent_dir_name}/{subdir_name}')
        eff_presel = h1_eff.GetBinContent(ptbin_index)
        # create the subdir in the output file
        #output_subdir = cent_dir.mkdir(subdir_name)
        #output_subdir.cd()
        nbins = int((EFF_MAX-EFF_MIN)/EFF_STEP)
        hist_BS = TH1D("hist_BS_" + subdir_name, ";BDT efficiency;B/S", nbins, EFF_MAX, EFF_MIN)
        #hist_BS.GetXaxis().SetNdivisions(10)
        hist_BS_efftot = TH1D("hist_BS_efftot_" + subdir_name, ";BDT efficiency;B/S", nbins, EFF_MAX*eff_presel, EFF_MIN*eff_presel)
        #hist_BS_efftot.GetXaxis().SetNdivisions(10)
        hist_Sgn = TH1D("hist_Sgn_" + subdir_name, ";BDT efficiency;Sign/#sqrt{n_{ev}}", nbins, EFF_MAX , EFF_MIN)
        #hist_Sgn.GetXaxis().SetNdivisions(10)
        hist_Sgn_efftot = TH1D("hist_Sgn_efftot_" + subdir_name, ";BDT efficiency;Sign/#sqrt{n_{ev}}", nbins, EFF_MAX*eff_presel, EFF_MIN*eff_presel)
        #hist_BS_efftot.GetXaxis().SetNdivisions(10)

        print("eff presel: ", eff_presel)
        print("multiplicity: ", MULTIPLICITY)
        print("b-ratio: ", BRATIO)
        # loop over all the histo in the dir
        for key in input_subdir.GetListOfKeys():
            keff = key.GetName()[-4:]
            #for latter in reversed(key.GetName()[-4:]):
            hist = TH1D(key.ReadObj())
            mass_range = [mass - nsigma*SIGMA, mass + nsigma*SIGMA]
            bkg_counts = 0
            for index in np.arange(mass_range[0], mass_range[1], hist.GetBinWidth(1)):
                bkg_counts += hist.GetBinContent(hist.GetXaxis().FindBin(index))

            pt_frac = 1# pt_spectrum.Integral(ptbin[0], ptbin[1], 1e-8) / pt_spectrum.Integral(0, 100, 1e-8)
            #ct_frac = ct_spectrum.Integral(ctbin[0], ctbin[1], 1e-8) / ct_spectrum.Integral(0, 100, 1e-8)

            print("pt_frac: ", eff_presel)
            sig_counts = MULTIPLICITY*BRATIO*n_events*EFF#*pt_frac#*float(keff)#*ct_frac
            print("eff: ",keff," sig: ",sig_counts," bkg: ",bkg_counts," B/S",round(bkg_counts/sig_counts,1)," bin: ",hist_BS.GetXaxis().FindBin(keff))
            hist_BS.SetBinContent(hist_BS.GetXaxis().FindBin(keff),bkg_counts/sig_counts)
            hist_BS_efftot.SetBinContent(hist_BS.GetXaxis().FindBin(keff),bkg_counts/sig_counts)
            hist_Sgn.SetBinContent(hist_BS.GetXaxis().FindBin(keff),sig_counts/math.sqrt(sig_counts+bkg_counts)/math.sqrt(n_events))
            hist_Sgn_efftot.SetBinContent(hist_BS.GetXaxis().FindBin(keff),sig_counts/math.sqrt(sig_counts+bkg_counts)/math.sqrt(n_events))

        output_file.cd()
        hist_BS.Write()
        hist_BS_efftot.Write()
        hist_Sgn.Write()
        hist_Sgn_efftot.Write()


output_file.Close()
