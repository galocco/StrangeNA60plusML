#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import time
import warnings

import numpy as np
import yaml

import analysis_utils as au
import plot_utils as pu

from ROOT import TH1D, TFile, gROOT, TDatabasePDG

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
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
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
EINT = pu.get_sNN(params['EINT'])
GAUSS = params['GAUSS']
PT_BINS = params['PT_BINS']
MASS_WINDOW = params['MASS_WINDOW']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

BKG_MODELS = params['BKG_MODELS']

PEAK_MODE = args.peak
MERGED = args.merged

LABELS = [f'{x:.3f}_{y}' for x in FIX_EFF_ARRAY for y in BKG_MODELS]

###############################################################################
# define paths for loading results
results_dir = os.environ['HYPERML_RESULTS']

input_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_merged.root' if MERGED else results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results.root'
input_file = TFile(input_file_name, 'read')

output_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_fit.root'
output_file = TFile(output_file_name, 'recreate')

###############################################################################
# define dictionaries for storing raw counts and significance
h1_rawcounts_dict = {}
significance_dict = {}

mean_fit = []
mean_fit_error = []
sigma_fit = []
sigma_fit_error = []
count=0
###############################################################################
# start the actual signal extraction
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
cent_dir_name = '0-5'
cent_dir = output_file.mkdir(cent_dir_name)
cent_dir.cd()

h1_eff = input_file.Get(cent_dir_name + '/PreselEff')
h1_BDT_eff = au.h1_rawcounts(PT_BINS, name = "BDTeff")

for lab in LABELS:
    h1_rawcounts_dict[lab] = au.h1_rawcounts(PT_BINS, suffix=lab)
    significance_dict[lab] = au.h1_significance(PT_BINS, suffix=lab)

for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
    ptbin_index = au.get_ptbin_index(h1_eff, ptbin)

    # get the dir where the inv mass histo are
    subdir_name = f'pt_{ptbin[0]}{ptbin[1]}'
    input_subdir = input_file.Get(f'{cent_dir_name}/{subdir_name}')

    # create the subdir in the output file
    output_subdir = cent_dir.mkdir(subdir_name)
    output_subdir.cd()

    for bkgmodel in BKG_MODELS:
        # create dirs for models
        fit_dir = output_subdir.mkdir(bkgmodel)
        fit_dir.cd()

        # loop over all the histo in the dir
        for key in input_subdir.GetListOfKeys():
            keff = key.GetName()[-5:]
            
            hist = TH1D(key.ReadObj())
            hist.SetDirectory(0)

            if key == input_subdir.GetListOfKeys()[0] and bkgmodel=="pol2":
                rawcounts, err_rawcounts, significance, err_significance, mu, mu_err, sigma, sigma_err = au.fit_hist(hist, ptbin, mass, model=bkgmodel, Eint=EINT, peak_mode=PEAK_MODE, gauss=GAUSS, mass_range=MASS_WINDOW)
                mean_fit.append(mu)
                mean_fit_error.append(mu_err)
                sigma_fit.append(sigma)
                sigma_fit_error.append(sigma_err)
                
            else:
                rawcounts, err_rawcounts, significance, err_significance, _, _, _, _ = au.fit_hist(hist, ptbin, mass, model=bkgmodel, Eint=EINT, peak_mode=PEAK_MODE, gauss=GAUSS, mass_range=MASS_WINDOW)

            dict_key = f'{keff}_{bkgmodel}'

            h1_rawcounts_dict[dict_key].SetBinContent(ptbin_index, rawcounts)
            h1_rawcounts_dict[dict_key].SetBinError(ptbin_index, err_rawcounts)

            significance_dict[dict_key].SetBinContent(ptbin_index, significance)
            significance_dict[dict_key].SetBinError(ptbin_index, err_significance)

            if key == input_subdir.GetListOfKeys()[0]:
                h1_BDT_eff.SetBinContent(ptbin_index, float(keff))                           

cent_dir.cd()
h1_eff.Write()
h1_BDT_eff.Write()
for lab in LABELS:
    h1_rawcounts_dict[lab].Write()

hist_mean = h1_eff.Clone("Mean")
hist_mean.SetTitle("; #it{p}_{T} (GeV/#it{c}); #mu (GeV/#it{c}^{2})")
hist_sigma = h1_eff.Clone("Sigma")
hist_sigma.SetTitle( "; #it{p}_{T} (GeV/#it{c}); #sigma (GeV/#it{c}^{2})")

for iBin in range(1, hist_mean.GetNbinsX()+1):
    hist_mean.SetBinContent(iBin, mean_fit[iBin-1])
    hist_mean.SetBinError(iBin, mean_fit_error[iBin-1])
    hist_sigma.SetBinContent(iBin, sigma_fit[iBin-1])
    hist_sigma.SetBinError(iBin, sigma_fit_error[iBin-1])

hist_mean.Write()
hist_sigma.Write()

output_file.Close()
