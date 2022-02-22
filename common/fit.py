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

from ROOT import TH1D, TFile, gROOT, TDatabasePDG, gSystem

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-f', '--fix', help='Fix the parameters taken from the signal-only fit', action='store_true')
parser.add_argument('-p', '--print', help='Produce png and pdf of the invariant mass plots', action='store_true')
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
PT_BINS = params['PT_BINS']
MASS_WINDOW = params['MASS_WINDOW']
SIGMA = params['SIGMA']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

SIG_MODELS = params['SIG_MODELS']
BKG_MODELS = params['BKG_MODELS']

results_dir = os.environ['RESULTS']
PLOT_DIR = results_dir+f'/{FILE_PREFIX}/{FILE_PREFIX}_mass_plots'

if gSystem.AccessPathName(PLOT_DIR):
    gSystem.Exec('mkdir '+PLOT_DIR)

FIX = args.fix
PRINT = ""
LABELS = [f'{x:.3f}_{y}_{z}' for x in FIX_EFF_ARRAY for y in SIG_MODELS for z in BKG_MODELS]

###############################################################################
# define paths for loading results


input_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results.root'
input_file = TFile(input_file_name, 'read')

output_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_fit.root'
output_file = TFile(output_file_name, 'recreate')

mc_fit_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_mc_fit.root'

mc_fit_file = TFile(mc_fit_file_name,"read")
###############################################################################
# define dictionaries for storing raw counts and significance
h1_rawcounts_dict = {}
significance_dict = {}

count=0
###############################################################################
start_time = time.time()                          # for performances evaluation
# start the actual signal extraction
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
output_file.cd()

h1_eff = input_file.Get('PreselEff')
h1_BDT_eff = au.h1_rawcounts(PT_BINS, name = "BDTeff")

for lab in LABELS:
    h1_rawcounts_dict[lab] = au.h1_rawcounts(PT_BINS, suffix=lab)
    significance_dict[lab] = au.h1_significance(PT_BINS, suffix=lab)

for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):

    print('\n==================================================')
    print('pT:', ptbin)
    print('Fitting ...', end='\r')
    ptbin_index = au.get_ptbin_index(h1_eff, ptbin)

    # get the dir where the inv mass histo are
    subdir_name = f'pt_{ptbin[0]}{ptbin[1]}'
    input_subdir = input_file.Get(f'{subdir_name}')

    # create the subdir in the output file
    output_subdir = output_file.mkdir(subdir_name)
    output_subdir.cd()
    for sigmodel in SIG_MODELS:
        # create dirs for models
        fit_sig_dir = output_subdir.mkdir(sigmodel)
        for bkgmodel in BKG_MODELS:
            fit_bkg_dir = fit_sig_dir.mkdir(bkgmodel)
            fit_bkg_dir.cd()

            # loop over all the histo in the dir
            for key in input_subdir.GetListOfKeys():
                keff = key.GetName()[-5:]
                
                hist = TH1D(key.ReadObj())
                hist.SetDirectory(0)

                if args.print:
                    PRINT = f'{PLOT_DIR}/{FILE_PREFIX}_pt_{ptbin[0]}_{ptbin[1]}_sig_{bkgmodel}_bkg_{bkgmodel}_bdt_eff_{keff}'

                print("fit model: ",bkgmodel,"+",sigmodel," BDT efficiency: ",keff)
                rawcounts, err_rawcounts = au.fit_hist(hist, ptbin, mass, sig_model=sigmodel, bkg_model=bkgmodel, Eint=EINT, mass_range=MASS_WINDOW, mc_fit_file = mc_fit_file, directory = fit_bkg_dir, fix_params = FIX, peak_width=SIGMA*4, print=PRINT)

                dict_key = f'{keff}_{sigmodel}_{bkgmodel}'

                h1_rawcounts_dict[dict_key].SetBinContent(ptbin_index, rawcounts)
                h1_rawcounts_dict[dict_key].SetBinError(ptbin_index, err_rawcounts)

                if key == input_subdir.GetListOfKeys()[0]:
                    h1_BDT_eff.SetBinContent(ptbin_index, float(keff))                           

output_file.cd()
h1_eff.Write()
h1_BDT_eff.Write()
for lab in LABELS:
    h1_rawcounts_dict[lab].Write()

output_file.Close()

print('')
print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')