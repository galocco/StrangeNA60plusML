#!/usr/bin/env python3
import argparse
# import collections.abc
import os
import warnings
import math
import yaml
from array import array

import analysis_utils as au
import plot_utils as pu

from ROOT import TH1D, TFile, gROOT, TDatabasePDG, TF1
import ROOT
# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()
import numpy as np


def s_over_b_with_uncertainty(B, S):
    """
    Compute the background-to-signal ratio R = B / S
    and its uncertainty δR using error propagation,
    assuming Poisson uncertainties: δS = sqrt(S), δB = sqrt(B).
    
    Parameters:
        S (float): number of signal events
        B (float): number of background events
        
    Returns:
        R (float): B/S ratio
        dR (float): uncertainty on the ratio
    """
    if S <= 0:
        raise ValueError("S must be positive to compute B/S.")

    R = B / S

    dS = np.sqrt(S)
    dB = np.sqrt(B)

    dR = np.sqrt((dB / S)**2 + ((B * dS) / S**2)**2)

    return R, dR

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
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
EINT = pu.get_sNN(params['EINT'])
PT_BINS = params['PT_BINS']
SIG_MODELS = params['SIG_MODELS']
BKG_MODELS = params['BKG_MODELS']

NEVENTS_DATA = params['NEVENTS_DATA']

full_run = 0.15*3*10**10
h1RawCounts = {}
h1RawCountsPt = {}

hist_BS = {}
hist_Sgn_ev = {}
hist_Sgn = {}

for sigmodel in SIG_MODELS:
    hist_BS[sigmodel] = {}
    hist_Sgn_ev[sigmodel] = {}
    hist_Sgn[sigmodel] = {}
    for bkgmodel in BKG_MODELS:
        hist_BS[sigmodel][bkgmodel]  = TH1D(f"hist_BS_{sigmodel}_{bkgmodel}", ";#it{p}_{T} (GeV/#it{c});B/S", len(PT_BINS)-1, array('d', PT_BINS))
        hist_Sgn_ev[sigmodel][bkgmodel]  = TH1D(f"hist_Sgn_ev_{sigmodel}_{bkgmodel}", ";#it{p}_{T} (GeV/#it{c});#frac{Significance}{#sqrt{n_{ev}}}", len(PT_BINS)-1, array('d', PT_BINS))
        hist_Sgn[sigmodel][bkgmodel]  = TH1D(f"hist_Sgn_{sigmodel}_{bkgmodel}", ";#it{p}_{T} (GeV/#it{c});#frac{S}{#sqrt{S+B}}", len(PT_BINS)-1, array('d', PT_BINS))


###############################################################################
# define paths for loading results
results_dir = os.environ['RESULTS']

input_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_fit.root'
input_file = TFile(input_file_name, 'read')

output_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_results_BS.root'
output_file = TFile(output_file_name, 'recreate')

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
nsigma = 3

print(FILE_PREFIX)
for sigmodel in SIG_MODELS:
    for bkgmodel in BKG_MODELS:
        pt_counter = 0
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            pt_counter += 1
            input_subdir = input_file.Get(f'pt_{ptbin[0]}{ptbin[1]}/{sigmodel}/{bkgmodel}')

            # loop over all the histo in the dir
            for key in input_subdir.GetListOfKeys():
                hist = key.ReadObj()
                retrieved_fit = hist.GetFunction("fitTpl")
                nbkg_par = 2
                if bkgmodel == "pol2":
                    bkg = 3

                
                    
                if sigmodel == "d-gauss":
                    signal = (retrieved_fit.GetParameter(nbkg_par)+retrieved_fit.GetParameter(3+nbkg_par))/ hist.GetBinWidth(1)
                    
                    sigma = min([retrieved_fit.GetParameter(nbkg_par+2)+retrieved_fit.GetParameter(nbkg_par+5)])
                    if ptbin[1] == 3 and "K0S" in FILE_PREFIX:
                        signal = retrieved_fit.GetParameter(nbkg_par)/ hist.GetBinWidth(1) if sigma == retrieved_fit.GetParameter(nbkg_par+2) else retrieved_fit.GetParameter(nbkg_par+3)/ hist.GetBinWidth(1)

                    
                elif sigmodel == "gauss":
                    signal = retrieved_fit.GetParameter(nbkg_par)/ hist.GetBinWidth(1)
                    sigma = retrieved_fit.GetParameter(nbkg_par+2)
                
                #bkg = bkg_fit.Integral(mass-nsigma*sigma, mass+nsigma*sigma)

                # Get the bin numbers corresponding to x_min and x_max
                bin_min = hist.FindBin(mass-nsigma*sigma)
                bin_max = hist.FindBin(mass+nsigma*sigma)

                # Compute the integral
                bkg = hist.Integral(bin_min, bin_max)-signal

                print(f"ptbin: {ptbin}, signal: {signal}, bkg: {bkg}")
                sign, sign_err = au.significance_with_uncertainty(signal, bkg)
                ratio, ratio_err = s_over_b_with_uncertainty(signal, bkg)

                hist_BS[sigmodel][bkgmodel].SetBinContent(pt_counter, ratio)
                hist_BS[sigmodel][bkgmodel].SetBinError(pt_counter, ratio_err)

                hist_Sgn_ev[sigmodel][bkgmodel].SetBinContent(pt_counter, sign/math.sqrt(NEVENTS_DATA))
                hist_Sgn_ev[sigmodel][bkgmodel].SetBinError(pt_counter, sign_err/math.sqrt(NEVENTS_DATA))


                hist_Sgn[sigmodel][bkgmodel].SetBinContent(pt_counter, sign*math.sqrt(full_run/NEVENTS_DATA))
                hist_Sgn[sigmodel][bkgmodel].SetBinError(pt_counter, sign_err*math.sqrt(full_run/NEVENTS_DATA))

                break

output_file.cd()

for sigmodel in SIG_MODELS:
    for bkgmodel in BKG_MODELS:
        hist_BS[sigmodel][bkgmodel].Write()
        hist_Sgn[sigmodel][bkgmodel].Write()
        hist_Sgn_ev[sigmodel][bkgmodel].Write()
output_file.Close()
