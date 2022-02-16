#!/usr/bin/env python3
import argparse
import math
import os
import time
import warnings
from array import array
from StrangeNA60plusML.common.run_analysis import NEVENTS
import numpy as np
import yaml
from ROOT import TFile, gROOT, TF1, TH1D

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--custom', help='Run with customized bdt efficiencies', action='store_true')
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

PT_BINS = params['PT_BINS']

BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

CUSTOM = args.custom

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
NEVENTS = params['NEVENTS']
results_dir = os.environ['RESULTS']
###############################################################################
start_time = time.time()                          # for performances evaluation
file_name = results_dir+f"/{FILE_PREFIX}/{FILE_PREFIX}_results_fit.root"
input_file = TFile(file_name)
h1BDTEff = input_file.Get(f'BDTeff')

if CUSTOM:
    print("insert custom bdt efficiencies:")
    best_sig_eff = []
    for index in range(0,len(PT_BINS)):
        best_sig_eff.append(input())
else:
    best_sig_eff = np.round(np.array(h1BDTEff)[1:-1], 3)
n_bkgpars = 3
nsigma = 3

binning = array('d',PT_BINS)
output_file = TFile(results_dir + '/' + FILE_PREFIX + '/' + FILE_PREFIX+'_signal_results.root',"RECREATE")
for model in BKG_MODELS:
    if 'pol' in str(model):
        n_bkgpars = int(model[3]) + 1
    elif 'expo' in str(model):
        n_bkgpars = 2
    else:
        print(f'Unsupported model {model}')

    hist_sig = TH1D(f'hist_sig_{model}', ';#it{p}_{T} (GeV/#it{c}); #frac{1}{#it{p}_{T}}#frac{Significance}{#sqrt{N_{ev}}}',len(PT_BINS)-1,binning)
    hist_BS = TH1D(f'hist_BS_{model}', ';#it{p}_{T} (GeV/#it{c}); B/S',len(PT_BINS)-1,binning)
    hist_sigma = TH1D(f'hist_sigma_{model}', ';#it{p}_{T} (GeV/#it{c}); #sigma (GeV/#it{c}^{2})',len(PT_BINS)-1,binning)

    for index in range(0,len(best_sig_eff)):
        dir_name = f'pt_{PT_BINS[index]}{PT_BINS[index+1]}/'
        obj_name = f'eff{best_sig_eff[index]:.3f}'

        histo = input_file.Get(dir_name+f'{model}/'+obj_name)
        lineshape = histo.GetFunction("fitTpl")
        bkg_tpl = TF1('bkgTpl', f'{model}(0)', 0, 5)
        for parameter in range(0,n_bkgpars):
            bkg_tpl.SetParameter(parameter,lineshape.GetParameter(parameter))
        # get the fit parameters
        mu1 = lineshape.GetParameter(n_bkgpars+1)
        muErr1 = lineshape.GetParError(n_bkgpars+1)
        sigma1 = lineshape.GetParameter(n_bkgpars+2)
        sigmaErr1 = lineshape.GetParError(n_bkgpars+2)
        mu2 = lineshape.GetParameter(n_bkgpars+4)
        muErr2 = lineshape.GetParError(n_bkgpars+4)
        sigma2 = lineshape.GetParameter(n_bkgpars+5)
        sigmaErr2 = lineshape.GetParError(n_bkgpars+5)
        signal = (lineshape.GetParameter(n_bkgpars)+lineshape.GetParameter(n_bkgpars+3)) / histo.GetBinWidth(1)
        errsignal = (lineshape.GetParError(n_bkgpars)+lineshape.GetParError(n_bkgpars+3)) / histo.GetBinWidth(1)
        mu = (mu1+mu2)/2.
        sigma = (sigma1+sigma2)/2.
        bkg = bkg_tpl.Integral(mu - nsigma * sigma, mu +
                                nsigma * sigma) / histo.GetBinWidth(1)

        if bkg > 0:
            errbkg = math.sqrt(bkg)
        else:
            errbkg = 0
        if signal > 0:
            errsignal = math.sqrt(signal)
        else:
            errsignal = 0
        # compute the significance
        if signal+bkg > 0:
            signif = signal/math.sqrt(signal+bkg)
            deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
            deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
            errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
        else:
            signif = 0
            errsignif = 0

        if signal > 0:
            ratio = bkg/signal
            ratioErr = math.sqrt((errbkg/bkg)**2+(errsignal/signal)**2)*ratio
        else:
            ratio = 0
            ratioErr = 0
        sigmaErr = math.sqrt(sigmaErr1**2+sigmaErr2**2)
        if math.isnan(sigmaErr):
            sigmaErr = 0
        hist_sig.SetBinContent(index+1, signif/(PT_BINS[index+1]-PT_BINS[index])/math.sqrt(NEVENTS)
        hist_sig.SetBinError(index+1, errsignif/(PT_BINS[index+1]-PT_BINS[index])/math.sqrt(NEVENTS)
        hist_sigma.SetBinContent(index+1, sigma)
        hist_sigma.SetBinError(index+1, sigmaErr)
        
        hist_BS.SetBinContent(index+1, ratio)
        hist_BS.SetBinError(index+1, ratioErr)
        
    output_file.cd()
    hist_sig.Write()
    hist_BS.Write()
    hist_sigma.Write()
output_file.Close()

print('')
print(f'--- execution in {((time.time() - start_time) / 60):.2f} minutes ---')