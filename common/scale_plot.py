#!/usr/bin/env python3

import argparse
import math
import os
import random
from array import array
from multiprocessing import Pool
import plot_utils as pu
import numpy as np
import yaml
import ROOT
from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame,
                  TIter, TKey, TPaveText, gDirectory, gPad, gROOT, gStyle,
                  kBlue, kRed)
from scipy import stats

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--custom', help='Run with customized bdt efficiencies', action='store_true')
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
FILE_PREFIX = params['FILE_PREFIX']
EINT = pu.get_sNN(params['EINT'])

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']

BKG_MODELS = params['BKG_MODELS']

SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

CUSTOM = args.custom
###############################################################################
# define paths for loading data
event_path = os.path.expandvars(params['EVENT_PATH'])

###############################################################################
file_name = f"../Results/2Body/{FILE_PREFIX}_results_fit.root"
inDirName = f'{CENT_CLASSES[0][0]}-{CENT_CLASSES[0][1]}'
input_file = TFile(file_name)
h2BDTEff = input_file.Get(f'{inDirName}/BDTeff')
h1BDTEff = h2BDTEff.ProjectionX("bdteff", 1, h2BDTEff.GetNbinsY()+1)
if CUSTOM:
    print("insert custom bdt efficiencies:")
    best_sig_eff = []
    for index in range(0,len(PT_BINS)):
        best_sig_eff.append(input())
else:
    best_sig_eff = np.round(np.array(h1BDTEff)[1:-1], 2)
#best_sig_eff[2] = 0.28
background_file = TFile(event_path)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
background_file.Close()
full_run = 10**10
sig_index = 0

for split in SPLIT_LIST:
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        results_file = TFile(file_name)

        cv = TCanvas("cv","cv")
        histo = results_file.Get(f'0-5/pt_{ptbin[0]}{ptbin[1]}/pol2/ct0100_pT{ptbin[0]}{ptbin[1]}_cen05_eff{best_sig_eff[sig_index]:.2f}{split}')
        lineshape = histo.GetFunction("fitTpl")
        parameter = []
        for par_index in range(0,6):
            parameter.append(lineshape.GetParameter(par_index))
        histo.GetListOfFunctions().Remove(lineshape)
        for index in range(1,histo.GetNbinsX()+1):
            mass = histo.GetBinCenter(index)

            count = int(lineshape.Eval(mass)*full_run/n_ev)
            if count<0:
                count = 0
            histo.SetBinContent(index,count)
            histo.SetBinError(index,math.sqrt(count))

        true_mass = histo.GetBinCenter(20)
        histo.GetYaxis().SetRangeUser(0,1.5*int(lineshape.Eval(true_mass)*full_run/n_ev))
        histo.Draw()

        for par_index in range(0,4):
            lineshape.SetParameter(par_index,lineshape.GetParameter(par_index)*full_run/n_ev)

        lineshape.Draw("same")

        bkg_tpl = TF1('fitTpl', 'pol2(0)', 0, 5)

        bkg_tpl.SetParameter(0,lineshape.GetParameter(0))
        bkg_tpl.SetParameter(1,lineshape.GetParameter(1))
        bkg_tpl.SetParameter(2,lineshape.GetParameter(2))

        # get the fit parameters
        n_bkgpars = 3
        nsigma = 3
        mu = lineshape.GetParameter(n_bkgpars+1)
        muErr = lineshape.GetParError(n_bkgpars+1)
        sigma = lineshape.GetParameter(n_bkgpars+2)
        sigmaErr = lineshape.GetParError(n_bkgpars+2)
        signal = lineshape.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
        errsignal = lineshape.GetParError(n_bkgpars) / histo.GetBinWidth(1)
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

        # print fit info on the canvas
        pinfo2 = TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
        pinfo2.SetBorderSize(0)
        pinfo2.SetFillStyle(0)
        pinfo2.SetTextAlign(30+3)
        pinfo2.SetTextFont(42)

        string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{EINT} GeV, centrality {0}-{5}%'
        pinfo2.AddText(string)

        string = '0 #leq #it{ct} < 100 cm, '+f'{ptbin[0]:.1f}'+' #leq #it{p}_{T} < '+f'{ptbin[1]:.1f}'+' GeV/#it{c} '
        pinfo2.AddText(string)

        string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f}'
        pinfo2.AddText(string)

        string = f'S ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
        pinfo2.AddText(string)
            
        string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
        pinfo2.AddText(string)

        if bkg > 0:
            ratio = signal/bkg
            string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'

        pinfo2.AddText(string)

        string = '#sigma__{fit}'+f' {1000*sigma:.0f} #pm {1000*sigmaErr:.0f} MeV/'+'#it{c}^{2}'
        pinfo2.AddText(string)
        print("sigma: ",sigma*1000," +- ",1000*sigmaErr)
        
        pinfo2.Draw()
        gStyle.SetOptStat(0)
        
        
        
        bkg_tpl.SetNpx(300)
        bkg_tpl.SetLineWidth(2)
        bkg_tpl.SetLineStyle(2)
        bkg_tpl.SetLineColor(ROOT.kBlue)
        bkg_tpl.Draw("same")

        sig_index += 1
        cv.SaveAs('../Results/2Body/'+FILE_PREFIX+f'_pol2_pt_{ptbin[0]}{ptbin[1]}{split}.png')
        cv.SaveAs('../Results/2Body/'+FILE_PREFIX+f'_pol2_pt_{ptbin[0]}{ptbin[1]}{split}.pdf')