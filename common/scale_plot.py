#!/usr/bin/env python3

import argparse
from inspect import getattr_static
import math
import os
import plot_utils as pu
import numpy as np
import yaml
import ROOT
from ROOT import TF1, TCanvas, TFile, TPaveText, gStyle, gPad
from scipy import stats

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
FILE_PREFIX = params['FILE_PREFIX']
EINT = pu.get_sNN(params['EINT'])

PT_BINS = params['PT_BINS']
BKG_MODELS = params['BKG_MODELS']
SIG_MODELS = params['SIG_MODELS']

CUSTOM = args.custom
NEVENTS = params['NEVENTS']
###############################################################################
# define paths for loading data

###############################################################################
file_name = f"../Results/{FILE_PREFIX}/{FILE_PREFIX}_scaled.root"
results_file = TFile(file_name,"recreate")
file_name = f"../Results/{FILE_PREFIX}/{FILE_PREFIX}_results_fit.root"
inDirName = '0-5'
input_file = TFile(file_name,"read")
h1BDTEff = input_file.Get(f'{inDirName}/BDTeff')
if CUSTOM:
    print("insert custom bdt efficiencies:")
    best_sig_eff = []
    for index in range(0,len(PT_BINS)):
        best_sig_eff.append(input())
else:
    best_sig_eff = np.round(np.array(h1BDTEff)[1:-1], 3)
    print(best_sig_eff)
full_run = 10**10
for sigmodel in SIG_MODELS:
    for bkgmodel in BKG_MODELS:
        if 'pol' in str(bkgmodel):
            n_bkgpars = int(bkgmodel[3]) + 1
        elif 'expo' in str(bkgmodel):
            n_bkgpars = 2
        else:
            print(f'Unsupported model {bkgmodel}')
        cv = TCanvas("cv","cv")
        sig_index = 0
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):

            histo = input_file.Get(f'pt_{ptbin[0]}{ptbin[1]}/{sigmodel}/{bkgmodel}/eff{best_sig_eff[sig_index]:.3f}')
            fitTpl = histo.GetFunction("fitTpl")
            if sigmodel == "gauss":
                n_sigpars = 3
                lineshape = TF1('lineshape', f'{bkgmodel}(0)+gausn({n_bkgpars})', 0, 5)
            elif sigmodel == "kde":
                continue
            else:
                n_sigpars = 7
                lineshape = TF1('lineshape', f'{bkgmodel}(0) + [{n_bkgpars+6}]*(gausn({n_bkgpars}) + gausn({n_bkgpars+3}))', 0, 5)
            
            for par_index in range(0,n_bkgpars+n_sigpars):
                print("par(",par_index,") = ",fitTpl.GetParameter(par_index))
                lineshape.SetParameter(par_index, fitTpl.GetParameter(par_index))
            
            histo.GetListOfFunctions().Remove(fitTpl)
            cv.SetLeftMargin(0.15)
            
            for index in range(1,histo.GetNbinsX()+1):
                mass = histo.GetBinCenter(index)

                count = int(lineshape.Eval(mass)*full_run/NEVENTS)
                if count<0:
                    count = 0
                histo.SetBinContent(index,count)
                histo.SetBinError(index,math.sqrt(count))

            true_mass = histo.GetBinCenter(int(histo.GetNbinsX()/2))
            histo.GetYaxis().SetRangeUser(0,1.5*int(lineshape.Eval(true_mass)*full_run/NEVENTS))
            histo.Draw()

            for par_index in range(0, n_bkgpars):
                lineshape.SetParameter(par_index, lineshape.GetParameter(par_index)*full_run/NEVENTS)
            if sigmodel == "d-gauss":
                lineshape.SetParameter(n_bkgpars + 6, lineshape.GetParameter(n_bkgpars + 6)*full_run/NEVENTS)
            
            lineshape.Draw("same")

            bkg_tpl = TF1('bkg_tpl', f'{bkgmodel}(0)', 0, 5)
            for par_index in range(0, n_bkgpars):
                bkg_tpl.SetParameter(par_index, lineshape.GetParameter(par_index))

            # get the fit parameters
            nsigma = 3
            mu = lineshape.GetParameter(n_bkgpars+1)
            muErr = lineshape.GetParError(n_bkgpars+1)
            if sigmodel == "gauss":
                signal = lineshape.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
                errsignal = lineshape.GetParError(n_bkgpars) / histo.GetBinWidth(1)
                sigma = lineshape.GetParameter(n_bkgpars+2)
                sigmaErr = lineshape.GetParError(n_bkgpars+2)
            else:
                signal = (lineshape.GetParameter(n_bkgpars)+lineshape.GetParameter(n_bkgpars+3))*lineshape.GetParameter(n_bkgpars+6) / histo.GetBinWidth(1)
                errsignal = ROOT.TMath.Sqrt(lineshape.GetParError(n_bkgpars+6)) / histo.GetBinWidth(1)
                sigma = lineshape.GetParameter(n_bkgpars+2)
                sigmaErr = lineshape.GetParError(n_bkgpars+2)
            bkg = 10#bkg_tpl.Integral(mu - nsigma * sigma, mu +
                    #                nsigma * sigma) / histo.GetBinWidth(1)

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

            string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{EINT} GeV, centrality 0-5%'
            pinfo2.AddText(string)

            string = f'{ptbin[0]:.1f}'+' #leq #it{p}_{T} < '+f'{ptbin[1]:.1f}'+' GeV/#it{c} '
            pinfo2.AddText(string)

            #string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f}'
            #pinfo2.AddText(string)

            string = f'S ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
            pinfo2.AddText(string)
                
            #string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
            #pinfo2.AddText(string)

            #if bkg > 0:
            #    ratio = signal/bkg
            #    string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'

            pinfo2.AddText(string)
            
            pinfo2.Draw()
            gStyle.SetOptStat(0)
            
            
            
            bkg_tpl.SetNpx(300)
            bkg_tpl.SetLineWidth(2)
            bkg_tpl.SetLineStyle(2)
            bkg_tpl.SetLineColor(ROOT.kBlue)
            bkg_tpl.Draw("same")

            sig_index += 1
            results_file.cd()
            cv.Write()
            histo.Write()
            cv.SaveAs(FILE_PREFIX+f'_{sigmodel}_{bkgmodel}_pt_{ptbin[0]}{ptbin[1]}.pdf')
results_file.Close()
input_file.Close()