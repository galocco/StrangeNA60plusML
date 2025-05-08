#!/usr/bin/env python3

import argparse
import math
import os
import random
import time
from array import array

import numpy as np
import yaml
import plot_utils as pu
import analysis_utils as au
from ROOT import (TF1, TH1D, TCanvas, TFile,
                  TLine, gPad, gROOT, TDatabasePDG)
import ROOT

random.seed(1996)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
parser.add_argument('-s', '--scale', help='Scale the results to a complete run', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

FILE_PREFIX = params['FILE_PREFIX']
EINT = pu.get_sNN(params['EINT'])
T = params['T']
MULTIPLICITY = params['MULTIPLICITY']
PDG_CODE = params['PDG']
PT_BINS = params['PT_BINS']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
NEVENTS_DATA = params['NEVENTS_DATA']
BRATIO = params["BRATIO"]

SIG_MODELS = params['SIG_MODELS']
BKG_MODELS = params['BKG_MODELS']

SCALE = args.scale

###############################################################################
start_time = time.time()                          # for performances evaluation

gROOT.SetBatch()

rate = 75000 #hz
running_time = 30#days

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
pt_binning = array('d',PT_BINS)

resultsSysDir = os.environ['RESULTS']+"/"+FILE_PREFIX

var = '#it{p}_{T}'
unit = 'GeV/#it{c}'

file_name = resultsSysDir + '/' + FILE_PREFIX + '_yield.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + FILE_PREFIX + '_results_fit.root'
results_file = TFile(file_name, 'read')

if SCALE:
    n_run = rate*running_time*24*60*60*5
else:
    n_run = NEVENTS_DATA

h1BDTEff = results_file.Get('BDTeff')

best_sig = np.round(np.array(h1BDTEff)[1:-1], 3)
sig_ranges = []
eff_m = params["EFF_RANGE_SYST"][0]
eff_p = params["EFF_RANGE_SYST"][1]
for i in best_sig:
    if EFF_MAX < i+eff_p:
        eff_p = EFF_MAX-i
    if EFF_MIN > i-eff_m:
        eff_m = i-EFF_MIN
    sig_ranges.append([i-eff_m, i+eff_p, 0.01])
ranges = {
        'BEST': best_sig,
        'SCAN': sig_ranges
}

results_file.cd()

h1PreselEff = results_file.Get('PreselEff')

for i in range(1, h1PreselEff.GetNbinsX() + 1):
    h1PreselEff.SetBinError(i, 0)

h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
h1PreselEff.UseCurrentStyle()
h1PreselEff.SetMinimum(0)
distribution.cd()
h1PreselEff.Write("h1PreselEff")

hRawCounts = []
raws = []
errs = []


h1RawCounts = {}
h1RawCountsPt = {}

h1RawCountsPt = {}
for sigmodel in SIG_MODELS:
    h1RawCountsPt[sigmodel] = {}
    for bkgmodel in BKG_MODELS:
        h1RawCountsPt[sigmodel][bkgmodel] = ROOT.TH1D(f"pt_best_{sigmodel}_{bkgmodel}",";#it{p}_{T} [GeV/#it{c}];Counts/N_{ev} ",len(PT_BINS)-1,pt_binning)


pt_distr = TF1("pt_distr", "x*exp(-TMath::Sqrt(x**2+[1]**2)/[0])", PT_BINS[0], PT_BINS[-1])
pt_distr.FixParameter(0, T)
pt_distr.FixParameter(1, mass)


pt_range_factor = au.get_pt_integral(pt_distr, PT_BINS[0],PT_BINS[-1])/au.get_pt_integral(pt_distr)
print("pt_range_factor: ",pt_range_factor)
for sigmodel in SIG_MODELS:
    for bkgmodel in BKG_MODELS:

        for iBin in range(1, h1RawCountsPt[sigmodel][bkgmodel].GetNbinsX() + 1):
            h1Counts = results_file.Get(f'RawCounts{ranges["BEST"][iBin-1]:.3f}_{sigmodel}_{bkgmodel}')
            #print('RawCounts{ranges["BEST"][iBin-1]:.3f}_{sigmodel}_{bkgmodel}')
            h1RawCountsPt[sigmodel][bkgmodel].SetBinContent(iBin, h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1]/ NEVENTS_DATA)
            h1RawCountsPt[sigmodel][bkgmodel].SetBinError(iBin, h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1]/ math.sqrt(NEVENTS_DATA*n_run))
            
            raws.append([])
            errs.append([])

            for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                h1Counts = results_file.Get(f'RawCounts{eff:.3f}_{sigmodel}_{bkgmodel}')
                raws[iBin-1].append(h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / eff/ NEVENTS_DATA)
                errs[iBin-1].append(h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / eff/ math.sqrt(NEVENTS_DATA*n_run) )


        distribution.cd()

        ###########################################################################
        #h1RawCounts.UseCurrentStyle()

        h1RawCountsPt[sigmodel][bkgmodel].GetXaxis().SetRangeUser(PT_BINS[0], PT_BINS[-1])
        h1RawCountsPt[sigmodel][bkgmodel].Write()
        hRawCounts.append(h1RawCountsPt[sigmodel][bkgmodel])

        myCv = TCanvas(f"pT_SpectraCv_{bkgmodel}")
        gPad.SetLeftMargin(0.15); 
                    
        min_value = h1RawCountsPt[sigmodel][bkgmodel].GetMinimum()*0.8
        max_value = h1RawCountsPt[sigmodel][bkgmodel].GetMaximum()*1.2
        
        h1RawCountsPt[sigmodel][bkgmodel].GetYaxis().SetRangeUser(min_value, max_value)
        mult=0
        err_mult=0
        for i in range(1,h1RawCountsPt[sigmodel][bkgmodel].GetNbinsX()+1):
            mult += h1RawCountsPt[sigmodel][bkgmodel].GetBinContent(i)
            err_mult += h1RawCountsPt[sigmodel][bkgmodel].GetBinError(i)**2
        err_mult = ROOT.TMath.Sqrt(err_mult)/BRATIO/pt_range_factor
        mult /= BRATIO
        mult /= pt_range_factor
        print("sig model: ",sigmodel,"bkg model: ",bkgmodel)
        print("multiplicity: ", mult," +- ",err_mult)
        print("multiplicity gen: ", MULTIPLICITY)
        print("z_gauss: ", (MULTIPLICITY-mult)/err_mult)
        print("**************************************************")
        h1RawCountsPt[sigmodel][bkgmodel].SetMarkerStyle(20)
        h1RawCountsPt[sigmodel][bkgmodel].SetMarkerColor(ROOT.kBlue)
        h1RawCountsPt[sigmodel][bkgmodel].SetLineColor(ROOT.kBlue)
        h1RawCountsPt[sigmodel][bkgmodel].SetStats(0)
        h1RawCountsPt[sigmodel][bkgmodel].Draw("ex0same")


        tmpSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hSyst")
        corSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCountsPt[sigmodel][bkgmodel].GetNbinsX() + 1):
            val = h1RawCountsPt[sigmodel][bkgmodel].GetBinContent(iBin)

distribution.cd()
if PDG_CODE == 3334:
    syst = TH1D("syst", ";Yield;Entries", 1000, MULTIPLICITY*0.5, MULTIPLICITY*4)
else:
    syst = TH1D("syst", ";Yield;Entries", 1000, MULTIPLICITY*0.9, MULTIPLICITY*1.1)
tmpCt = hRawCounts[0].Clone("tmpCt")

combinations = set()
size = 10000
count=0

for _ in range(size):
    tmpCt.Reset()
    comboList = []
    mult = 0
    err_mult = 0
    for iBin in range(1, tmpCt.GetNbinsX() + 1):
        index = random.randint(0, len(raws[iBin-1])-1)
        comboList.append(index)
        mult += raws[iBin-1][index]
        err_mult += errs[iBin-1][index]**2
    combo = (x for x in comboList)
    if combo in combinations:
        continue
    mult/=BRATIO*pt_range_factor
    err_mult = ROOT.TMath.Sqrt(err_mult)/BRATIO/pt_range_factor
    combinations.add(combo)
    syst.Fill(mult)

syst.SetFillColor(600)
syst.SetFillStyle(3345)
print(syst.GetMaximum()*1.2)
max_counts = syst.GetMaximum()*1.2
syst.GetYaxis().SetRangeUser(0, max_counts)
syst.Write()
myCv = TCanvas(f"cv_syst")
gPad.SetLeftMargin(0.15)
syst.Draw()        
true_value_line = TLine(MULTIPLICITY, 0, MULTIPLICITY, max_counts)
true_value_line.SetLineColor(ROOT.kRed)
true_value_line.SetLineStyle(7)
true_value_line.Draw("same")
myCv.SaveAs(resultsSysDir+"/yield_syst_"+FILE_PREFIX+".pdf")
myCv.SaveAs(resultsSysDir+"/yield_syst_"+FILE_PREFIX+".png")
results_file.Close()

print('')
print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')