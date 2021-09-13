#!/usr/bin/env python3

import argparse
import math
import os
import random
import time
from array import array
from typing import MutableMapping

import numpy as np
import yaml
import plot_utils as pu
import analysis_utils as au
from ROOT import (TF1, TH1D, TH2D, TCanvas, TColor, TFile,
                  TPaveText, gPad, gROOT,
                  TDatabasePDG)
import ROOT

kBlueC = TColor.GetColor('#1f78b4')
kBlueCT = TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = TColor.GetColor('#e31a1c')
kRedCT = TColor.GetColorTransparent(kRedC, 0.5)
kPurpleC = TColor.GetColor('#911eb4')
kPurpleCT = TColor.GetColorTransparent(kPurpleC, 0.5)
kOrangeC = TColor.GetColor('#ff7f00')
kOrangeCT = TColor.GetColorTransparent(kOrangeC, 0.5)
kGreenC = TColor.GetColor('#33a02c')
kGreenCT = TColor.GetColorTransparent(kGreenC, 0.5)
kMagentaC = TColor.GetColor('#f032e6')
kMagentaCT = TColor.GetColorTransparent(kMagentaC, 0.5)
kYellowC = TColor.GetColor('#ffe119')
kYellowCT = TColor.GetColorTransparent(kYellowC, 0.5)
kBrownC = TColor.GetColor('#b15928')
kBrownCT = TColor.GetColorTransparent(kBrownC, 0.5)

random.seed(1989)

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
EVENT_PATH = os.path.expandvars(params['EVENT_PATH'][0])

SCALE = args.scale


###############################################################################
start_time = time.time()                          # for performances evaluation

gROOT.SetBatch()

background_file = ROOT.TFile(EVENT_PATH)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
background_file.Close()
rate = 75000 #hz
running_time = 30#days

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
MT_BINS = au.pt_array_to_mt_m0_array(PT_BINS, mass)
binning = array('d',MT_BINS)
pt_binning = array('d',PT_BINS)

resultsSysDir = os.environ['HYPERML_RESULTS']+"/"+FILE_PREFIX

var = '#it{p}_{T}'
unit = 'GeV/#it{c}'

file_name = resultsSysDir + '/' + FILE_PREFIX + '_yield.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + FILE_PREFIX + '_results_fit.root'
results_file = TFile(file_name, 'read')

bkgModels = params['BKG_MODELS']

if SCALE:
    n_run = rate*running_time*24*60*60*5
else:
    n_run = n_ev
    
inDirName = '0-5'

h1BDTEff = results_file.Get(f'{inDirName}/BDTeff')

best_sig = np.round(np.array(h1BDTEff)[1:-1], 3)
#best_sig = [0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991]
#best_sig = [0.8,0.8,0.8,0.8]
#best_sig[0] = 0.15
sig_ranges = []
eff_m = 0.05
eff_p = 0.05
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

results_file.cd(inDirName)
out_dir = distribution.mkdir(inDirName)
cvDir = out_dir.mkdir("canvas")

h1PreselEff = results_file.Get(f'{inDirName}/PreselEff')

for i in range(1, h1PreselEff.GetNbinsX() + 1):
    h1PreselEff.SetBinError(i, 0)

h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
h1PreselEff.UseCurrentStyle()
h1PreselEff.SetMinimum(0)
out_dir.cd()
h1PreselEff.Write("h1PreselEff")

hRawCounts = []
raws = []
errs = []

for model in bkgModels:      
    h1RawCountsPt = TH1D(f"pt_best_{model}",";#it{p}_{T} [GeV/#it{c}];1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",len(PT_BINS)-1,pt_binning)

    for iBin in range(1, h1RawCountsPt.GetNbinsX() + 1):
        h1Counts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.3f}_{model}')

        h1RawCountsPt.SetBinContent(iBin, h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1]/ n_ev)
        h1RawCountsPt.SetBinError(iBin, h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1]/ math.sqrt(n_ev*n_run))
        
        raws.append([])
        errs.append([])

        for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
            if eff >= EFF_MAX:
                continue
            
            h1Counts = results_file.Get(f'{inDirName}/RawCounts{eff:.3f}_{model}')
            raws[iBin-1].append(h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / eff/ n_ev)
            errs[iBin-1].append(h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / eff/ math.sqrt(n_ev*n_run) )


    out_dir.cd()

    ###########################################################################
    #h1RawCounts.UseCurrentStyle()

    h1RawCountsPt.GetXaxis().SetRangeUser(0,2.75)
    h1RawCountsPt.Write()
    hRawCounts.append(h1RawCountsPt)

    cvDir.cd()
    myCv = TCanvas(f"pT_SpectraCv_{model}")
    gPad.SetLeftMargin(0.15); 
                
    min_value = h1RawCountsPt.GetMinimum()*0.8
    max_value = h1RawCountsPt.GetMaximum()*1.2
    
    h1RawCountsPt.GetYaxis().SetRangeUser(min_value, max_value)
    print("**************************************************")
    bratio = 0.489
    mult=0
    err_mult=0
    for i in range(1,h1RawCountsPt.GetNbinsX()+1):
        print("pt int: ",i," multy: ",mult)
        mult += h1RawCountsPt.GetBinContent(i)
        err_mult += h1RawCountsPt.GetBinError(i)**2
    err_mult = ROOT.TMath.Sqrt(err_mult)/bratio
    mult /= bratio
    print("multiplicity: ", mult," +- ",err_mult)
    mult_gen = 39.15
    print("multiplicity gen: ", mult)
    print("**************************************************")
    h1RawCountsPt.SetMarkerStyle(20)
    h1RawCountsPt.SetMarkerColor(ROOT.kBlue)
    h1RawCountsPt.SetLineColor(ROOT.kBlue)
    h1RawCountsPt.SetStats(0)
    h1RawCountsPt.Draw("ex0same")

    myCv.SetLogy()
    myCv.SaveAs(resultsSysDir+"/yield_"+FILE_PREFIX+"_"+model+".png")
    myCv.SaveAs(resultsSysDir+"/yield_"+FILE_PREFIX+"_"+model+".pdf")

    tmpSyst = h1RawCountsPt.Clone("hSyst")
    corSyst = h1RawCountsPt.Clone("hCorr")
    tmpSyst.SetFillStyle(0)
    corSyst.SetFillStyle(3345)
    for iBin in range(1, h1RawCountsPt.GetNbinsX() + 1):
        val = h1RawCountsPt.GetBinContent(iBin)

    out_dir.cd()
    myCv.Write()

    h1RawCountsPt.Draw("ex0same")
    myCv.SetLogy()
    cvDir.cd()
    myCv.Write()

out_dir.cd()

syst = TH1D("syst", ";Yield;Entries", 800, 33, 40)
tmpCt = hRawCounts[0].Clone("tmpCt")

combinations = set()
size = 20000
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
    mult/=bratio
    err_mult = ROOT.TMath.Sqrt(err_mult)/bratio
    #print(mult)
    combinations.add(combo)
    syst.Fill(mult)

syst.SetFillColor(600)
syst.SetFillStyle(3345)
syst.Write()

results_file.Close()

print('')
print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')