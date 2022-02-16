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
from ROOT import (TF1, TH1D, TH2D, TCanvas, TFile,
                  TPaveText, gPad, gROOT, TLine, gStyle,
                  TDatabasePDG)
import ROOT

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

SIG_MODELS = params['SIG_MODELS']
BKG_MODELS = params['BKG_MODELS']

SCALE = args.scale
NEVENTS = params['NEVENTS']

###############################################################################
start_time = time.time()                          # for performances evaluation

gROOT.SetBatch()

rate = 75000 #hz
running_time = 30#days

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
MT_BINS = au.pt_array_to_mt_m0_array(PT_BINS, mass)
MT_BINS_M = au.pt_array_to_mt_array(PT_BINS, mass)
binning = array('d',MT_BINS)
pt_binning = array('d',PT_BINS)

mt_distr = TF1("mt_distr", "[0]*exp(-(x-[2])/[1])",MT_BINS[0],MT_BINS[-1])
mt_distr.SetParameter(1, T)
mt_distr.SetParLimits(1, T*0.6, T*1.2)
mt_distr.FixParameter(2, mass)

pt_distr = TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",PT_BINS[0],PT_BINS[-1])
pt_distr.SetParameter(1, T)
pt_distr.SetParLimits(1, T*0.6, T*1.2)
pt_distr.FixParameter(2, mass)

resultsSysDir = os.environ['RESULTS']+"/"+FILE_PREFIX

var = '#it{p}_{T}'
unit = 'GeV/#it{c}'

file_name = resultsSysDir + '/' + FILE_PREFIX + '_dist.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + FILE_PREFIX + '_results_fit.root'
results_file = TFile(file_name, 'read')


if SCALE:
    n_run = rate*running_time*24*60*60*5
else:
    n_run = NEVENTS
    

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
cvDir = distribution.mkdir("canvas")

h1PreselEff = results_file.Get('PreselEff')

for i in range(1, h1PreselEff.GetNbinsX() + 1):
    h1PreselEff.SetBinError(i, 0)

h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
h1PreselEff.UseCurrentStyle()
h1PreselEff.SetMinimum(0)
distribution.cd()
h1PreselEff.Write("h1PreselEff")

h1TotalEff = h1PreselEff.Clone("TotalEff")
for i in range(1, h1TotalEff.GetNbinsX() + 1):
    presel_eff = h1TotalEff.GetBinContent(i)
    bdt_eff = h1BDTEff.GetBinContent(i)
    h1TotalEff.SetBinContent(i, presel_eff*bdt_eff)
    h1TotalEff.SetBinError(i, 0)


h1TotalEff.SetTitle(f';{var} ({unit}); Preselection efficiency x BDT efficiency')
h1TotalEff.UseCurrentStyle()
h1TotalEff.SetMinimum(0)

myCv = TCanvas(f"cv_eff")
gStyle.SetOptStat(0)
gPad.SetLeftMargin(0.15)
h1TotalEff.Draw()
myCv.SaveAs(resultsSysDir+"/efficiency_"+FILE_PREFIX+".pdf")
myCv.SaveAs(resultsSysDir+"/efficiency_"+FILE_PREFIX+".png")
gStyle.SetOptStat(1)
hRawCounts = []
raws = []
errs = []

h1RawCounts = {}
h1RawCountsPt = {}
for sigmodel in SIG_MODELS:
    h1RawCounts[sigmodel] = {}
    h1RawCountsPt[sigmodel] = {}
    for bkgmodel in BKG_MODELS:
        h1RawCounts[sigmodel][bkgmodel] = ROOT.TH1D(f"mt_best_{sigmodel}_{bkgmodel}",";m_{T}-m_{0} [GeV];1/N_{ev}1/m_{T}dN/dm_{T} [GeV^{-2}]",len(PT_BINS)-1,binning)
        h1RawCountsPt[sigmodel][bkgmodel] = ROOT.TH1D(f"pt_best_{sigmodel}_{bkgmodel}",";#it{p}_{T} [GeV/#it{c}];1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",len(PT_BINS)-1,pt_binning)

for sigmodel in SIG_MODELS:
    for bkgmodel in BKG_MODELS:

        for iBin in range(1, h1RawCounts[sigmodel][bkgmodel].GetNbinsX() + 1):
            h1Counts = results_file.Get(f'RawCounts{ranges["BEST"][iBin-1]:.3f}_{sigmodel}_{bkgmodel}')
            h1RawCounts[sigmodel][bkgmodel].SetBinContent(iBin, h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / (MT_BINS_M[iBin]-MT_BINS_M[iBin-1])/ (MT_BINS_M[iBin-1]+MT_BINS_M[iBin])/2 / NEVENTS)
            h1RawCounts[sigmodel][bkgmodel].SetBinError(iBin, h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / (MT_BINS_M[iBin]-MT_BINS_M[iBin-1])/ (MT_BINS_M[iBin-1]+MT_BINS_M[iBin])/2 / math.sqrt(NEVENTS*n_run))
            
            h1RawCountsPt[sigmodel][bkgmodel].SetBinContent(iBin, h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin) / NEVENTS)
            h1RawCountsPt[sigmodel][bkgmodel].SetBinError(iBin, h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin) / math.sqrt(NEVENTS*n_run))
            
            raws.append([])
            errs.append([])
            print(ranges['SCAN'][iBin - 1][0]," ", ranges['SCAN'][iBin - 1][1]," ", ranges['SCAN'][iBin - 1][2])
            for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                if eff == 1.000:
                    continue
                h1Counts = results_file.Get(f'RawCounts{eff:.3f}_{sigmodel}_{bkgmodel}')
               # h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin) / NEVENTS
                raws[iBin-1].append(h1Counts.GetBinContent(iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin)/ NEVENTS)
                errs[iBin-1].append(h1Counts.GetBinError(iBin) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin)/ math.sqrt(NEVENTS*n_run) )


        distribution.cd()
        h1RawCounts[sigmodel][bkgmodel].Fit(mt_distr, "0IRM+", "",MT_BINS[0],MT_BINS[-1])
        fit_function = h1RawCounts[sigmodel][bkgmodel].GetFunction("mt_distr")
        fit_function.SetLineColor(ROOT.kOrange)
        h1RawCounts[sigmodel][bkgmodel].Write()
        #hRawCounts.append(h1RawCounts[sigmodel][bkgmodel])

        cvDir.cd()
        myCv = TCanvas(f"mT_SpectraCv_{sigmodel}_{bkgmodel}")
        gPad.SetLeftMargin(0.15); 
                    
        min_value = h1RawCounts[sigmodel][bkgmodel].GetMinimum()*0.8
        max_value = h1RawCounts[sigmodel][bkgmodel].GetMaximum()*1.2
        
        h1RawCounts[sigmodel][bkgmodel].GetYaxis().SetRangeUser(min_value, max_value)
        pinfo = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(30+3)
        pinfo.SetTextFont(42)
        err_decimal = pu.get_decimal(mt_distr.GetParError(1)*1000)
        string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  0-5%'
        pinfo.AddText(string)
        string = 'T = {:.'+f'{err_decimal}'+'f} #pm {:.'+f'{err_decimal}'+'f} MeV '
        string = string.format(
            mt_distr.GetParameter(1)*1000, mt_distr.GetParError(1)*1000)
        pinfo.AddText(string)
        string = 'T_{gen}'+' = {:.2f} MeV'.format(T*1000)
        pinfo.AddText(string)
        if mt_distr.GetNDF() != 0:
            string = f'#chi^{{2}} / NDF = {(mt_distr.GetChisquare() / mt_distr.GetNDF()):.2f}'
        pinfo.AddText(string)
        h1RawCounts[sigmodel][bkgmodel].SetMarkerStyle(20)
        h1RawCounts[sigmodel][bkgmodel].SetMarkerColor(ROOT.kBlue)
        h1RawCounts[sigmodel][bkgmodel].SetLineColor(ROOT.kBlue)
        h1RawCounts[sigmodel][bkgmodel].SetStats(0)
        h1RawCounts[sigmodel][bkgmodel].Draw("ex0same")
        mt_distr.Draw("same")

        pinfo.Draw("x0same")
        myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+".png")
        myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+".pdf")
        tmpSyst = h1RawCounts[sigmodel][bkgmodel].Clone("hSyst")
        corSyst = h1RawCounts[sigmodel][bkgmodel].Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCounts[sigmodel][bkgmodel].GetNbinsX() + 1):
            val = h1RawCounts[sigmodel][bkgmodel].GetBinContent(iBin)

        distribution.cd()
        myCv.Write()

        h1RawCounts[sigmodel][bkgmodel].Draw("ex0same")
        pinfo.Draw()
        myCv.SetLogy()
        cvDir.cd()
        myCv.Write()

        ###########################################################################
        #h1RawCounts.UseCurrentStyle()

        h1RawCountsPt[sigmodel][bkgmodel].Fit(pt_distr, "MI0R+", "",PT_BINS[0],PT_BINS[-1])
        fit_function = h1RawCountsPt[sigmodel][bkgmodel].GetFunction("pt_distr")
        fit_function.SetLineColor(ROOT.kOrange)
        h1RawCountsPt[sigmodel][bkgmodel].GetXaxis().SetRangeUser(PT_BINS[0],PT_BINS[-1])
        h1RawCountsPt[sigmodel][bkgmodel].Write()
        hRawCounts.append(h1RawCountsPt[sigmodel][bkgmodel])

        cvDir.cd()
        myCv = TCanvas(f"pT_SpectraCv_{sigmodel}_{bkgmodel}")
        gPad.SetLeftMargin(0.15); 
                    
        min_value = h1RawCountsPt[sigmodel][bkgmodel].GetMinimum()*0.8
        max_value = h1RawCountsPt[sigmodel][bkgmodel].GetMaximum()*1.2
        
        h1RawCountsPt[sigmodel][bkgmodel].GetYaxis().SetRangeUser(min_value, max_value)
        pinfo = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(30+3)
        pinfo.SetTextFont(42)
        err_decimal = pu.get_decimal(pt_distr.GetParError(1)*1000)
        string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  0-5%'
        pinfo.AddText(string)
        string = 'T = {:.'+f'{3}'+'f} #pm {:.'+f'{3}'+'f} MeV '
        string = string.format(
            pt_distr.GetParameter(1)*1000, pt_distr.GetParError(1)*1000)
        pinfo.AddText(string)
        string = 'T_{gen}'+' = {:.2f} MeV'.format(T*1000)
        pinfo.AddText(string)
        if not SCALE:
            if pt_distr.GetNDF() != 0:
                string = f'#chi^{{2}} / NDF = {(pt_distr.GetChisquare() / pt_distr.GetNDF()):.2f}'
            pinfo.AddText(string)
        h1RawCountsPt[sigmodel][bkgmodel].SetMarkerStyle(20)
        h1RawCountsPt[sigmodel][bkgmodel].SetMarkerColor(ROOT.kBlue)
        h1RawCountsPt[sigmodel][bkgmodel].SetLineColor(ROOT.kBlue)
        h1RawCountsPt[sigmodel][bkgmodel].SetStats(0)
        h1RawCountsPt[sigmodel][bkgmodel].Draw("ex0same")
        pt_distr.Draw("same")

        myCv.SetLogy()
        pinfo.Draw("x0same")
        myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+".png")
        myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+".pdf")
        tmpSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hSyst")
        corSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCountsPt[sigmodel][bkgmodel].GetNbinsX() + 1):
            val = h1RawCountsPt[sigmodel][bkgmodel].GetBinContent(iBin)

        distribution.cd()
        myCv.Write()

        h1RawCountsPt[sigmodel][bkgmodel].Draw("ex0same")
        pinfo.Draw()
        myCv.SetLogy()
        cvDir.cd()
        myCv.Write()

distribution.cd()

syst = TH1D("syst", ";T (MeV);Entries", 100, T*0.98*1000, T*1.02*1000)
prob = TH1D("prob", ";Fit probability;Entries", 100, 0, 1)
pars = TH2D("pars", ";T (MeV);Normalisation;Entries", 600, T*0.8*1000, T*1.2*1000, 300, 700000, 3000000)
tmpCt = hRawCounts[0].Clone("tmpCt")

combinations = set()
size = 10000
count=0
for _ in range(size):
    tmpCt.Reset()
    comboList = []

    for iBin in range(1, tmpCt.GetNbinsX() + 1):
        index = random.randint(0, len(raws[iBin-1])-1)
        comboList.append(index)
        tmpCt.SetBinContent(iBin, raws[iBin-1][index])
        tmpCt.SetBinError(iBin, errs[iBin-1][index])

    combo = (x for x in comboList)
    if combo in combinations:
        continue
    #pt_distr.SetParameter(0, 1.15104e+03)
    combinations.add(combo)
    tmpCt.Fit(pt_distr, "QIR0+","",PT_BINS[0],PT_BINS[-2])
    prob.Fill(pt_distr.GetProb())
    syst.Fill(pt_distr.GetParameter(1)*1000)
    pars.Fill(pt_distr.GetParameter(1)*1000, pt_distr.GetParameter(0))

syst.SetFillColor(600)
syst.SetFillStyle(3345)

print(syst.GetMaximum()*1.2)
max_counts = syst.GetMaximum()*1.2
syst.GetYaxis().SetRangeUser(0, max_counts)
syst.Write()
prob.Write()
pars.Write()

myCv = TCanvas(f"cv_syst")
gPad.SetLeftMargin(0.15)
syst.Draw()        
true_value_line = TLine(T*1000, 0, T*1000, max_counts)
true_value_line.SetLineColor(ROOT.kRed)
true_value_line.SetLineStyle(7)
true_value_line.Draw("same")
myCv.SaveAs(resultsSysDir+"/pT_slope_syst_"+FILE_PREFIX+".pdf")
myCv.SaveAs(resultsSysDir+"/pT_slope_syst_"+FILE_PREFIX+".png")
results_file.Close()

print('')
print(f'--- pt distribution fitting in {((time.time() - start_time) / 60):.2f} minutes ---')