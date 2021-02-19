#!/usr/bin/env python3

import argparse
import math
import os
import random
from array import array
from multiprocessing import Pool

import numpy as np
import yaml
import plot_utils as pu

from ROOT import (TF1, TH1D, TH2D, TAxis, TCanvas, TColor, TFile, TFrame,
                  TIter, TKey, TPaveText, gDirectory, gPad, gROOT, gStyle,
                  kBlue, kRed, TDatabasePDG)
import ROOT
from scipy import stats

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
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-s', '--scale', help='Scale the results to a complete run', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

EINT = pu.get_sNN(params['EINT'])
T = params['T']
PDG_CODE = params['PDG']
PT_BINS = params['PT_BINS']

if args.split:
    SPLIT_LIST = ['_matter', '_antimatter']
else:
    SPLIT_LIST = ['']

EVENT_PATH = os.path.expandvars(params['EVENT_PATH'])

SCALE = args.scale

gROOT.SetBatch()

background_file = ROOT.TFile(EVENT_PATH)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
background_file.Close()
rate = 75000 #hz
running_time = 30#days

pt_distr = TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x*x+[1])/([2]))",PT_BINS[1],PT_BINS[-1])

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
pt_distr.FixParameter(1,mass)
pt_distr.SetParameter(2,T/1000)
pt_distr.SetParLimits(2,0.270,0.350)


resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

var = '#it{p}_{T}'
unit = 'GeV/#it{c}'

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_dist.root'
distribution = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + params['FILE_PREFIX'] + '_results_fit.root'
results_file = TFile(file_name, 'read')

bkgModels = params['BKG_MODELS'] if 'BKG_MODELS' in params else ['pt_distr']
hist_list = []

for split in SPLIT_LIST:
    for cclass in params['CENTRALITY_CLASS']:
        if SCALE:
            n_run = rate*running_time*24*60*60*(cclass[1]-cclass[0])
        else:
            n_run = n_ev
            
        inDirName = f'{cclass[0]}-{cclass[1]}' + split
        
        h2BDTEff = results_file.Get(f'{inDirName}/BDTeff')
        h1BDTEff = h2BDTEff.ProjectionX("bdteff", 1, 1)

        best_sig = np.round(np.array(h1BDTEff)[1:-1], 2)
        best_sig[0]=0.06
        best_sig[1]=0.13
        sig_ranges = []
        for i in best_sig:
            if i== best_sig[0]:
                sig_ranges.append([i-0.03, i+0.03, 0.01])
            else:
                sig_ranges.append([i-0.1, i+0.1, 0.01])

        ranges = {
                'BEST': best_sig,
                'SCAN': sig_ranges
        }
        print("BDT efficiencies: ",best_sig)

        results_file.cd(inDirName)
        out_dir = distribution.mkdir(inDirName)
        cvDir = out_dir.mkdir("canvas")

        h2PreselEff = results_file.Get(f'{inDirName}/PreselEff')
        h1PreselEff = h2PreselEff.ProjectionX("preseleff", 1, 1)

        for i in range(1, h1PreselEff.GetNbinsX() + 1):
            h1PreselEff.SetBinError(i, 0)

        h1PreselEff.SetTitle(f';{var} ({unit}); Preselection efficiency')
        h1PreselEff.UseCurrentStyle()
        h1PreselEff.SetMinimum(0)
        out_dir.cd()
        h1PreselEff.Write("h1PreselEff" + split)

        hRawCounts = []
        raws = []
        errs = []

        for model in bkgModels:
            h1RawCounts = h1PreselEff.Clone(f"best_{model}")
            h1RawCounts.GetYaxis().SetTitle("1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]")
            h1RawCounts.Reset()

            out_dir.cd()

            for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
                h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{ranges["BEST"][iBin-1]:.2f}_{model}')

                h1RawCounts.SetBinContent(iBin, h2RawCounts.GetBinContent(
                    iBin,1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin) / n_ev)
                h1RawCounts.SetBinError(iBin, h2RawCounts.GetBinError(
                    iBin,1) / h1PreselEff.GetBinContent(iBin) / ranges['BEST'][iBin-1] / h1RawCounts.GetBinWidth(iBin) / math.sqrt(n_ev*n_run))
                raws.append([])
                errs.append([])

                for eff in np.arange(ranges['SCAN'][iBin - 1][0], ranges['SCAN'][iBin - 1][1], ranges['SCAN'][iBin - 1][2]):
                    if eff > 0.79:
                        continue

                    h2RawCounts = results_file.Get(f'{inDirName}/RawCounts{eff:.2f}_{model}')
                    raws[iBin-1].append(h2RawCounts.GetBinContent(iBin,1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin) )
                    errs[iBin-1].append(h2RawCounts.GetBinError(iBin,1) / h1PreselEff.GetBinContent(iBin) / eff / h1RawCounts.GetBinWidth(iBin) )


            out_dir.cd()
            #h1RawCounts.UseCurrentStyle()
            if(split!=""):
                if(model=="pol2"):
                    hist_list.append(h1RawCounts.Clone("hist"+split))
            h1RawCounts.Fit(pt_distr, "M0+", "",0.5,2.5)
            fit_function = h1RawCounts.GetFunction("pt_distr")
            fit_function.SetLineColor(kOrangeC)
            h1RawCounts.Write()
            hRawCounts.append(h1RawCounts)

            cvDir.cd()
            myCv = TCanvas(f"ctSpectraCv_{model}{split}")
            gPad.SetLeftMargin(0.15); 
            h1RawCounts.GetXaxis().SetRangeUser(0.5,2.5)
            h1RawCounts.GetYaxis().SetRangeUser(0,h1RawCounts.GetMaximum()*1.25)
            pinfo = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)

            string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  {cclass[0]}-{cclass[1]}%'
            pinfo.AddText(string)
            string = 'T = {:.3f} #pm {:.3f} MeV '.format(
                pt_distr.GetParameter(2)*1000, pt_distr.GetParError(2)*1000)
            pinfo.AddText(string)
            if pt_distr.GetNDF() != 0:
                string = f'#chi^{{2}} / NDF = {(pt_distr.GetChisquare() / pt_distr.GetNDF()):.2f}'
            pinfo.AddText(string)
            h1RawCounts.SetMarkerStyle(20)
            h1RawCounts.SetMarkerColor(ROOT.kBlue)
            h1RawCounts.SetLineColor(ROOT.kBlue)
            h1RawCounts.SetStats(0)
            h1RawCounts.Draw("ex0same")
            pt_distr.Draw("same")
            #fit_function.Draw("same")
            #h1RawCounts.SetMinimum(0.001)
            #h1RawCounts.SetMaximum(1000)
            #frame.GetYaxis().SetTitleSize(26)
            #frame.GetYaxis().SetLabelSize(22)
            #frame.GetXaxis().SetTitleSize(26)
            #frame.GetXaxis().SetLabelSize(22)

            pinfo.Draw("x0same")
            myCv.SaveAs("pT_spectra_"+model+".png")
            myCv.SaveAs("pT_spectra_"+model+".pdf")
            tmpSyst = h1RawCounts.Clone("hSyst")
            corSyst = h1RawCounts.Clone("hCorr")
            tmpSyst.SetFillStyle(0)
            #tmpSyst.SetMinimum(0.001)
            #tmpSyst.SetMaximum(1000)
            corSyst.SetFillStyle(3345)
            for iBin in range(1, h1RawCounts.GetNbinsX() + 1):
                val = h1RawCounts.GetBinContent(iBin)

            out_dir.cd()
            myCv.Write()

            h1RawCounts.Draw("ex0same")
            pinfo.Draw()
            cvDir.cd()
            myCv.Write()

        out_dir.cd()

        syst = TH1D("syst", ";T (MeV);Entries", 300, 270, 350)
        prob = TH1D("prob", ";Fit probability;Entries", 100, 0, 1)
        pars = TH2D("pars", ";T (MeV);Normalisation;Entries", 300, 270, 350, 300, 700000, 3000000)
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

            combinations.add(combo)
            tmpCt.Fit(pt_distr, "M0+","",0.5,2.5)
            prob.Fill(pt_distr.GetProb())
            if pt_distr.GetChisquare() < 3 * pt_distr.GetNDF():
                #if(pt_distr.GetParameter(2))>270 and count==0:
                #    tmpCt.Write()
                #    count=1
                syst.Fill(pt_distr.GetParameter(2)*1000)
                pars.Fill(pt_distr.GetParameter(2)*1000, pt_distr.GetParameter(0))

        syst.SetFillColor(600)
        syst.SetFillStyle(3345)
        syst.Write()
        prob.Write()
        pars.Write()

results_file.Close()