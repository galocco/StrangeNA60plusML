#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array

import numpy as np
import yaml
import math
import analysis_utils as au
import plot_utils as pu
import pandas as pd

import ROOT
from ROOT import TFile, gROOT, TDatabasePDG, TF1, TCanvas, TColor, TPaveText, gPad

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

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-s', '--scale', help='Scale the results to a complete run', action='store_true')
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
GAUSS = params['GAUSS']
MASS_WINDOW = params['MASS_WINDOW']
PT_BINS = params['PT_BINS']
EINT = params['EINT']
BKG_MODELS = params['BKG_MODELS']
STD_SELECTION = params['STD_SELECTION']
T = params['T']

PEAK_MODE = args.peak
SCALE = args.scale

for index in range(0,len(params['EVENT_PATH'])):

    ###############################################################################
    # define paths for loading data
    data_path = os.path.expandvars(params['DATA_PATH'][index])
    event_path = os.path.expandvars(params['EVENT_PATH'][index])
    BKG_MODELS = params['BKG_MODELS']

    results_dir = f"../Results"

    ###############################################################################
    start_time = time.time()                          # for performances evaluation

    resultsSysDir = os.environ['HYPERML_RESULTS']

    file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_std_results.root'
    results_file = TFile(file_name, 'recreate')

    mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
    cv = ROOT.TCanvas("cv","cv")

    background_file = ROOT.TFile(event_path)
    hist_ev = background_file.Get('hNevents')
    n_ev = hist_ev.GetBinContent(1)
    background_file.Close()
    rate = 75000 #hz
    running_time = 30#days
    if SCALE:
        n_run = rate*running_time*24*60*60*5
    else:
        n_run = n_ev

    hnsparse = au.get_skimmed_large_data_std_hsp(mass, data_path, PT_BINS, STD_SELECTION, MASS_WINDOW)
    results_file.cd()
    hnsparse.Write()


    resultsSysDir = os.environ['HYPERML_RESULTS']+"/"+FILE_PREFIX
    file_name = resultsSysDir + '/' + FILE_PREFIX + '_results_fit.root'
    efficiency_file = TFile(file_name, 'read')

    h1PreselEff = efficiency_file.Get('0-5/PreselEff')


    MT_BINS = au.pt_array_to_mt_m0_array(PT_BINS, mass)
    binning = array('d',MT_BINS)
    pt_binning = array('d',PT_BINS)
    mt_distr = TF1("mt_distr", "[0]*exp(-x/[1])",MT_BINS[0],MT_BINS[-1])
    mt_distr.SetParameter(1,T)
    mt_distr.SetParLimits(1,T*0.8,T*1.2)

    pt_distr = TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",PT_BINS[0],PT_BINS[-1])
    pt_distr.SetParameter(1,T)
    pt_distr.SetParLimits(1,T*0.8,T*1.2)
    pt_distr.FixParameter(2,mass)

    h1RawCounts = {}
    h1RawCountsPt = {}
    for bkgmodel in BKG_MODELS:
        h1RawCounts[bkgmodel] = ROOT.TH1D(f"mt_best_{bkgmodel}",";m_{T}-m_{0} [GeV];1/N_{ev}1/m_{T}dN/dm_{T} [GeV^{-2}]",len(PT_BINS)-1,binning)
        h1RawCountsPt[bkgmodel] = ROOT.TH1D(f"pt_best_{bkgmodel}",";#it{p}_{T} [GeV/#it{c}];1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",len(PT_BINS)-1,pt_binning)

    cent_dir = results_file.mkdir('0-5')
    cent_dir.cd()
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        sub_dir = cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
        sub_dir.cd()
        h1_minv = au.h1_from_sparse_std(hnsparse, ptbin, f'pt_{ptbin[0]}{ptbin[1]}')
        
        for bkgmodel in BKG_MODELS:
            fit_dir = sub_dir.mkdir(bkgmodel)
            fit_dir.cd()
            rawcounts, err_rawcounts, _, _, _, _, _, _ = au.fit_hist(h1_minv, ptbin, mass, model=bkgmodel, Eint=EINT, peak_mode=PEAK_MODE, gauss=GAUSS, mass_range=MASS_WINDOW)
            index = PT_BINS.index(ptbin[0])+1
            h1RawCounts[bkgmodel].SetBinContent(index, rawcounts/h1RawCounts[bkgmodel].GetBinWidth(index)/h1PreselEff.GetBinContent(index)/ n_ev)
            h1RawCountsPt[bkgmodel].SetBinContent(index, rawcounts/h1RawCountsPt[bkgmodel].GetBinWidth(index)/h1PreselEff.GetBinContent(index) / math.sqrt(n_ev*n_run))            
            h1RawCounts[bkgmodel].SetBinError(index, err_rawcounts/h1RawCounts[bkgmodel].GetBinWidth(index)/h1PreselEff.GetBinContent(index)/ n_ev)
            h1RawCountsPt[bkgmodel].SetBinError(index, err_rawcounts/h1RawCountsPt[bkgmodel].GetBinWidth(index)/h1PreselEff.GetBinContent(index) / math.sqrt(n_ev*n_run))
        h1_minv.Write()
    results_file.cd()
    for bkgmodel in BKG_MODELS:
        ####################################################################
        ## 1/Nev*dN/dpt vs pt
        ####################################################################
        h1RawCountsPt[bkgmodel].Fit(pt_distr, "M0+", "",PT_BINS[0],PT_BINS[-1])
        fit_function = h1RawCountsPt[bkgmodel].GetFunction("pt_distr")
        fit_function.SetLineColor(kOrangeC)
        myCv = TCanvas(f"pT_SpectraCv_{bkgmodel}")
        gPad.SetLeftMargin(0.15); 
                    
        min_value = h1RawCountsPt[bkgmodel].GetMinimum()*0.8
        max_value = h1RawCountsPt[bkgmodel].GetMaximum()*1.2
        
        h1RawCountsPt[bkgmodel].GetYaxis().SetRangeUser(min_value, max_value)
        pinfo = TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
        pinfo.SetBorderSize(0)
        pinfo.SetFillStyle(0)
        pinfo.SetTextAlign(30+3)
        pinfo.SetTextFont(42)
        err_decimal = pu.get_decimal(pt_distr.GetParError(1)*1000)
        string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  0-5%'
        pinfo.AddText(string)
        string = 'T = {:.'+f'{5}'+'f} #pm {:.'+f'{5}'+'f} MeV '
        string = string.format(
            pt_distr.GetParameter(1)*1000, pt_distr.GetParError(1)*1000)
        pinfo.AddText(string)
        string = 'T_{gen}'+' = {:.2f} MeV'.format(T*1000)
        pinfo.AddText(string)
        if not SCALE:
            if pt_distr.GetNDF() != 0:
                string = f'#chi^{{2}} / NDF = {(pt_distr.GetChisquare() / pt_distr.GetNDF()):.2f}'
            pinfo.AddText(string)
        h1RawCountsPt[bkgmodel].SetMarkerStyle(20)
        h1RawCountsPt[bkgmodel].SetMarkerColor(ROOT.kBlue)
        h1RawCountsPt[bkgmodel].SetLineColor(ROOT.kBlue)
        h1RawCountsPt[bkgmodel].SetStats(0)
        h1RawCountsPt[bkgmodel].Draw("ex0same")
        pt_distr.Draw("same")

        pinfo.Draw("x0same")
        myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+bkgmodel+"_std.png")
        myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+bkgmodel+"_std.pdf")
        tmpSyst = h1RawCountsPt[bkgmodel].Clone("hSyst")
        corSyst = h1RawCountsPt[bkgmodel].Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCountsPt[bkgmodel].GetNbinsX() + 1):
            val = h1RawCountsPt[bkgmodel].GetBinContent(iBin)

        results_file.cd()
        h1RawCountsPt[bkgmodel].Draw("ex0same")
        pinfo.Draw()
        myCv.Write()
        h1RawCountsPt[bkgmodel].Write()
        ####################################################################
        ## 1/(Nev*mT)*dN/dmT vs mT-m0
        ####################################################################
        h1RawCounts[bkgmodel].Fit(mt_distr, "M0+", "",MT_BINS[0],MT_BINS[-1])
        fit_function = h1RawCounts[bkgmodel].GetFunction("mt_distr")
        fit_function.SetLineColor(kOrangeC)

        myCv = TCanvas(f"mT_SpectraCv_{bkgmodel}")
        gPad.SetLeftMargin(0.15); 
                    
        min_value = h1RawCounts[bkgmodel].GetMinimum()*0.8
        max_value = h1RawCounts[bkgmodel].GetMaximum()*1.2
        
        h1RawCounts[bkgmodel].GetYaxis().SetRangeUser(min_value, max_value)
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
        h1RawCounts[bkgmodel].SetMarkerStyle(20)
        h1RawCounts[bkgmodel].SetMarkerColor(ROOT.kBlue)
        h1RawCounts[bkgmodel].SetLineColor(ROOT.kBlue)
        h1RawCounts[bkgmodel].SetStats(0)
        h1RawCounts[bkgmodel].Draw("ex0same")
        mt_distr.Draw("same")

        pinfo.Draw("x0same")
        myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+bkgmodel+"_std.png")
        myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+bkgmodel+"_std.pdf")
        tmpSyst = h1RawCounts[bkgmodel].Clone("hSyst")
        corSyst = h1RawCounts[bkgmodel].Clone("hCorr")
        tmpSyst.SetFillStyle(0)
        corSyst.SetFillStyle(3345)
        for iBin in range(1, h1RawCounts[bkgmodel].GetNbinsX() + 1):
            val = h1RawCounts[bkgmodel].GetBinContent(iBin)

        h1RawCounts[bkgmodel].Draw("ex0same")
        pinfo.Draw()
        myCv.Write()
        h1RawCounts[bkgmodel].Write()
    results_file.Close()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

