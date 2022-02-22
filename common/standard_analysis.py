#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from array import array
import uproot
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
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-t', '--test', help='Test mode', action='store_true')
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
MASS_WINDOW = params['MASS_WINDOW']
PT_BINS = params['PT_BINS']
EINT = params['EINT']
NBINS = params['NBINS']
BKG_MODELS = params['BKG_MODELS']
SIG_MODELS = params['SIG_MODELS']
T = params['T']
MULTIPLICITY = params['MULTIPLICITY']
bratio = params['BRATIO']
SCALE = args.scale
TEST = args.test
results_dir = os.environ['RESULTS']
mc_file_name = params['MC_PATH']

if TEST:
    file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_sig_only_results.root'
    STD_SELECTION = params['STD_SELECTION']
else:
    file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_std_results.root'
    STD_SELECTION = params['STD_SELECTION']

mc_fit_file_name = results_dir + '/' + FILE_PREFIX + f'/{FILE_PREFIX}_mc_fit.root'
mc_fit_file = TFile(mc_fit_file_name,"read")

results_file = TFile(file_name, 'recreate')
binning = array("d", PT_BINS)
print("binning: ",len(binning)-1," -- ",binning)
h1PreselEff = ROOT.TH1D("PreselEff","Total efficiency;#it{p}_{T} (GeV/#it{c}); efficiency",len(binning)-1,binning)
h1SelectEff = ROOT.TH1D("SelectEff","Rec x Acc;#it{p}_{T} (GeV/#it{c}); efficiency",len(binning)-1,binning)
h1AfterEff = ROOT.TH1D("AfterEff","Selection efficiency;#it{p}_{T} (GeV/#it{c}); efficiency",len(binning)-1,binning)
h1Gen = ROOT.TH1D("h1Gen",";#it{p}_{T} (GeV/#it{c}); efficiency",len(binning)-1,binning)
h1RecPt = ROOT.TH1D("h1RecPt", ";#it{p}_{T} (GeV/#it{c}); efficiency", 1, 0, 5)
h1GenPt = ROOT.TH1D("h1GenPt", ";#it{p}_{T} (GeV/#it{c}); efficiency", 1, 0, 5)

df_rec = uproot.open(mc_file_name)['ntcand'].arrays(library="pd").query(STD_SELECTION)
#for pt in df_rec['pt']:
#df_rec = df_rec
for pt in df_rec['pt']:
    h1PreselEff.Fill(pt)
    h1AfterEff.Fill(pt)
    h1SelectEff.Fill(pt)
    h1RecPt.Fill(pt)

del df_rec

df_gen = uproot.open(mc_file_name)['ntgen'].arrays(library="pd")

for pt in df_gen['pt']:
    h1Gen.Fill(pt)
    h1GenPt.Fill(pt)
h1GenPt.Write()
h1RecPt.Write()
for iBin in range(1, len(binning)):
    counts_num = h1PreselEff.GetBinContent(iBin)
    counts_den = h1Gen.GetBinContent(iBin)
    eff = counts_num/counts_den
    h1PreselEff.SetBinContent(iBin, eff)
    h1PreselEff.SetBinError(iBin, ROOT.TMath.Sqrt(eff*(1-eff)/counts_den))

    counts_num = h1AfterEff.GetBinContent(iBin)
    counts_den = h1SelectEff.GetBinContent(iBin)
    eff = counts_num/counts_den
    h1AfterEff.SetBinContent(iBin, eff)
    h1AfterEff.SetBinError(iBin, ROOT.TMath.Sqrt(eff*(1-eff)/counts_den))

    counts_num = h1SelectEff.GetBinContent(iBin)
    counts_den = h1Gen.GetBinContent(iBin)
    eff = counts_num/counts_den
    h1SelectEff.SetBinContent(iBin, eff)
    h1SelectEff.SetBinError(iBin, ROOT.TMath.Sqrt(eff*(1-eff)/counts_den))

h1AfterEff.Write()
h1PreselEff.Write()
h1SelectEff.Write()
if True:
    del df_gen

    if TEST:
        STD_SELECTION += " and true > 0.5"

    resultsSysDir = os.environ['RESULTS']+"/"+FILE_PREFIX

    mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
    pt_distr_gen = TF1("pt_distr_gen", "x*exp(-TMath::Sqrt(x**2+[1]**2)/[0])",PT_BINS[0],10)
    pt_distr_gen.FixParameter(0,T)
    pt_distr_gen.FixParameter(1,mass)
    print(PT_BINS[0]," - ", PT_BINS[-1])
    print(au.get_pt_integral(pt_distr_gen, 0, PT_BINS[0]))
    print(au.get_pt_integral(pt_distr_gen, PT_BINS[0], PT_BINS[-1]))
    print(au.get_pt_integral(pt_distr_gen))
    pt_range_factor = au.get_pt_integral(pt_distr_gen, PT_BINS[0], PT_BINS[-1])/au.get_pt_integral(pt_distr_gen)

    ###############################################################################
    # define paths for loading data
    if TEST:
        data_path = os.path.expandvars(params['DATA_PATH'])
    else:
        data_path = os.path.expandvars(params['DATA_PATH'])
    NEVENTS = params['NEVENTS']
    BKG_MODELS = params['BKG_MODELS']


    ###############################################################################
    start_time = time.time()                          # for performances evaluation

    cv = ROOT.TCanvas("cv","cv")

    rate = 75000 #hz
    running_time = 30#days
    if SCALE:
        n_run = rate*running_time*24*60*60*5
    else:
        n_run = NEVENTS

    hnsparse = au.get_skimmed_large_data_std_hsp(mass, data_path, PT_BINS, STD_SELECTION, MASS_WINDOW, NBINS)
    results_file.cd()
    hnsparse.Write()



    MT_BINS = au.pt_array_to_mt_m0_array(PT_BINS, mass)
    MT_BINS_M = au.pt_array_to_mt_array(PT_BINS, mass)
    binning = array('d', MT_BINS)
    pt_binning = array('d', PT_BINS)
    mt_distr = TF1("mt_distr", "[0]*exp(-(x-[2])/[1])",MT_BINS[0],MT_BINS[-1])
    mt_distr.SetParameter(0, 10.)
    mt_distr.SetParameter(1,T)
    mt_distr.SetParLimits(1,T*0.8,T*1.2)
    mt_distr.FixParameter(2,mass)


    pt_distr = TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",PT_BINS[0],PT_BINS[-1])
    pt_distr.SetParameter(1,T)
    pt_distr.SetParLimits(1,T*0.8,T*1.2)
    pt_distr.FixParameter(2,mass)

    h1RawCounts = {}
    h1RawCountsPt = {}

    for sigmodel in SIG_MODELS:
        h1RawCounts[sigmodel] = {}
        h1RawCountsPt[sigmodel] = {}
        for bkgmodel in BKG_MODELS:
            h1RawCounts[sigmodel][bkgmodel] = ROOT.TH1D(f"mt_best_{bkgmodel}",";m_{T}-m_{0} [GeV];1/N_{ev}1/m_{T}dN/dm_{T} [GeV^{-2}]",len(PT_BINS)-1,binning)
            h1RawCountsPt[sigmodel][bkgmodel] = ROOT.TH1D(f"pt_best_{bkgmodel}",";#it{p}_{T} [GeV/#it{c}];1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",len(PT_BINS)-1,pt_binning)

    results_file.cd()
    iBin = 0
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        sub_dir = results_file.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
        sub_dir.cd()
        h1_minv = au.h1_from_sparse_std(hnsparse, ptbin, f'pt_{ptbin[0]}{ptbin[1]}')
        
        for sigmodel in SIG_MODELS:
            fit_sig_dir = sub_dir.mkdir(sigmodel)
            for bkgmodel in BKG_MODELS:
                fit_bkg_dir = fit_sig_dir.mkdir(bkgmodel)
                fit_bkg_dir.cd()
                index = PT_BINS.index(ptbin[0])+1
                if TEST:
                    rawcounts = h1_minv.GetEntries()
                    err_rawcounts = ROOT.TMath.Sqrt(rawcounts)
                    h1_minv.Write()
                else:
                    rawcounts, err_rawcounts = au.fit_hist(h1_minv, ptbin, mass, sig_model=sigmodel, bkg_model=bkgmodel, Eint=EINT, mass_range=MASS_WINDOW, mc_fit_file=mc_fit_file, directory = fit_bkg_dir, fix_params = True)

                h1RawCounts[sigmodel][bkgmodel].SetBinContent(index, rawcounts / h1RawCounts[sigmodel][bkgmodel].GetBinWidth(index) / h1PreselEff.GetBinContent(index) / (MT_BINS_M[iBin]-MT_BINS_M[iBin-1])/ (MT_BINS_M[iBin-1]+MT_BINS_M[iBin])/2 / NEVENTS)
                h1RawCountsPt[sigmodel][bkgmodel].SetBinContent(index, rawcounts / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(index) / h1PreselEff.GetBinContent(index)/ NEVENTS)            
                h1RawCounts[sigmodel][bkgmodel].SetBinError(index, err_rawcounts / h1RawCounts[sigmodel][bkgmodel].GetBinWidth(index) / h1PreselEff.GetBinContent(index) / (MT_BINS_M[iBin]-MT_BINS_M[iBin-1])/ (MT_BINS_M[iBin-1]+MT_BINS_M[iBin])/2 / NEVENTS)
                h1RawCountsPt[sigmodel][bkgmodel].SetBinError(index, err_rawcounts / h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(index) / h1PreselEff.GetBinContent(index) / NEVENTS)

            h1_minv.Write()
        iBin += 1
    results_file.cd()
    for sigmodel in SIG_MODELS:
        for bkgmodel in BKG_MODELS:
            ####################################################################
            ## 1/Nev*dN/dpt vs pt
            ####################################################################
            h1RawCountsPt[sigmodel][bkgmodel].Fit(pt_distr, "MI0+", "",PT_BINS[0],PT_BINS[-1])
            fit_function = h1RawCountsPt[sigmodel][bkgmodel].GetFunction("pt_distr")
            fit_function.SetLineColor(ROOT.kOrange)
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
            h1RawCountsPt[sigmodel][bkgmodel].SetMarkerStyle(20)
            h1RawCountsPt[sigmodel][bkgmodel].SetMarkerColor(ROOT.kBlue)
            h1RawCountsPt[sigmodel][bkgmodel].SetLineColor(ROOT.kBlue)
            h1RawCountsPt[sigmodel][bkgmodel].SetStats(0)
            h1RawCountsPt[sigmodel][bkgmodel].Draw("ex0same")
            pt_distr.Draw("same")

            mult = 0 
            err_mult = 0
                
            iBin = 0
            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                iBin += 1
                mult += h1RawCountsPt[sigmodel][bkgmodel].GetBinContent(iBin)*h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin)
                err_mult += (h1RawCountsPt[sigmodel][bkgmodel].GetBinError(iBin)*h1RawCountsPt[sigmodel][bkgmodel].GetBinWidth(iBin))**2

            err_mult = ROOT.TMath.Sqrt(err_mult)/pt_range_factor/bratio
            mult /= pt_range_factor*bratio

            print("pt_range_factor: ",pt_range_factor)
            print("bratio: ",bratio)
            print("multiplicity: ", mult," +- ",err_mult)
            print("multiplicity gen: ", MULTIPLICITY)
            print("z_gauss: ", (MULTIPLICITY-mult)/err_mult)
            print("**************************************************")
        
            pinfo.Draw("x0same")
            myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+"_std.png")
            myCv.SaveAs(resultsSysDir+"/pT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+"_std.pdf")
            tmpSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hSyst")
            corSyst = h1RawCountsPt[sigmodel][bkgmodel].Clone("hCorr")
            tmpSyst.SetFillStyle(0)
            corSyst.SetFillStyle(3345)
            for iBin in range(1, h1RawCountsPt[sigmodel][bkgmodel].GetNbinsX() + 1):
                val = h1RawCountsPt[sigmodel][bkgmodel].GetBinContent(iBin)

            results_file.cd()
            h1RawCountsPt[sigmodel][bkgmodel].Draw("ex0same")
            pinfo.Draw()
            myCv.Write()
            h1RawCountsPt[sigmodel][bkgmodel].Write()
            ####################################################################
            ## 1/(Nev*mT)*dN/dmT vs mT-m0
            ####################################################################
            h1RawCounts[sigmodel][bkgmodel].Fit(mt_distr, "MI0+", "",MT_BINS[0],MT_BINS[-1])
            fit_function = h1RawCounts[sigmodel][bkgmodel].GetFunction("mt_distr")
            fit_function.SetLineColor(kOrangeC)

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
            myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+"_std.png")
            myCv.SaveAs(resultsSysDir+"/mT_spectra_"+FILE_PREFIX+"_"+sigmodel+"_"+bkgmodel+"_std.pdf")
            tmpSyst = h1RawCounts[sigmodel][bkgmodel].Clone("hSyst")
            corSyst = h1RawCounts[sigmodel][bkgmodel].Clone("hCorr")
            tmpSyst.SetFillStyle(0)
            corSyst.SetFillStyle(3345)
            for iBin in range(1, h1RawCounts[sigmodel][bkgmodel].GetNbinsX() + 1):
                val = h1RawCounts[sigmodel][bkgmodel].GetBinContent(iBin)

            h1RawCounts[sigmodel][bkgmodel].Draw("ex0same")
            pinfo.Draw()
            myCv.Write()
            h1RawCounts[sigmodel][bkgmodel].Write()
            
    results_file.Close()
    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

