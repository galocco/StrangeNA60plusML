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
from ROOT import TFile, gROOT, TDatabasePDG, TF1, TH1D
from analysis_classes import ModelApplication

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')
parser.add_argument('-f', '--full', help='Run with the full simulation', action='store_true')
parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
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
N_BODY = params['NBODY']
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
MULTIPLICITY = params['MULTIPLICITY']
BRATIO = params['BRATIO']
EINT = pu.get_sNN(params['EINT'])
T = params['T']
EFF = params['EFF']
SIGMA = params['SIGMA']

CENT_CLASSES = params['CENTRALITY_CLASS']
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
COLUMNS = params['TRAINING_COLUMNS']

LARGE_DATA = params['LARGE_DATA']
LOAD_LARGE_DATA = params['LOAD_LARGE_DATA']
PRESELECTION = params['PRESELECTION']
SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

FULL_SIM = args.full
PEAK_MODE = args.peak

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
if FULL_SIM:
    bkg_path = os.path.expandvars(params['BKG_PATH'])
else:
    bkg_path = os.path.expandvars(params['BKG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])
if FULL_SIM:
    data_sig_path = os.path.expandvars(params['DATA_PATH'])
    event_path = os.path.expandvars(params['EVENT_PATH_FULL'])
else:
    data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
    event_path = os.path.expandvars(params['EVENT_PATH'])
BKG_MODELS = params['BKG_MODELS']

results_dir = f"../Results/2Body"

###############################################################################
start_time = time.time()                          # for performances evaluation

resultsSysDir = os.environ['HYPERML_RESULTS_{}'.format(params['NBODY'])]

file_name = results_dir + f'/{FILE_PREFIX}_std_results.root'
results_file = TFile(file_name, 'recreate')

file_name = resultsSysDir + '/' + FILE_PREFIX + '_results.root'
eff_file = TFile(file_name, 'read')

standard_selection = 'y > 0.5'
application_columns = ['pt','m','ct','centrality','score','y','cosp']
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

background_file = ROOT.TFile(event_path)
hist_ev = background_file.Get('hNevents')
n_ev = hist_ev.GetBinContent(1)
background_file.Close()
cv = ROOT.TCanvas("cv","cv")

for split in SPLIT_LIST:

    if LARGE_DATA:
        if LOAD_LARGE_DATA:
            df_skimmed = pd.read_parquet(os.path.dirname(data_sig_path) + f'/{FILE_PREFIX}skimmed_df.parquet.gzip')
        else:
            if FULL_SIM:
                df_skimmed = au.get_skimmed_large_data_full(data_sig_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split, FILE_PREFIX, PRESELECTION)
            else:
                df_skimmed = au.get_skimmed_large_data(MULTIPLICITY, BRATIO, EFF, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split, FILE_PREFIX, PRESELECTION)
                df_skimmed.to_parquet(os.path.dirname(data_sig_path) + f'/{FILE_PREFIX}skimmed_df.parquet.gzip', compression='gzip')

        ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, FULL_SIM, PRESELECTION, df_skimmed)

    else:
        ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, FULL_SIM, PRESELECTION)

    #initialize the histogram with the mass pol0 fit
    iBin = 0
    for cclass in CENT_CLASSES:
        inDirName = f'{cclass[0]}-{cclass[1]}' + split

        eff_file.cd(inDirName)
        h2PreselEff = eff_file.Get(f'{inDirName}/PreselEff')
        h1PreselEff = h2PreselEff.ProjectionX("preseleff", 1, 1)

        for i in range(1, h1PreselEff.GetNbinsX() + 1):
            h1PreselEff.SetBinError(i, 0)
        MT_BINS = au.pt_array_to_mt_m0_array(PT_BINS, mass)
        binning = array('d',MT_BINS)
        
        mt_spectra_counts_list = []
        for bkgmodel in BKG_MODELS:
            mt_spectra_counts = TH1D(f"mt_best_{bkgmodel}",";m_{T}-m_{0} [GeV];1/N_{ev}1/m_{T}dN/dm_{T} [GeV^{-2}]",len(PT_BINS)-1,binning)
            mt_spectra_counts_list.append(mt_spectra_counts)

        binning = array('d',PT_BINS)
        pt_spectra_counts_list = []
        for bkgmodel in BKG_MODELS:
            pt_spectra_counts = TH1D(f"pt_best_{bkgmodel}",";#it{p}_{T} [GeV/#it{c}];1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",len(PT_BINS)-1,binning)
            pt_spectra_counts_list.append(pt_spectra_counts)

        cent_dir = results_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')
        df_applied = ml_application.df_data.query(standard_selection)

        hist_rec = ml_application.get_pt_hist(PT_BINS)
        counts_eff, _ = np.histogram(df_applied.query("y > 0.5 and "+standard_selection)['pt'], len(PT_BINS)-1, range=[PT_BINS[0],PT_BINS[len(PT_BINS)-1]])
        cent_dir.cd()
        hist_eff = ROOT.TH1D('hist_eff', ';#it{p}_{T} (GeV/c);efficiency', len(PT_BINS)-1, PT_BINS[0], PT_BINS[len(PT_BINS)-1])
        for index in range(0, len(PT_BINS)-1):
            eff = counts_eff[index]/hist_rec.GetBinContent(index+1)
            if eff>1:
                eff = 1
            hist_eff.SetBinContent(index + 1, eff)
            hist_eff.SetBinError(index + 1, math.sqrt(eff*(1-eff)/hist_rec.GetBinContent(index+1)))
        hist_eff.Write()
        h1PreselEff.Write()
        iBin = 1
        for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
            for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                mass_bins = 40
                sub_dir = cent_dir.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
                sub_dir.cd()
                mass_array = np.array(df_applied.query("ct<@ctbin[1] and ct>@ctbin[0] and pt<@ptbin[1] and pt>@ptbin[0]")['m'].values, dtype=np.float64)
                counts, _ = np.histogram(mass_array, bins=mass_bins, range=[mass*0.97, mass*1.03])
                h1_minv = au.h1_invmass_ov(counts, cclass, ptbin, ctbin, hist_range = [mass*0.97, mass*1.03])

                for bkgmodel in BKG_MODELS:
                    # create dirs for models
                    fit_dir = sub_dir.mkdir(bkgmodel)
                    fit_dir.cd()
                    rawcounts, err_rawcounts, significance, err_significance, mu, mu_err, _, _ = au.fit_hist(h1_minv, cclass, ptbin, ctbin, mass, model=bkgmodel, mode=N_BODY, split=split, Eint = 17.3, peak_mode=PEAK_MODE)
                    rawcounts = sum(counts)
                    err_rawcounts = math.sqrt(rawcounts)
                    mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetBinContent(iBin , rawcounts / h1PreselEff.GetBinContent(iBin) / h1PreselEff.GetBinWidth(iBin) / hist_eff.GetBinContent(iBin) / (ptbin[0]+ptbin[1]) / 2 / n_ev)
                    mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetBinError(iBin, err_rawcounts / h1PreselEff.GetBinContent(iBin) / h1PreselEff.GetBinWidth(iBin) / hist_eff.GetBinContent(iBin) / (ptbin[0]+ptbin[1]) / 2 / n_ev)
                
                    pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetBinContent(iBin , rawcounts / h1PreselEff.GetBinContent(iBin) / h1PreselEff.GetBinWidth(iBin) / hist_eff.GetBinContent(iBin) / n_ev)
                    pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetBinError(iBin, err_rawcounts / h1PreselEff.GetBinContent(iBin) / h1PreselEff.GetBinWidth(iBin) / hist_eff.GetBinContent(iBin) / n_ev)
                
                iBin += 1

        mt_distr = TF1("mt_distr", "[0]*exp(-x/[1])",MT_BINS[3],MT_BINS[-1])
        cent_dir.cd()
        for bkgmodel in BKG_MODELS:
            cv.Clear()
            mt_distr.SetParameter(0,MULTIPLICITY/(math.e**(-mass/T)*(mass/T+1))/T)
            mt_distr.SetParLimits(0,0,3000)
            mt_distr.SetParameter(1,T)
            mt_distr.SetParLimits(1,T*0.5,T*1.5)
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Fit(mt_distr, "M0+", "",MT_BINS[0],MT_BINS[-1])
            
            min_value = mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetMinimum()*0.8
            max_value = mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetMaximum()*1.2
            
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetYaxis().SetRangeUser(min_value, max_value)
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Write()
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetMarkerStyle(20)
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetStats(0)
            cv.cd()
            cv.SetLogy()
            mt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Draw("ex0same")

            pinfo = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)
            err_decimal = pu.get_decimal(mt_distr.GetParError(1)*1000)
            string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  {cclass[0]}-{cclass[1]}%'
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
            pinfo.Draw("x0same")
            mt_distr.Draw("same")

            cv.SetName(f"mT_best_{bkgmodel}")
            cv.Write()
        ############################
        pt_distr = TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",PT_BINS[0],PT_BINS[-1])
        cent_dir.cd()
        for bkgmodel in BKG_MODELS:
            cv.Clear()
            pt_distr.SetParLimits(0,0,3000)
            pt_distr.SetParameter(1,T)
            pt_distr.SetParLimits(1,T*0.5,T*1.5)
            pt_distr.FixParameter(2,mass)
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Fit(pt_distr, "M0+", "",PT_BINS[0],PT_BINS[-1])
            
            min_value = pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetMinimum()*0.8
            max_value = pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetMaximum()*1.2
            
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].GetYaxis().SetRangeUser(min_value, max_value)
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Write()
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetMarkerStyle(20)
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].SetStats(0)
            cv.cd()
            cv.SetLogy()
            pt_spectra_counts_list[BKG_MODELS.index(bkgmodel)].Draw("ex0same")

            pinfo = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
            pinfo.SetBorderSize(0)
            pinfo.SetFillStyle(0)
            pinfo.SetTextAlign(30+3)
            pinfo.SetTextFont(42)
            err_decimal = pu.get_decimal(pt_distr.GetParError(1)*1000)
            string = 'Pb-Pb, #sqrt{#it{s}_{NN}} = '+f'{EINT} GeV,  {cclass[0]}-{cclass[1]}%'
            pinfo.AddText(string)
            string = 'T = {:.'+f'{err_decimal}'+'f} #pm {:.'+f'{err_decimal}'+'f} MeV '
            string = string.format(
                pt_distr.GetParameter(1)*1000, pt_distr.GetParError(1)*1000)
            pinfo.AddText(string)
            string = 'T_{gen}'+' = {:.2f} MeV'.format(T*1000)
            pinfo.AddText(string)
            if pt_distr.GetNDF() != 0:
                string = f'#chi^{{2}} / NDF = {(pt_distr.GetChisquare() / pt_distr.GetNDF()):.2f}'
            pinfo.AddText(string)
            pinfo.Draw("x0same")
            pt_distr.Draw("same")

            cv.SetName(f"pT_best_{bkgmodel}")
            cv.Write()

    results_file.cd()
print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

