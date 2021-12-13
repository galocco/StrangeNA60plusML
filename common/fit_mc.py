#!/usr/bin/env python3

import argparse
from itertools import count
import math
import os
import random
import time

from analysis_utils import double_sided_crystal_ball, gauss_exp_tails
import analysis_utils as au
import uproot
import numpy as np
import yaml
from array import array
from ROOT import (TF1, TH1D, TFile, gROOT, TDatabasePDG)
import ROOT

random.seed(1989)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
parser.add_argument('-v', '--voigt', help='Fit with the voigtian', action='store_true')
parser.add_argument('-s', '--sum', help='Fit with guassian + breit-wigner', action='store_true')
parser.add_argument('-d', '--double', help='Fit with double-shoulded exponential', action='store_true')
parser.add_argument('-c', '--crystal', help='Fit with the extended crystalball', action='store_true')
parser.add_argument('-dg', '--dgauss', help='Fit with the two gaussians', action='store_true')
parser.add_argument('-g', '--gauss', help='Fit with the gaussian', action='store_true')
parser.add_argument('-l', '--lorentz', help='Fit with the lorentzian', action='store_true')
parser.add_argument('-k', '--kde', help='Fit with the KDE', action='store_true')
parser.add_argument('-f', '--fit', help='Compute the KDE', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


FILE_PREFIX = params['FILE_PREFIX']
PDG_CODE = params['PDG']
PT_BINS = params['PT_BINS']
PRESELECTION = params['PRESELECTION']
MASS_WINDOW = params['MASS_WINDOW']
SIGMA = params['SIGMA']
NBINS = params['NBINS']
def parameter_finder(hist):
    exp_l = TF1("exp_l","exp([0]*x-[1])", mass*(1-MASS_WINDOW), 0.485)
    exp_r = TF1("exp_l","exp([0]*x-[1])", 0.505, mass*(1+MASS_WINDOW))
    gauss = TF1("gauss","gaus", mass-SIGMA*2, mass+SIGMA*2)
    hist.Fit(exp_l,"MR0")
    hist.Fit(exp_r,"MR0")
    hist.Fit(gauss,"MR0")
    parlist = []
    parlist.append(gauss.GetParameter(0))
    parlist.append(gauss.GetParameter(1))
    parlist.append(gauss.GetParameter(2))
    parlist.append(exp_l.GetParameter(0))
    parlist.append(exp_l.GetParameter(1))
    parlist.append(exp_r.GetParameter(0))
    parlist.append(exp_r.GetParameter(1))

    return parlist

VOIGT = args.voigt
SUM = args.sum
DOUBLE = args.double
DGAUSS = args.dgauss
GAUSS = args.gauss
CRYSTAL = args.crystal
KDE = args.kde
LORENTZ = args.lorentz
FIT = args.fit

###############################################################################
start_time = time.time()                          # for performances evaluation

gROOT.SetBatch()
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
binning = array("d", PT_BINS)
resultsSysDir = os.environ['RESULTS']+"/"+FILE_PREFIX

mc_file_name = os.path.expandvars(params['MC_PATH'])
if FIT:
    cpp_code = """
    void fit_kde(TString filename, TString treename, double binning[], int nptbins, double mass, double range = 0.04, double rho = 2, double nbin = """+str(NBINS)+"""){
        // read the Tree generated by tree1w and fill two histograms
        // note that we use "new" to create the TFile and TTree objects,
        // to keep them alive after leaving this function.
        TFile *f = new TFile(treename.Data());
        TTree *t1 = (TTree*)f->Get("ntcand");
        TFile results(filename.Data(),"recreate");
        Float_t  pt,m;
        for(int bin=0; bin<nptbins; bin++)
        {
            std::cout<<"kde bin: "<<bin<<std::endl;
            t1->SetBranchAddress("m",&m);
            t1->SetBranchAddress("pt",&pt);
            // create two histograms
            TH1D * h1 = new TH1D(Form("mass_%0.2f_%0.2f",binning[bin],binning[bin+1]),";m (GeV/#it{c}^{2});Counts",nbin, mass*(1-range), mass*(1+range));
            //read all entries and fill the histograms
            Int_t nentries = (Int_t)t1->GetEntries();
            vector<double> mass_vector;
            for (Int_t i=0; i<nentries; i++){
                t1->GetEntry(i);
                if(pt > binning[bin+1] || pt < binning[bin])
                    continue;
                if(m > mass*(1+range) || m < mass*(1-range))
                    continue;
                mass_vector.push_back(m);
                h1->Fill(m);
            }
            // create TKDE class
            TKDE * kde = new TKDE(mass_vector.size(), &mass_vector[0], mass*(1-range), mass*(1+range), "Binning:RelaxedBinning", rho);
            kde->SetName(Form("kde_%0.2f_%0.2f",binning[bin],binning[bin+1]));

            // We do not close the file. We want to keep the generated
            // histograms we open a browser and the TreeViewer
            h1->GetYaxis()->SetRangeUser(0.1,h1->GetMaximum()*1.2);
            h1->Write();
            kde->Write();
            
        }
        results.Close();
    }
    """
    ROOT.gInterpreter.Declare(cpp_code)
    file_name = resultsSysDir + '/' + FILE_PREFIX + '_kde.root'
    file_string = ROOT.TString(file_name)
    tree_string = ROOT.TString(mc_file_name)
    ROOT.fit_kde(file_string, tree_string, binning, len(binning)-1, mass, MASS_WINDOW, 3, 40)

fit_dict = {}
name_dict = {}
color_dict = {}
marker_dict = {}
save_dict = {}
if VOIGT:
    fit_tpl_voigt = TF1('fitTpl_voigt', 'TMath::Voigt(x-[1],[2],[3])*[0]', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl_voigt.SetParameter(1,mass)
    fit_tpl_voigt.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl_voigt.SetParameter(2,SIGMA/2.)
    fit_tpl_voigt.SetParLimits(2,0,SIGMA)
    fit_tpl_voigt.SetParameter(3,SIGMA/2.)
    fit_tpl_voigt.SetParLimits(3,0,SIGMA)
    fit_tpl_voigt.SetLineColor(ROOT.kRed)
    #fit_tpl_voigt.SetParameter(5,SIGMA/2.)
    #fit_tpl_voigt.SetParLimits(5,0,SIGMA)
    fit_dict["VOIGT"] = fit_tpl_voigt
    color_dict['VOIGT'] = ROOT.kYellow
    name_dict["VOIGT"] = "voigtian"
    marker_dict["VOIGT"] = 45
    save_dict["VOIGT"] = "voigt"
if SUM:
    fit_tpl_sum = TF1('fitTpl_sum', 'gausn(0)+[3]/((x-[4])**2+[5]**2)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl_sum.SetParameter(1,mass)
    fit_tpl_sum.SetParLimits(1,mass-SIGMA,mass+SIGMA)
    fit_tpl_sum.SetParameter(2,SIGMA/2.)
    fit_tpl_sum.SetParLimits(2,0,SIGMA)
    fit_tpl_sum.SetParameter(5,SIGMA/2.)
    fit_tpl_sum.SetParLimits(5,0,SIGMA)    
    fit_tpl_sum.SetParameter(4,mass)
    fit_tpl_sum.SetParLimits(4,mass-SIGMA,mass+SIGMA)
    fit_tpl_sum.SetLineColor(ROOT.kGreen)
    #fit_tpl_sum.SetParameter(5,SIGMA/2.)
    #fit_tpl_sum.SetParLimits(5,0,SIGMA)
    color_dict['SUM'] = ROOT.kYellow
    fit_dict["SUM"] = fit_tpl_sum
    name_dict["SUM"] = "gaussian + lorentzian"
    marker_dict["SUM"] = 39
    save_dict["SUM"] = "sum"
if LORENTZ:
    fit_tpl_lorentz = TF1('fitTpl_lorentz', '[0]/((x-[1])**2+[2]**2/4.)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl_lorentz.SetParameter(2,SIGMA)
    fit_tpl_lorentz.SetParLimits(2,0,SIGMA*3)    
    fit_tpl_lorentz.SetParameter(1,mass)
    fit_tpl_lorentz.SetParLimits(1,mass-SIGMA,mass+SIGMA)
    fit_tpl_lorentz.SetLineColor(ROOT.kGreen)
    #fit_tpl_lorentz.SetParameter(5,SIGMA/2.)
    #fit_tpl_lorentz.SetParLimits(5,0,SIGMA)
    fit_dict["LORENTZ"] = fit_tpl_lorentz
    name_dict["LORENTZ"] = "lorentzian"
    marker_dict["LORENTZ"] = 34
    save_dict["LORENTZ"] = "lorentz"
if DOUBLE:
    fit_tpl_double = TF1('fitTpl_double', gauss_exp_tails, mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW), 5)
    fit_tpl_double.SetLineColor(ROOT.kBlack)
    fit_dict["DOUBLE"] = fit_tpl_double
    color_dict['DOUBLE'] = ROOT.kBlack
    name_dict["DOUBLE"] = "gaussian with exponential tails"
    marker_dict["DOUBLE"] = 21
    save_dict["DOUBLE"] = "exp-gauss"
if DGAUSS:
    fit_tpl_dgauss = TF1('fitTpl_dgauss', 'gausn(0)+gausn(3)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    #fit_tpl_dgauss = TF1('fitTpl_dgauss', '[6]*(TMath.Exp(-((x-[1])/[2])**2/2)*[0]+TMath.Exp(-((x-[3])/[4])**2/2)*[5])', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl_dgauss.SetParameter(1,mass)
    fit_tpl_dgauss.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl_dgauss.SetParameter(4,mass)
    fit_tpl_dgauss.SetParLimits(4,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl_dgauss.SetParameter(2,SIGMA/2.)
    fit_tpl_dgauss.SetParLimits(2,0,SIGMA*3)
    fit_tpl_dgauss.SetParameter(5,SIGMA/2.)
    fit_tpl_dgauss.SetParLimits(5,0,SIGMA*3)
    color_dict["DGAUSS"] = ROOT.kViolet
    fit_tpl_dgauss.SetLineColor(ROOT.kViolet)
    fit_dict["DGAUSS"] = fit_tpl_dgauss
    name_dict["DGAUSS"] = "two gaussians"
    save_dict["DGAUSS"] = "d-gauss"
    marker_dict["DGAUSS"] = 22
if GAUSS:
    fit_tpl_gauss = TF1('fitTpl_gauss', 'gausn(0)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl_gauss.SetParameter(1,mass)
    fit_tpl_gauss.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl_gauss.SetParameter(2,SIGMA/2.)
    fit_tpl_gauss.SetParLimits(2,0,SIGMA*3)
    color_dict['GAUSS'] = ROOT.kPink
    fit_tpl_gauss.SetLineColor(ROOT.kPink)
    fit_dict["GAUSS"] = fit_tpl_gauss
    name_dict["GAUSS"] = "gaussian"
    save_dict["GAUSS"] = "gauss"
    marker_dict["GAUSS"] = 23
if CRYSTAL:
    fit_tpl_crystal = TF1('fitTpl_crystal', double_sided_crystal_ball, mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW), 7)
    fit_tpl_crystal.SetLineColor(ROOT.kOrange)
    fit_dict["CRYSTAL"] = fit_tpl_crystal
    color_dict['CRYSTAL'] = ROOT.kOrange
    name_dict["CRYSTAL"] = "double sided crystal ball"
    save_dict["CRYSTAL"] = "d-crystal"
    marker_dict["CRYSTAL"] = 33
if KDE:
    color_dict['KDE'] = ROOT.kBlue
    save_dict['KDE'] = "kde"
    name_dict["KDE"] = "kernel density estimation"
    marker_dict["KDE"] = 20
    file_name = resultsSysDir + '/' + FILE_PREFIX + '_kde.root'
    kde_file = TFile(file_name)

file_name = resultsSysDir + '/' + FILE_PREFIX + '_mc_fit.root'
fits_file = TFile(file_name, 'recreate')

columns = ['m','pt']
df_signal = uproot.open(mc_file_name)['ntcand'].arrays(library="pd").query(PRESELECTION)[columns]
fits_file.cd()
plot_dir = fits_file.mkdir("plot")
fit_dir = fits_file.mkdir("fit")
#ratio_dir = fits_file.mkdir("ratio")
ROOT.gStyle.SetOptStat(0)
cv = ROOT.TCanvas("cv","cv")
res_dict = {}
for key in color_dict:
    hist_res = ROOT.TH1D("hist_res_"+key,";#it{p}_{T} (GeV/#it{c}); #frac{hist counts - fit result}{hist counts}",len(binning)-1,binning)
    hist_res.SetMarkerColor(color_dict[key])
    hist_res.SetLineColor(color_dict[key])
    hist_res.SetMarkerStyle(marker_dict[key])
    res_dict[key] = hist_res
counter = 0
for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
    print('\n==================================================')
    print(' pT:', ptbin)
    df_selected = df_signal.query("pt<@ptbin[1] and pt>@ptbin[0]")
    x_train = df_selected['m'].to_numpy().reshape(-1,1)

    counts, _ = np.histogram(df_selected['m'], bins=NBINS, range=[mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW)])
    hist = TH1D(f'hist_mc_{ptbin[0]}_{ptbin[1]}', ';m (GeV/#it{c}^{2});Counts',NBINS,mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    hist.SetMarkerStyle(20)
    fit_dir.cd()
    for index in range(0, NBINS):
        hist.SetBinContent(index+1, counts[index])
        hist.SetBinError(index + 1, math.sqrt(counts[index]))

    if CRYSTAL:
        fit_dict['CRYSTAL'].SetParameters(1, 2.5, mass, hist.GetRMS()/2, hist.Integral(1, hist.GetNBINSX()), 1, 2.5)

        fit_dict['CRYSTAL'].SetParLimits(0, 0.5, 2.5)
        fit_dict['CRYSTAL'].SetParLimits(1, 0.5, 4)
        fit_dict['CRYSTAL'].SetParLimits(2, mass-SIGMA/2., mass+SIGMA/2.)
        fit_dict['CRYSTAL'].SetParLimits(5, 0.5, 2)
        fit_dict['CRYSTAL'].SetParLimits(6, 1, 4)
    if DOUBLE:
        norm = sum(counts)/hist.GetBinWidth(1)

        fit_tpl_double.SetParameter(1,hist.GetMean())
        fit_tpl_double.SetParLimits(1,hist.GetMean()-0.003,hist.GetMean()+0.003)
        fit_dict['DOUBLE'].SetParameter(0, norm/ROOT.TMath.Sqrt(2*ROOT.TMath.Pi())*hist.GetRMS())
        fit_dict['DOUBLE'].SetParameter(2, hist.GetRMS()/3*2)
        fit_dict['DOUBLE'].SetParLimits(2,hist.GetRMS()/2,hist.GetRMS()*3/2)
        fit_dict['DOUBLE'].SetParameter(3, (0.480-mass)/hist.GetRMS())
        fit_dict['DOUBLE'].SetParameter(4, (0.505-mass)/hist.GetRMS())
        fit_dict['DOUBLE'].SetParLimits(3, (0.475-mass)/hist.GetRMS(), 0)
        fit_dict['DOUBLE'].SetParLimits(4, 0, (0.510-mass)/hist.GetRMS())
    if KDE:
        func = kde_file.Get(f'kde_{ptbin[0]:.2f}_{ptbin[1]:.2f}')
        def KDE(x, par):
            return func.GetValue(x[0])*par[0]

        fit_dict['KDE'] = TF1('fitTpl_kde', KDE, mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW), 1)
        fit_dict['KDE'].SetLineColor(ROOT.kBlue)
        func.Write()
    legend = ROOT.TLegend(0.12,0.63,0.38,0.88)
    legend.SetLineWidth(0)
    cv = ROOT.TCanvas(f'cv_{ptbin[0]}_{ptbin[1]}',"cv")

    hist_clone = hist.Clone(hist.GetName()+"_clone")
    #hist.GetYaxis().SetRangeUser(0.1,hist.GetMaximum()*5)
    hist_log = hist.Clone(hist.GetName()+"_log")
    hist.GetYaxis().SetRangeUser(0.1,hist.GetMaximum()*1.2)
    cv.cd()
    hist.Draw()
    cv_log = ROOT.TCanvas(f'cv_{ptbin[0]}_{ptbin[1]}_log',"cv")
    cv_log.cd()
    hist_log.GetYaxis().SetRangeUser(0.1,hist_log.GetMaximum()*5)
    hist_log.Draw()
    for key in fit_dict:
        print("fit with: ",key)
        hist.Fit(fit_dict[key],"MR+","", mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
        fit_dict[key].SetName(f'{save_dict[key]}_{ptbin[0]:.2f}_{ptbin[1]:.2f}')
        if key != "KDE":
            fit_dict[key].Write()
        cv.cd()
        fit_dict[key].Draw("same")
        cv_log.cd()
        fit_dict[key].Draw("same")
        legend.AddEntry(fit_dict[key].GetName(),name_dict[key],"l")
        cv_single = ROOT.TCanvas(f'cv_{ptbin[0]}_{ptbin[1]}_'+key,f'cv_{ptbin[0]}_{ptbin[1]}_'+key)
        cv_single.Clear()
        cv_single.cd()
        hist_tmp = hist_clone.Clone(hist.GetName()+"_"+key)
        hist_tmp.SetTitle(name_dict[key])

        hist_tmp.Draw()
        fit_dict[key].Draw("same")
        #cv_single.SaveAs("../Results/"+FILE_PREFIX+"/"+hist_tmp.GetName()+".png")
        #cv_single.SaveAs("../Results/"+FILE_PREFIX+"/"+hist_tmp.GetName()+".pdf")
        cv_single_2 = ROOT.TCanvas(f'cv_{ptbin[0]}_{ptbin[1]}_2_'+key,f'cv_{ptbin[0]}_{ptbin[1]}_2_'+key)
        cv_single_2.cd()
        hist_tmp.GetYaxis().SetRangeUser(0.1,hist.GetMaximum()*5)
        hist_tmp.Draw()
        fit_dict[key].Draw("same")
        cv_single_2.SetLogy()

        if key=="VOIGT":
            counts_fit = fit_dict[key].GetParameter(0)
            err_counts_fit = fit_dict[key].GetParError(0)
        if key=="DGAUSS":
            counts_fit = (fit_dict[key].GetParameter(0)+fit_dict[key].GetParameter(3))
            err_counts_fit = ROOT.TMath.Sqrt(fit_dict[key].GetParError(0)**2+fit_dict[key].GetParError(3)**2)
        elif key=="GAUSS":
            counts_fit = fit_dict[key].GetParameter(0)
            err_counts_fit = fit_dict[key].GetParError(0)
        elif key=="DOUBLE":
            counts_fit = fit_dict[key].GetParameter(0)
            err_counts_fit = fit_dict[key].GetParError(0)
            fit_dict[key].SetParameter(0, 1)
            fit_dict[key].SetParError(0, 0)
            integral = au.IntExp(fit_dict[key].GetParameter(3))+au.IntExp(fit_dict[key].GetParameter(4))+au.IntGauss(fit_dict[key].GetParameter(3))+au.IntGauss(fit_dict[key].GetParameter(4))
            integral = fit_dict[key].Integral(mass*(1-MASS_WINDOW)-1, mass*(1+MASS_WINDOW)+1)
            counts_fit *= integral
            err_counts_fit *= integral#ROOT.TMath.Sqrt(err_counts_fit**2+fit_dict[key].IntegralError(mass*(1-MASS_WINDOW)-1, mass*(1+MASS_WINDOW)+1)**2)
        elif key=="CRYSTAL":
            counts_fit = fit_dict[key].GetParameter(4)
            err_counts_fit = fit_dict[key].GetParError(4)
            fit_dict[key].SetParameter(4, 1)
            fit_dict[key].SetParError(4, 0)
            integral = fit_dict[key].Integral(mass*(1-MASS_WINDOW)-1, mass*(1+MASS_WINDOW)+1)
            counts_fit *= integral
            err_counts_fit *= integral#ROOT.TMath.Sqrt(err_counts_fit**2+fit_dict[key].IntegralError(mass*(1-MASS_WINDOW)-1, mass*(1+MASS_WINDOW)+1)**2)
        elif key=="KDE":
            counts_fit = fit_dict[key].GetParameter(0)
            err_counts_fit = fit_dict[key].GetParError(0)
        
        res_dict[key].SetBinContent(counter+1, (sum(counts)-counts_fit/hist.GetBinWidth(1))/sum(counts))
        res_dict[key].SetBinError(counter+1, err_counts_fit/sum(counts)/hist.GetBinWidth(1))
    cv.cd()
    legend.Draw()
    plot_dir.cd()
    cv.Write()
    cv.SaveAs("../Results/"+FILE_PREFIX+"/"+hist.GetName()+"_all.pdf")
    cv_log.SetLogy()
    cv_log.SaveAs("../Results/"+FILE_PREFIX+"/"+hist.GetName()+"_all_logy.pdf")
    hist.Write()
    counter += 1

    del df_selected
cv_res = ROOT.TCanvas("cv_res","cv_res")
cv_res.SetLeftMargin(0.15)
legend_res = ROOT.TLegend(0.2, 0.5, 0.5, 0.85)
legend_res.SetLineWidth(0)
for key in fit_dict:
    res_dict[key].GetYaxis().SetRangeUser(-0.06, 0.12)
    res_dict[key].GetXaxis().SetRangeUser(0, 2.75)
    legend_res.AddEntry(res_dict[key],name_dict[key],"ep")
    res_dict[key].Draw("same")
    res_dict[key].Write()
legend_res.Draw()
cv_res.Write()
cv_res.SaveAs("../Results/"+FILE_PREFIX+"/residual_counts.pdf")

fits_file.Close()

print('')
print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')