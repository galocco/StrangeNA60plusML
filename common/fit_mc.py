#!/usr/bin/env python3

import argparse
import math
import os
import random
import time
from analysis_utils import crystal_ball, double_sided_crystal_ball
import uproot
import numpy as np
import yaml
from ROOT import (TF1, TH1D, TFile, gROOT, TDatabasePDG)
import ROOT

from sklearn.neighbors import KernelDensity

random.seed(1989)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to the YAML configuration file")
parser.add_argument('-v', '--voigt', help='Fit with the voigtian', action='store_true')
parser.add_argument('-s', '--sum', help='Fit with guassian + breit-wigner', action='store_true')
parser.add_argument('-d', '--exp', help='Fit with double-shoulded exponential', action='store_true')
#parser.add_argument('-c', '--crystal', help='Fit with the crystalball', action='store_true')
#parser.add_argument('-p', '--peak', help='Take signal from the gaussian fit', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def gauss_exp_tails(x, par):
    N = par[0]
    mu = par[1]
    sig = par[2]
    tau0 = par[3]
    tau1 = par[4]
    u = (x[0] - mu) / sig
    if (u < tau0):
        return N*ROOT.TMath.Exp(-tau0 * (u - 0.5 * tau0))
    elif (u <= tau1):
        return N*ROOT.TMath.Exp(-u * u * 0.5)
    else:
        return N*ROOT.TMath.Exp(-tau1 * (u - 0.5 * tau1))

def IntExp(x, tau):
  return -1. * ROOT.TMath.Exp(tau * (0.5 * tau - x)) / tau


def IntGaus(x):
  rootPiBy2 = ROOT.TMath.Sqrt(ROOT.TMath.PiOver2())
  return rootPiBy2 * (ROOT.TMath.Erf(x / ROOT.TMath.Sqrt2()))


FILE_PREFIX = params['FILE_PREFIX']
PDG_CODE = params['PDG']
PT_BINS = params['PT_BINS']
PRESELECTION = params['PRESELECTION']
MASS_WINDOW = params['MASS_WINDOW']
SIGMA = params['SIGMA']

def parameter_finder(hist):
    exp_l = TF1("exp_l","exp([0]*x-[1])", mass*(1-MASS_WINDOW), 0.485)
    exp_r = TF1("exp_l","exp([0]*x-[1])", 0.505, mass*(1+MASS_WINDOW))
    gauss = TF1("gauss","gaus", mass-SIGMA*2, mass+SIGMA*2)
    hist.Fit(exp_l,"MR+")
    hist.Fit(exp_r,"MR+")
    hist.Fit(gauss,"MR+")
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
EXP = args.exp

###############################################################################
start_time = time.time()                          # for performances evaluation

gROOT.SetBatch()

mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

if VOIGT:
    fit_tpl = TF1('fitTpl_mc', 'TMath::Voigt(x-[1],[2],[3])*[0]', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl.SetParameter(1,mass)
    fit_tpl.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl.SetParameter(2,SIGMA/2.)
    fit_tpl.SetParLimits(2,0,SIGMA)
    fit_tpl.SetParameter(3,SIGMA/2.)
    fit_tpl.SetParLimits(3,0,SIGMA)
    #fit_tpl.SetParameter(5,SIGMA/2.)
    #fit_tpl.SetParLimits(5,0,SIGMA)
elif SUM:
    fit_tpl = TF1('fitTpl_mc', 'gausn(0)+[3]/((x-[4])**2+[5]**2)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl.SetParameter(1,mass)
    fit_tpl.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl.SetParameter(2,SIGMA/2.)
    fit_tpl.SetParLimits(2,0,SIGMA)
    fit_tpl.SetParameter(5,SIGMA/2.)
    fit_tpl.SetParLimits(5,0,SIGMA)    
    fit_tpl.SetParameter(4,mass)
    fit_tpl.SetParLimits(4,mass-SIGMA/2.,mass+SIGMA/2.)
    #fit_tpl.SetParameter(5,SIGMA/2.)
    #fit_tpl.SetParLimits(5,0,SIGMA)
elif EXP:
    fit_tpl = TF1('fitTpl_mc', gauss_exp_tails, mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW),6)
    fit_tpl.SetParameter(1,mass)
    fit_tpl.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl.SetParameter(2,SIGMA)
    fit_tpl.SetParLimits(2,0,SIGMA*2)
else:
    fit_tpl = TF1('fitTpl_mc', 'gausn(0)+gausn(3)', mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    fit_tpl.SetParameter(1,mass)
    fit_tpl.SetParLimits(1,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl.SetParameter(4,mass)
    fit_tpl.SetParLimits(4,mass-SIGMA/2.,mass+SIGMA/2.)
    fit_tpl.SetParameter(2,SIGMA/2.)
    fit_tpl.SetParLimits(2,0,SIGMA)
    fit_tpl.SetParameter(5,SIGMA/2.)
    fit_tpl.SetParLimits(5,0,SIGMA)
    



resultsSysDir = os.environ['HYPERML_RESULTS']+"/"+FILE_PREFIX

file_name = resultsSysDir + '/' + FILE_PREFIX + '_mc_fit.root'
fits_file = TFile(file_name, 'recreate')

mc_file_name = os.path.expandvars(params['MC_PATH'])
columns = ['m','pt']
df_signal = uproot.open(mc_file_name)['ntcand'].arrays(library="pd").query(PRESELECTION)[columns]
nbins = 40
fits_file.cd()
fit_dir = fits_file.mkdir("fit")
ratio_dir = fits_file.mkdir("ratio")

model = KernelDensity(kernel="gaussian",bandwidth=0.0003)
cv = ROOT.TCanvas("cv","cv")
for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
    print('\n==================================================')
    print(' pT:', ptbin)
    if ptbin[0] > 0.3:
        continue
    df_selected = df_signal.query("pt<@ptbin[1] and pt>@ptbin[0]")
    #print(df_selected['m'].to_numpy().reshape(-1,1))
    x_train = df_selected['m'].to_numpy().reshape(-1,1)
    model.fit(x_train)
    def KDE(x, par):
        t = [[x[0]]]
        val = model.score(t)
        return np.exp(val)*par[0]

    fit_kde = TF1('fit_kde_mc', KDE, mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW), 1)
    fit_kde.SetLineColor(ROOT.kBlue)
    #fit_kde.SetParameter(1,0.0001)
    #fit_kde.SetParLimits(1,0.00005, 0.0005)
    counts, _ = np.histogram(df_selected['m'], bins=nbins, range=[mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW)])
    hist = TH1D(f'hist_mc_{ptbin[0]}_{ptbin[1]}', ';m (GeV/#it{c}^{2});Counts',nbins,mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    hist2 = TH1D(f'hist2_mc_{ptbin[0]}_{ptbin[1]}', ';m (GeV/#it{c}^{2});Counts',nbins,mass*(1-MASS_WINDOW), mass*(1+MASS_WINDOW))
    #cv.cd()
    #fit_kde.FixParameter(0, sum(counts))
    #hist2.GetYaxis().SetRangeUser(0,300)
    #hist2.Draw()
    #fit_kde.Draw("same")
    #cv.SaveAs(f'kde_{ptbin[0]}_{ptbin[1]}.pdf')
    #fit_dir.cd()
    #cv.Write()
    for index in range(0, nbins):
        hist.SetBinContent(index+1, counts[index])
        hist.SetBinError(index + 1, math.sqrt(counts[index]))
    #fit_tpl.SetParameters(1, 2.5, mass, hist.GetRMS()/2, hist.GetMaximum(), 1, 2.5);

    #parlist = parameter_finder(hist)
    #fit_tpl.SetParameter(0, parlist[0])
    if EXP:
        norm = sum(counts)/hist.GetBinWidth(1)/(ROOT.TMath.Sqrt(2*ROOT.TMath.Pi())*0.002)
        fit_tpl.SetParameter(0, norm)
        fit_tpl.SetParameter(3, (0.480-mass)/0.002)
        fit_tpl.SetParameter(4, (0.510-mass)/0.002)

    #fit_tpl.SetParLimits(0, norm/2., norm*3./2.)
    #fit_tpl.SetParameter(3, parlist[3])
    #fit_tpl.SetParameter(4, parlist[4])
    #fit_tpl.SetParameter(5, parlist[5])
    #fit_tpl.SetParameter(6, parlist[6])
    hist_ratio = hist.Clone(f'hist_ratio_{ptbin[0]}_{ptbin[1]}')
    hist.Fit(fit_tpl,"MR+")
    hist.Fit(fit_kde,"MR+")
    hist_ratio.GetYaxis().SetTitle('Counts/Fit')
    for index in range(0, nbins):
        value = fit_tpl.Eval(hist_ratio.GetBinCenter(index+1))
        if value == 0:
            value=1
        hist_ratio.SetBinContent(index+1, counts[index]/value)
        hist_ratio.SetBinError(index + 1, math.sqrt(counts[index])/value)
    print("hist counts: ",sum(counts))
    print("fit counts: ",(fit_tpl.GetParameter(0))/hist.GetBinWidth(1)*(ROOT.TMath.Sqrt(2*ROOT.TMath.Pi()))*0.002," +- ",(fit_tpl.GetParError(0))/hist.GetBinWidth(1)*(ROOT.TMath.Sqrt(2*ROOT.TMath.Pi())*0.002))
    print("kde counts: ",fit_kde.GetParameter(0)/hist.GetBinWidth(1)," +- ",fit_kde.GetParError(0)/hist.GetBinWidth(1))
    fit_dir.cd()
    hist.Write()
    ratio_dir.cd()
    hist_ratio.Write()
    del df_selected


fits_file.Close()

print('')
print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')