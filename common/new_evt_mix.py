#!/usr/bin/env python3
import argparse

from numpy import array
import uproot
import ROOT
import os
import yaml
import analysis_utils as au
from array import array
import plot_utils as pu

parser = argparse.ArgumentParser()
parser.add_argument('-dg', '--dgauss', help='Fit with two gaussians', action='store_true')
parser.add_argument("config", help="Path to the YAML configuration file")
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

################################################################################
INPUT_FILE = params["INPUT_FILE"]
FILE_PREFIX = params["FILE_PREFIX"]
dgauss = args.dgauss  
SIGNAL_FILE = params["SIGNAL_FILE"]
results_path = os.environ['RESULTS'] + '/'+FILE_PREFIX
T = params["T"]
MULTIPLICITY = params["MULTIPLICITY"]
PT_BINS = params['PT_BINS']
PDG_CODE = params["PDG"]
MASS_MIN = params["MASS_MIN"]
MASS_MAX = params["MASS_MAX"]
BRATIO = params["BRATIO"]
NEVENTS_DATA = params["NEVENTS_DATA"]
EINT = pu.get_sNN(params["EINT"])

###############################################################################

full_run = 0.15*3*10**10
input = ROOT.TFile(INPUT_FILE)
output = ROOT.TFile(results_path+"/event_mixing_"+FILE_PREFIX+".root", "recreate")

mass = ROOT.TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

#################################################
# loop over the pT bins to extract the signal
#################################################




fit_tpl = ROOT.TF1("fit_tpl", "pol1(0)+TMath::Voigt(x-[3],[4],[5])*[2]", MASS_MIN, MASS_MAX)
bkg_tpl = ROOT.TF1("bkg_tpl", "[0]+[1]*x", MASS_MIN, MASS_MAX)


lifetime = ROOT.TDatabasePDG.Instance().GetParticle(PDG_CODE).Lifetime()

fit_tpl.FixParameter(5, ROOT.TMath.Hbar()*6.242*ROOT.TMath.Power(10,9)/lifetime)
fit_tpl.SetParameter(2, 1.65244e+03)
fit_tpl.FixParameter(3, mass)
fit_tpl.SetParameter(4, 1.43971e-03)

fit_tpl.SetNpx(10000)
bkg_tpl.SetNpx(10000)


bkg_tpl.SetLineWidth(2)
bkg_tpl.SetLineStyle(2)
bkg_tpl.SetLineColor(ROOT.kRed)


fit_tpl.SetLineWidth(2)
fit_tpl.SetLineColor(ROOT.kBlue)

###############################################

mix_sparse = input.Get("hsp_mix")
data_sparse = input.Get("hsp_data")

pt_binx = mix_sparse.GetAxis(1).GetNbins()
bin_width = 0.1 #GeV/c
output.cd()

hist_data_int = data_sparse.Projection(0)
hist_data_int.SetName(f"hist_data_pT_int")

def get_counts(hist, min_value):
    bin_min = hist.FindBin(min_value)
    counts = 0
    for i in range(bin_min, hist.GetNbinsX()+1):
        counts += hist.GetBinContent(i)
    return counts

def set_cosmetics(hist):
    bin_width = hist.GetBinWidth(1)*1000
    hist.GetYaxis().SetTitle(f"Counts/{bin_width:.2f}"+" MeV/#it{c}")
    hist.GetXaxis().SetTitle("m (GeV/#it{c}^{2})")
    hist.SetTitle("")
    hist.SetMarkerStyle(20)
    hist.GetYaxis().SetLabelSize(0.04)
    hist.GetXaxis().SetLabelSize(0.04)

    hist.GetYaxis().SetTitleSize(0.05)
    hist.GetXaxis().SetTitleSize(0.05)

for bin in range(1, hist_data_int.GetNbinsX()+1):
    hist_data_int.SetBinContent(bin,0)
for bin in range(0,pt_binx):
    #if (bin+1)*bin_width > 3:
    #    break

    mix_sparse.GetAxis(4).SetRange(2, 1)
    data_sparse.GetAxis(4).SetRange(2, 1)

    mix_sparse.GetAxis(4).SetRange(1, 1)
    data_sparse.GetAxis(4).SetRange(1, 1)

    mix_sparse.GetAxis(3).SetRange(12, 30-12)
    data_sparse.GetAxis(3).SetRange(12, 30-12)

    mix_sparse.GetAxis(1).SetRange(bin+1, bin+1)
    data_sparse.GetAxis(1).SetRange(bin+1, bin+1)

    hist_mix = mix_sparse.Projection(0)
    hist_mix.SetName(f"hist_mix_pT_{bin*bin_width:.1f}_{(bin+1)*bin_width:.1f}")

    hist_data = data_sparse.Projection(0)
    hist_data.SetName(f"hist_data_pT_{bin*bin_width:.1f}_{(bin+1)*bin_width:.1f}")

    if hist_mix.Integral() == 0:
        break
    counts_mix = get_counts(hist_mix, 1.04)
    counts_data = get_counts(hist_data, 1.04)
    
    hist_mix.Scale(hist_data.Integral()/hist_mix.Integral())
    #hist_mix.Scale(counts_data/counts_mix)
    hist_data.Add(hist_mix, -1)
    set_cosmetics(hist_data)
    hist_data.Write()
    hist_data_int.Add(hist_data)

set_cosmetics(hist_data_int)
cv = ROOT.TCanvas("cv","cv")
cv.SetLeftMargin(0.15)
hist_data_int.GetXaxis().SetRangeUser(mass-0.03,mass+0.03)
hist_data_int.Fit(fit_tpl,"MR+","",mass-0.03,mass+0.03)
bkg_tpl.SetParameter(0, fit_tpl.GetParameter(0))
bkg_tpl.SetParameter(1, fit_tpl.GetParameter(1))
ROOT.gStyle.SetOptStat(0)
hist_data_int.Write()
hist_data_int.Draw()
bkg_tpl.Draw("same")
cv.SaveAs(f"{results_path}/PHI_pT_integrated_mass.png")
cv.SaveAs(f"{results_path}/PHI_pT_integrated_mass.pdf")

hist_data_int_scal = hist_data_int.Clone("hist_data_int_scal")
hist_data_int_scal.GetListOfFunctions().Clear()

for bin in range(1, hist_data_int.GetNbinsX()+1):
    hist_data_int_scal.SetBinContent(bin, full_run/NEVENTS_DATA*fit_tpl.Eval(hist_data_int.GetBinCenter(bin)))
    #hist_data_int_scal.SetBinContent(bin, ROOT.TMath.Sqrt(full_run/NEVENTS_DATA*fit_tpl.Eval(hist_data_int.GetBinCenter(bin))))
hist_data_int_scal.Draw()
hist_data_int_scal.GetYaxis().SetRangeUser(1.5*hist_data_int_scal.GetMinimum(),1.5*hist_data_int_scal.GetMaximum())
for i in range(0,3):
    fit_tpl.SetParameter(i, fit_tpl.GetParameter(i)*full_run/NEVENTS_DATA)
for i in range(0,2):
    bkg_tpl.SetParameter(i, bkg_tpl.GetParameter(i)*full_run/NEVENTS_DATA)

fit_tpl.Draw("same")
bkg_tpl.Draw("same")
# print fit info on the canvas
pinfo2 = ROOT.TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
pinfo2.SetBorderSize(0)
pinfo2.SetFillStyle(0)
pinfo2.SetTextAlign(30+3)
pinfo2.SetTextFont(42)
pinfo2.SetTextSize(0.045)


string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{EINT} GeV, centrality 0-5%'
pinfo2.AddText(string)
string = '6e+11 ions on target, 15% target int. length'
pinfo2.AddText(string)

string = '#phi#rightarrow K^{+} + K^{-}'
pinfo2.AddText(string)

signal = fit_tpl.GetParameter(2)/hist_data_int_scal.GetBinWidth(1)

string = f'S {signal:.0f}' #pm {errsignal:.0f}'
pinfo2.AddText(string)

pinfo2.Draw()
cv.SaveAs(f"{results_path}/PHI_pT_integrated_mass_scaled.png")
cv.SaveAs(f"{results_path}/PHI_pT_integrated_mass_scaled.pdf")

output.Close()


    