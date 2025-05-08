#!/usr/bin/env python3
import argparse

from numpy import array
from plot_utils import get_decimal
import uproot
import ROOT
import os
import yaml
import analysis_utils as au
from array import array

#################################################
# TODO: 
#################################################

def get_entries(th1):
    counts = 0
    for i in range(1,th1.GetNbinsX()+1):
        counts+=th1.GetBinContent(i)
    return counts

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
inputFile = params["INPUT_FILE"]
suffix = params["FILE_PREFIX"]
dgauss = args.dgauss  
signal_file = params["SIGNAL_FILE"]
results_path = os.environ['RESULTS'] + '/'+suffix
T = params["T"]
MULTIPLICITY = params["MULTIPLICITY"]
PT_BINS = params['PT_BINS']
PDG_CODE = params["PDG"]
MASS_MIN = params["MASS_MIN"]
MASS_MAX = params["MASS_MAX"]
BRATIO = params["BRATIO"]
REBIN = params["REBIN"]
###############################################################################

full_run = 0.15*3*10**10
input = ROOT.TFile(inputFile)
hist_data = input.Get("data")
hist_data_sig = input.Get("data_sig")
hist_data_bkg = input.Get("data_bkg")
pt_bin_width = hist_data.GetXaxis().GetBinWidth(1)
hist_mix = input.Get("mix")
hist_ev = input.Get("ev")
nev = hist_ev.GetBinContent(1)
print("nev: ",nev)

output = ROOT.TFile(results_path+"/event_mixing_"+suffix+".root", "recreate")
counts_data = 0
counts_mix = 0
normalization = 0

bin_min = 0
bin_max = 0
dir_mix = output.mkdir("event mixing")
dir_sub = output.mkdir("subtraction")
dir_sub_no_sig = output.mkdir("subtraction no signal")
dir_sig = output.mkdir("signal")
dir_scale = output.mkdir("scaled")
dir_data = output.mkdir("data")
dir_ratio = output.mkdir("ratio")
nbins = int((PT_BINS[-1]-PT_BINS[0])/pt_bin_width/REBIN)
#print(PT_BINS[-1])
hist_raw = ROOT.TH1D("hist_raw", ";#it{p}_{T} (GeV/#it{c});dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]",nbins, PT_BINS[0], PT_BINS[-1])
hist_pt = ROOT.TH1D("hist_pt", ";#it{p}_{T} (GeV/#it{c});1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]", nbins, PT_BINS[0], PT_BINS[-1])
hist_raw_mc = ROOT.TH1D("hist_raw_mc", ";#it{p}_{T} (GeV/#it{c});dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]", nbins, PT_BINS[0], PT_BINS[-1])
hist_pt_mc = ROOT.TH1D("hist_pt_mc", ";#it{p}_{T} (GeV/#it{c});1/N_{ev}dN/d#it{p}_{T} [(GeV/#it{c})^{-1}]", nbins, PT_BINS[0], PT_BINS[-1])
hist_mass = ROOT.TH1D("hist_mass", ";#it{p}_{T} (GeV/#it{c}); mass (GeV/#it{c}^{2})", nbins, PT_BINS[0], PT_BINS[-1])

#effFile = ROOT.TFile(effFile)

#hist_eff = effFile.Get("hPtEff")
mass = ROOT.TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()

pt_distr = ROOT.TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])", 0, 2.5)

pt_distr.SetParameter(1, T)
pt_distr.SetParLimits(1, T*0.8, T*1.2)
pt_distr.FixParameter(2, mass)

pt_distr_gen = ROOT.TF1("pt_distr_gen", "x*exp(-TMath::Sqrt(x**2+[1]**2)/[0])", 0, 20)

pt_distr_gen.FixParameter(0, T)
pt_distr_gen.FixParameter(1, mass)

#################################################
# loop over the pT bins to extract the signal
#################################################


for bin in range(1, hist_data.GetNbinsX()*REBIN):
    
    fit_tpl = ROOT.TF1("fit_tpl", "pol1(0)+TMath::Voigt(x-[3],[4],[5])*[2]", MASS_MIN, MASS_MAX)
    sig_tpl = ROOT.TF1("sig_tpl", "TMath::Voigt(x-[1],[2],[3])*[0]", MASS_MIN, MASS_MAX)
    bkg_tpl = ROOT.TF1("bkg_tpl", "[0]+[1]*x", MASS_MIN, MASS_MAX)
    if dgauss:
        fit_tpl = ROOT.TF1("fit_tpl","gausn(2)+gausn(5)+pol1(0)", MASS_MIN, MASS_MAX)

    lifetime = ROOT.TDatabasePDG.Instance().GetParticle(PDG_CODE).Lifetime()

    fit_tpl.FixParameter(5, ROOT.TMath.Hbar()*6.242*ROOT.TMath.Power(10,9)/lifetime)
    sig_tpl.FixParameter(3, ROOT.TMath.Hbar()*6.242*ROOT.TMath.Power(10,9)/lifetime)

    sig_tpl.SetParameter(1, mass)
    sig_tpl.SetParLimits(1, mass-0.0003,mass+0.0003)

    fit_tpl.SetNpx(600)
    sig_tpl.SetNpx(600)
    bkg_tpl.SetNpx(600)

    fit_bkg = ROOT.TF1("fit_bkg","pol1(0)",MASS_MIN,MASS_MAX)
    fit_bkg.SetLineColor(ROOT.kBlue)
    fit_bkg.SetLineStyle(2)
    
    ptbin_lw = hist_data.GetXaxis().GetBinLowEdge((bin-1)*REBIN+1)
    ptbin_up = hist_data.GetXaxis().GetBinUpEdge(bin*REBIN)
    proj_data = hist_data.ProjectionY(f'{hist_data.GetName()}_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}', (bin-1)*REBIN+1, bin*REBIN)
    proj_data.SetMarkerStyle(20)
    proj_data_sig = hist_data_sig.ProjectionY(f'{hist_data_sig.GetName()}_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}', (bin-1)*REBIN+1, bin*REBIN)
    proj_data_sig.SetMarkerStyle(20)
    sig_counts = proj_data_sig.GetEntries()
    #sig_tpl.SetParameter(0, sig_counts*proj_data_sig.GetBinWidth(1))
    #sig_tpl.SetParLimits(0, (sig_counts-10)*proj_data_sig.GetBinWidth(1), (sig_counts+10)*proj_data_sig.GetBinWidth(1))
    proj_data_sig.Fit("sig_tpl","MIR0","", MASS_MIN, MASS_MAX)
    proj_data_sig.Fit("sig_tpl","MIR","", MASS_MIN, MASS_MAX)
    dir_sig.cd()
    proj_data_sig.Write()
    # save the exact generated signal particles for further checks
    if bin<=hist_raw_mc.GetNbinsX():
        entries = get_entries(proj_data_sig)
        hist_raw_mc.SetBinContent(bin, entries)
        hist_raw_mc.SetBinError(bin, ROOT.TMath.Sqrt(entries))
    

    proj_data_bkg = hist_data_bkg.ProjectionY(f'{hist_data_bkg.GetName()}_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}', (bin-1)*REBIN+1, bin*REBIN)
    proj_data_bkg.SetMarkerStyle(20)

    x0 = proj_data_bkg.GetBinCenter(2)
    x1 = proj_data_bkg.GetBinCenter(proj_data_bkg.GetNbinsX()-1)
    y0 = proj_data_bkg.GetBinContent(2)
    y1 = proj_data_bkg.GetBinContent(proj_data_bkg.GetNbinsX()-1)

    c = (y0*x1-x0*y1)/(x1-x0)
    m = (y1-y0)/(x1-x0)
    print("y = mx + c")
    print("m = ",m)
    print("c = ",c)
    bkg_tpl.SetParameter(0, -1000)
    bkg_tpl.SetParameter(1, 1000)
    bkg_tpl.SetParLimits(0, -5e3, 5e3)
    bkg_tpl.SetParLimits(1, -5e3, 5e3)

    proj_data_bkg.Fit("bkg_tpl","R","", MASS_MIN, MASS_MAX)
    proj_data_bkg.Fit("bkg_tpl","MR","", MASS_MIN, MASS_MAX)
    
    proj_mix = hist_mix.ProjectionY(f'{hist_mix.GetName()}_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}', (bin-1)*REBIN+1, bin*REBIN)
    proj_mix.SetMarkerStyle(20)

    counts_data = 0
    counts_mix = 0

    bin_min = proj_data.GetXaxis().FindBin(1.06)
    bin_max = proj_data.GetXaxis().FindBin(1.10)
    # compute the normalization of the event-mixing counts to the data counts
    # the mass range considered for the normalization is arbitrary -> possible source of systematic uncertainties
    for bin_pro in range(bin_min, bin_max+1):
        counts_data += proj_data.GetBinContent(bin_pro)
        counts_mix += proj_mix.GetBinContent(bin_pro)
    if counts_mix==0:
        normalization = 0
    else:
        normalization = counts_data/counts_mix
    

    #################################################
    # compute the ration between the data counts and
    # the event-mixing counts
    #################################################
    proj_sub = proj_data.Clone()
    proj_sub_bkg = proj_data_bkg.Clone()
    
    #################################################
    # compute the difference between the data and 
    # the normalized event-mixing
    #################################################

    proj_sub.SetName(f'sub_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}')
    proj_sub_bkg.SetName(f'sub_bkg_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}')
    for set in range(1, proj_sub.GetNbinsX()+1):
        delta = proj_data.GetBinContent(set)-normalization*normalization*proj_mix.GetBinContent(set)
        delta_bkg = proj_data.GetBinContent(set)-normalization*normalization*proj_mix.GetBinContent(set)
        proj_sub.SetBinError(set, ROOT.TMath.Sqrt(delta))
        proj_sub_bkg.SetBinError(set, ROOT.TMath.Sqrt(delta_bkg))
        proj_mix.SetBinContent(set, int(normalization*proj_mix.GetBinContent(set)))
    
    range_min = proj_mix.GetXaxis().GetBinLowEdge(8)
    range_max = proj_mix.GetXaxis().GetBinUpEdge(proj_mix.GetNbinsX()-4)
    proj_sub.Add(proj_mix, -1)
    proj_sub_bkg.Add(proj_mix, -1)
    dir_sub_no_sig.cd()
    proj_sub_bkg.GetXaxis().SetRangeUser(range_min, range_max)
    proj_sub_bkg.Write()
    # add counts to have an integrated invariant mass plot
    if bin==1:
        proj_sub_tot = proj_sub.Clone()
        proj_sub_tot.SetMarkerStyle(20)
        proj_sub_tot.SetName("sub_tot")
    else:
        proj_sub_tot.Add(proj_sub)
    
    #################################################
    # compute the ration between the data counts and
    # the event-mixing counts
    #################################################

    proj_ratio = proj_data.Clone()
    proj_ratio.SetName(f'ratio_pt_{ptbin_lw:.1f}_{ptbin_lw:.1f}')
    proj_ratio.Divide(proj_mix)
    dir_ratio.cd()
    proj_ratio.GetXaxis().SetRangeUser(range_min, range_max)
    proj_ratio.Write()

    range_min_y = 1.3*proj_sub.GetMinimum()  if proj_sub.GetMinimum()<0 else 0.8*proj_sub.GetMinimum()
    range_max_y = 1.3*proj_sub.GetMaximum()

    # save the event-mixing histogram
    dir_mix.cd()
    proj_mix.GetXaxis().SetRangeUser(range_min, range_max)
    proj_mix.Write()

    dir_data.cd()
    proj_data.GetXaxis().SetRangeUser(range_min, range_max)
    proj_data.GetYaxis().SetTitle(f'Counts/{1000*proj_data.GetBinWidth(1):.2f}'+' MeV/#it{c}^{2}')
    proj_data.GetXaxis().SetTitle("m (GeV/#it{c}^{2})")
    proj_data.Write()

    dir_sub.cd()
    proj_sub.GetXaxis().SetRangeUser(range_min, range_max)
    print("prefit of the background")
    # set the parameters for the inv-mass fit function
    if dgauss:
        fit_tpl.SetParameter(3, mass)
        fit_tpl.SetParLimits(3, mass-0.0003,mass+0.0003)
        fit_tpl.SetParameter(6, mass)
        fit_tpl.SetParLimits(6, mass-0.0003,mass+0.0003)
        fit_tpl.SetParameter(4, 0.002)
        fit_tpl.SetParameter(7, 0.002)
        fit_tpl.SetParLimits(4, 0.001,0.004)
        fit_tpl.SetParLimits(7, 0.001,0.004)
    else:
        #fit_tpl.SetParameter(0, bkg_tpl.GetParameter(0))
        #fit_tpl.SetParameter(1, bkg_tpl.GetParameter(1))
        fit_tpl.SetParameter(2, sig_tpl.GetParameter(0))
        fit_tpl.SetParLimits(2, sig_tpl.GetParameter(0)-10, sig_tpl.GetParameter(0)+10)
        fit_tpl.SetParameter(3, sig_tpl.GetParameter(1))
        fit_tpl.SetParLimits(3, sig_tpl.GetParameter(1)-0.001,sig_tpl.GetParameter(1)+0.001)
        fit_tpl.SetParameter(4, sig_tpl.GetParameter(2))
        fit_tpl.SetParLimits(4, sig_tpl.GetParameter(2)*0.98,sig_tpl.GetParameter(2)*1.02)

    print("sig+bkg fit")
    proj_sub.Fit("fit_tpl","MIR+","", MASS_MIN, MASS_MAX)
    proj_sub.GetYaxis().SetRangeUser(range_min_y, range_max_y)

    proj_sub.SetMarkerStyle(20)
    proj_sub.Write()
    # save the value of the invariant mass
    if bin<=hist_mass.GetNbinsX():
        hist_mass.SetBinContent(bin, fit_tpl.GetParameter(3))
        hist_mass.SetBinError(bin, fit_tpl.GetParError(3))

    # save the value of the number of signal particles
    if bin<=hist_raw.GetNbinsX():
        if not dgauss:
            hist_raw.SetBinContent(bin, fit_tpl.GetParameter(2)/proj_sub.GetBinWidth(1))
            hist_raw.SetBinError(bin, fit_tpl.GetParError(2)/proj_sub.GetBinWidth(1))
        else:
            hist_raw.SetBinContent(bin, (fit_tpl.GetParameter(2)+fit_tpl.GetParameter(5))/proj_sub.GetBinWidth(1))
            hist_raw.SetBinError(bin, ROOT.TMath.Sqrt(fit_tpl.GetParError(2)**2+fit_tpl.GetParError(5)**2)/proj_sub.GetBinWidth(1))
        hist_count = 0

        for i in range(1, proj_sub.GetNbinsX()+1):
            hist_count += proj_sub.GetBinContent(i)
    
    #################################################
    # compute the expected invariant mass plot after
    # one month of data taking
    #################################################

    proj_sub_scale = proj_data.Clone()
    if dgauss:
        fit_tpl.SetParameter(0, fit_tpl.GetParameter(0)*full_run/nev)
        fit_tpl.SetParameter(1, fit_tpl.GetParameter(1)*full_run/nev)
        fit_tpl.SetParameter(2, fit_tpl.GetParameter(2)*full_run/nev)
        fit_tpl.SetParameter(4, fit_tpl.GetParameter(5)*full_run/nev)
    else:
        fit_tpl.SetParameter(0, fit_tpl.GetParameter(0)*full_run/nev)
        fit_tpl.SetParameter(1, fit_tpl.GetParameter(1)*full_run/nev)
        fit_tpl.SetParameter(2, fit_tpl.GetParameter(2)*full_run/nev)
    fit_bkg.SetParameter(0, fit_tpl.GetParameter(0))
    fit_bkg.SetParameter(1, fit_tpl.GetParameter(1))
    
    for i in range(1, proj_sub_scale.GetNbinsX()+1):
        val = proj_sub_scale.GetXaxis().GetBinCenter(i)
        proj_sub_scale.SetBinContent(i,fit_tpl.Eval(val))
        proj_sub_scale.SetBinError(i,proj_sub.GetBinError(i)*ROOT.TMath.Sqrt(full_run/nev))
    
    proj_sub_scale.GetListOfFunctions().Add(fit_tpl)
    proj_sub_scale.GetListOfFunctions().Add(fit_bkg)

    cv_sub_or = ROOT.TCanvas("cv_sub_or","cv_sub_or")

    cv_sub_or.SetLeftMargin(0.12)
    cv_sub_or.cd()

    ROOT.gStyle.SetOptStat(0)

    proj_sub.GetXaxis().SetRangeUser(MASS_MIN,1.06)
    proj_sub.SetTitle("") 
    proj_sub.GetYaxis().SetRangeUser(proj_sub.GetMinimum()*1.5,proj_sub.GetMaximum()*1.5)  
    proj_sub.Draw("e")
    pinfo_sub = ROOT.TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo_sub.SetBorderSize(0)
    pinfo_sub.SetFillStyle(0)
    pinfo_sub.SetTextAlign(30+3)
    pinfo_sub.SetTextFont(42)
    pinfo_sub.AddText("Pb-Pb #sqrt{s_{NN}} = 8.8 GeV, centrality 0-5%")
    pinfo_sub.AddText(f'{(bin-1)*pt_bin_width:.1f}'+' #leq #it{p}_{T} <'+f' {(bin)*pt_bin_width:.1f}'+' GeV/#it{c}')
    pinfo_sub.Draw()
    cv_sub_or.SaveAs(results_path+f'/no_scale_{suffix}_{(bin-1)*pt_bin_width:.1f}_{(bin)*pt_bin_width:.1f}.pdf')

    cv_sub = ROOT.TCanvas("cv_sub","cv_sub")
    cv_sub.SetLeftMargin(0.12)
    cv_sub.cd()
    ROOT.gStyle.SetOptStat(0)
    proj_sub_scale.GetXaxis().SetRangeUser(MASS_MIN,1.06)
    proj_sub_scale.SetTitle("")
    proj_sub_scale.Draw("e")

    signal = fit_tpl.GetParameter(2) / proj_sub_scale.GetBinWidth(1)
    errsignal = fit_tpl.GetParError(2) / proj_sub_scale.GetBinWidth(1)

    pinfo_sub.AddText(f'S {signal:.0f} #pm {errsignal:.0f}')
    pinfo_sub.Draw()
    if bin <= 30:
        cv_sub.SaveAs(results_path+f'/scale_{suffix}_{(bin-1)*pt_bin_width:.1f}_{(bin)*pt_bin_width:.1f}.pdf')
    dir_scale.cd()
    proj_sub_scale.Write()

#################################################
# produce a series of integrated plots
#################################################
output.cd()
hist_raw.Write()
hist_raw_mc.Write()
ROOT.gStyle.SetOptStat(0)
proj_data_int = hist_data.ProjectionY(f'{hist_data.GetName()}_all_pt', 1, hist_data.GetNbinsY())
proj_data_int.Write()
proj_mix_int = hist_mix.ProjectionY(f'{hist_mix.GetName()}_all_pt', 1, hist_mix.GetNbinsY())
counts_data = 0
counts_mix = 0
bin_min = proj_data_int.GetXaxis().FindBin(1.06)
bin_max = proj_data_int.GetXaxis().FindBin(MASS_MAX)
for bin_pro in range(bin_min, bin_max+1):
    counts_data += proj_data_int.GetBinContent(bin_pro)
    counts_mix += proj_mix_int.GetBinContent(bin_pro)

normalization = counts_data/counts_mix

for bin_pro in range(1, proj_data_int.GetNbinsX()+1):
    proj_data_int.SetBinContent(bin_pro, full_run/nev*proj_data_int.GetBinContent(bin_pro))
    proj_mix_int.SetBinContent(bin_pro, full_run/nev*normalization*proj_mix_int.GetBinContent(bin_pro))
    proj_sub_tot.SetBinContent(bin_pro, full_run/nev*proj_sub_tot.GetBinContent(bin_pro))


cv_mix = ROOT.TCanvas("cv_mix","cv_mix")
cv_mix.cd()
range_min = proj_mix_int.GetXaxis().GetBinLowEdge(8)
range_max = proj_mix_int.GetXaxis().GetBinUpEdge(proj_mix_int.GetNbinsX()-4)

proj_mix_int.GetXaxis().SetRangeUser(range_min, range_max)
proj_mix_int.SetTitle("Event Mixing;m (GeV/#it{c}^{2});")
proj_mix_int.GetYaxis().SetTitle(f'Counts/{1000*proj_mix_int.GetBinWidth(1):.2f}'+' MeV/#it{c}^{2}')
proj_mix_int.SetMarkerStyle(20)
proj_mix_int.Draw("e")
cv_mix.Write()
cv_mix.SaveAs(results_path+f'/mix_{suffix}.png')
cv_mix.SaveAs(results_path+f'/mix_{suffix}.pdf')
cv_data = ROOT.TCanvas("cv_data","cv_data")
cv_data.cd()
proj_data_int.GetXaxis().SetRangeUser(range_min, range_max)
proj_data_int.SetTitle("Generated Data;m (GeV/#it{c}^{2});")
proj_data_int.GetYaxis().SetTitle(f'Counts/{1000*proj_data_int.GetBinWidth(1):.2f}'+' MeV/#it{c}^{2}')
proj_data_int.SetMarkerStyle(20)
proj_data_int.Draw("e")
cv_data.Write()
cv_data.SaveAs(results_path+f'/data_{suffix}.png')
cv_data.SaveAs(results_path+f'/data_{suffix}.pdf')

proj_sub_tot.Write()
proj_mix_int.Write()
proj_sub_int = proj_data_int.Clone()
proj_sub_int.SetName("sub_all_pt")
proj_sub_int.Add(proj_mix_int, -1)

cv_sub = ROOT.TCanvas("cv_sub","cv_sub")
cv_sub.cd()

range_min_y = MASS_MAX*proj_sub_int.GetMinimum() if (proj_sub_int.GetMinimum()<0) else 0.8*proj_sub_int.GetMinimum()
range_max_y = 1.3*proj_sub_int.GetMaximum()
proj_sub_int.GetXaxis().SetRangeUser(range_min, range_max)
proj_sub_int.GetYaxis().SetRangeUser(range_min_y, range_max_y)
proj_sub_int.SetTitle(';m (GeV/#it{c}^{2});')
proj_sub_int.GetYaxis().SetTitle(f'Counts/{1000*proj_sub_int.GetBinWidth(1):.2f}'+' MeV/#it{c}^{2}')
proj_sub_int.SetMarkerStyle(20)
proj_sub_int.Fit(fit_tpl,"IMR+","", MASS_MIN, MASS_MAX)

fit_bkg.SetParameter(0, fit_tpl.GetParameter(0))
fit_bkg.SetParameter(1, fit_tpl.GetParameter(1))

proj_sub_int.GetListOfFunctions().Add(fit_bkg)
proj_sub_int.Draw("e")
cv_sub.Write()
proj_scale = proj_sub_int.Clone()

cv_scale = ROOT.TCanvas("cv_scale","cv_scale")
cv_scale.cd()

for bin_pro in range(1, proj_scale.GetNbinsX()+1):
    val = proj_scale.GetXaxis().GetBinCenter(bin_pro)
    proj_scale.SetBinContent(bin_pro,fit_tpl.Eval(val))
    proj_scale.SetBinError(bin_pro,ROOT.TMath.Sqrt(ROOT.TMath.Abs(fit_tpl.Eval(val)))*5./4.)

proj_scale.GetListOfFunctions().Add(fit_tpl)
proj_scale.GetListOfFunctions().Add(fit_bkg)
proj_scale.Draw("e")

#print fit info on the canvas
pinfo_m = ROOT.TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
pinfo_m.SetBorderSize(0)
pinfo_m.SetFillStyle(0)
pinfo_m.SetTextAlign(30+3)
pinfo_m.SetTextFont(42)

signal = fit_tpl.GetParameter(2) / proj_scale.GetBinWidth(1)
errsignal = fit_tpl.GetParError(2) / proj_scale.GetBinWidth(1)
bkg = 0#fit_bkg.Integral(range_min, range_max) / proj_scale.GetBinWidth(1)

for bin_pro in range(1, proj_scale.GetNbinsX()+1):
    val = proj_scale.GetXaxis().GetBinCenter(bin_pro)
    bkg += fit_bkg.Eval(val)/proj_scale.GetBinWidth(1)

signif = signal/ROOT.TMath.Sqrt(signal+bkg)
pinfo_m.AddText("Pb-Pb #sqrt{s_{NN}} = 8.8 GeV, centrality 0-5%")
pinfo_m.AddText("0.0 #leq #it{p}_{T} < 3.0 GeV/#it{c}")
pinfo_m.AddText(f'S {signal} #pm {errsignal}')

pinfo_m.Draw()

cv_scale.Write()
cv_scale.SaveAs(results_path+f'/mass_{suffix}.pdf')

proj_ratio_int = proj_data_int.Clone()
proj_ratio_int.SetName("ratio_all_pt")
proj_ratio_int.Divide(proj_mix_int)

proj_sub_int.Write()
proj_ratio_int.Write()

#################################################
# fill the efficiency and mass shift histograms
# from the signal only file
#################################################

def preselection_efficiency(self, pt_bins, save=True, suffix=''):
        cut  =  f'{pt_bins[0]}<=pt<={pt_bins[len(pt_bins)-1]}'         
            
        pres_histo = au.h1_preselection_efficiency(pt_bins)
        gen_histo = au.h1_generated(pt_bins)

        pres_histo.Divide(gen_histo)

        if save:
            path = os.environ['EFFICIENCIES']

            filename = path + f'/PreselEff{suffix}.root'
            t_file = ROOT.TFile(filename, 'recreate')
            
            pres_histo.Write()
            t_file.Close()

binning = array("d", PT_BINS)
hist_eff = ROOT.TH1D("hist_eff",";#it{p}_{T} (GeV/#it{c}); Efficiency x Acceptance", nbins, PT_BINS[0],PT_BINS[-1])
hist_gen = ROOT.TH1D("hist_gen",";#it{p}_{T} (GeV/#it{c}); Counts", nbins, PT_BINS[0],PT_BINS[-1])

df_rec = uproot.open(signal_file)["ntcand"].arrays(library="pd")
df_gen = uproot.open(signal_file)["ntgen"].arrays(library="pd")

for pt in df_rec['pt']:#.to_records(index=False)
    hist_eff.Fill(pt)

for pt in df_gen['pt']:#.to_records(index=False)
    hist_gen.Fill(pt)

for bin in range(1, nbins+1):
    rec = hist_eff.GetBinContent(bin)
    gen = hist_gen.GetBinContent(bin)
    if gen < 1:
        gen = 1
    eff = rec/gen
    if eff > 1:
        eff = 1
    hist_eff.SetBinContent(bin, eff)
    hist_eff.SetBinError(bin, ROOT.TMath.Sqrt(eff*(1-eff)/gen))
    
hist2d_shift = ROOT.TH2D("hist2d_shift", ";#it{p}_{T} (GeV/#it{c}); #Delta m (GeV/#it{c}^{2}); Counts", nbins, PT_BINS[0], PT_BINS[-1], 600, -0.03, 0.03)
hist_shift = ROOT.TH1D("hist_shift", ";#it{p}_{T} (GeV/#it{c}); #Delta m (GeV/#it{c}^{2})", nbins, PT_BINS[0], PT_BINS[-1])
hist_mass_corr = hist_mass.Clone("hist_mass_corr")

for index, row in df_rec.iterrows():
    hist2d_shift.Fill(row["pt"],row["m"]-mass)

#################################################
# compute the pT spectra to obtain the T param 
# and the yield
#################################################


mult = 0
err_mult = 0
pt_range_factor = au.get_pt_integral(pt_distr_gen, 0.2,2.0)/au.get_pt_integral(pt_distr_gen)
hist_eff.Write()
for bin_pro in range(1, 12+1):
    eff = hist_eff.GetBinContent(bin_pro)
    if eff==0:
        eff = 1
    counts = hist_raw.GetBinContent(bin_pro)/nev
    err = hist_raw.GetBinError(bin_pro)/nev
    hist_pt.SetBinContent(bin_pro, counts/hist_pt.GetBinWidth(bin_pro)/eff)
    hist_pt.SetBinError(bin_pro, err/hist_pt.GetBinWidth(bin_pro)/eff)
    mult += counts/eff
    err_mult += (err/eff)**2

err_mult = ROOT.TMath.Sqrt(err_mult)/BRATIO/pt_range_factor
mult /= BRATIO*pt_range_factor
#if err_mult == 0:
#err_mult = 1

print("**************************************************")
print("multiplicity: ", mult," +- ",err_mult)
print("multiplicity gen: ", MULTIPLICITY)
print("z_gauss: ", (MULTIPLICITY-mult)/err_mult)
print("**************************************************")

hist_pt.Fit("pt_distr","IMR0","",0.2,2.0)
hist_pt.Fit("pt_distr","IMR+","",0.2,2.0)
cv = ROOT.TCanvas("cv","cv")
pinfo = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
pinfo.SetBorderSize(0)
pinfo.SetFillStyle(0)
pinfo.SetTextAlign(30+3)
pinfo.SetTextFont(42)
pinfo.AddText("Pb-Pb #sqrt{#it{s}_{NN}} = 8.8 GeV, 0-5%")
pinfo.AddText(f'T = {pt_distr.GetParameter(1)*1000:.1f} #pm {pt_distr.GetParError(1)*1000:.1f} MeV')
pinfo.AddText('T_{gen} = 244.6 MeV')
hist_pt.SetMarkerStyle(20)
hist_pt.SetMarkerColor(ROOT.kBlue)
hist_pt.GetXaxis().SetRangeUser(0.2,2.0)
hist_pt.Draw("e")
pinfo.Draw()
cv.Write()
cv.SaveAs(results_path+"/pt_"+suffix+".png")
cv.SaveAs(results_path+"/pt_"+suffix+".pdf")
cv.SetLogy()
cv.SaveAs(results_path+"/pt_"+suffix+"_logy.png")
cv.SaveAs(results_path+"/pt_"+suffix+"_logy.pdf")

scale_factor = ROOT.TMath.Sqrt(nev/full_run)
hist_pt_scaled = hist_pt.Clone("hist_pt_scaled")
for bin_pro in range(1, hist_pt_mc.GetNbinsX()+1):
    hist_pt_scaled.SetBinError(bin_pro, hist_pt.GetBinError(bin_pro)*scale_factor)
    if bin_pro < 3 or bin_pro > 23:
        hist_pt_scaled.SetBinContent(bin_pro, 0)
        hist_pt_scaled.SetBinError(bin_pro, 0)


#################################################
# produce the expected pT spectra after one month
# of data taking
#################################################

hist_pt_scaled.Fit("pt_distr","IMR+","",0.2,2.0)

for bin_pro in range(1, hist_pt_mc.GetNbinsX()+1):
    hist_pt_scaled.SetBinContent(bin_pro, pt_distr.Eval(hist_pt_scaled.GetBinCenter(bin_pro)))
    if bin_pro < 3 or bin_pro > 23:
        hist_pt_scaled.SetBinContent(bin_pro, 0)
        hist_pt_scaled.SetBinError(bin_pro, 0)

hist_pt_scaled.GetXaxis().SetRangeUser(0.2,2.0)
cv_scaled = ROOT.TCanvas("cv_scaled","cv_scaled")
pinfo_scaled = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
pinfo_scaled.SetBorderSize(0)
pinfo_scaled.SetFillStyle(0)
pinfo_scaled.SetTextAlign(30+3)
pinfo_scaled.SetTextFont(42)
pinfo_scaled.AddText("Pb-Pb #sqrt{#it{s}_{NN}} = 8.8 GeV, 0-5%")
pinfo_scaled.AddText(f'T = {pt_distr.GetParameter(1)*1000:.2f} #pm {pt_distr.GetParError(1)*1000:.2f} MeV')
hist_pt_scaled.SetMarkerStyle(20)
hist_pt_scaled.SetMarkerColor(ROOT.kBlue)
hist_pt_scaled.Draw("e")
pinfo_scaled.Draw()
cv_scaled.Write()
cv_scaled.SaveAs(results_path+"/pt_"+suffix+"_scaled.png")
cv_scaled.SaveAs(results_path+"/pt_"+suffix+"_scaled.pdf")
cv_scaled.SetLogy()
cv_scaled.SaveAs(results_path+"/pt_"+suffix+"_scaled_logy.png")
cv_scaled.SaveAs(results_path+"/pt_"+suffix+"_scaled_logy.pdf")
hist_pt_scaled.Write()

#################################################
# exact generated pT distribution 
#################################################
err_mult = 0
mult = 0

pt_range_factor = au.get_pt_integral(pt_distr_gen, 0.0,3.0)/au.get_pt_integral(pt_distr_gen)
for bin_pro in range(1, hist_pt_mc.GetNbinsX()+1):
    eff = hist_eff.GetBinContent(bin_pro)
    if eff==0 :
        eff = 1
    counts = hist_raw_mc.GetBinContent(bin_pro)/nev
    err = hist_raw_mc.GetBinError(bin_pro)/nev
    hist_pt_mc.SetBinContent(bin_pro, counts/hist_pt_mc.GetBinWidth(bin_pro)/eff)
    hist_pt_mc.SetBinError(bin_pro, err/hist_pt_mc.GetBinWidth(bin_pro)/eff)
 
    mult += counts/eff/BRATIO/pt_range_factor
    err_mult += (err/eff)**2
    print("mult(",bin_pro,"): ",mult)
    print("counts(",bin_pro,"): ",counts/eff/BRATIO/pt_range_factor)

err_mult = ROOT.TMath.Sqrt(err_mult)/BRATIO/pt_range_factor
#mult /= BRATIO*pt_range_factor
print("**************************************************")
print("TEST MC")
print("multiplicity: ", mult," +- ",err_mult)
print("multiplicity gen: ", MULTIPLICITY)
print("z_gauss: ", (MULTIPLICITY-mult)/err_mult)
print("**************************************************")
hist_pt_mc.Fit("pt_distr","IR+","",0.,3.0)
cv_mc = ROOT.TCanvas("cv_mc","cv_mc")
pinfo_mc = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
pinfo_mc.SetBorderSize(0)
pinfo_mc.SetFillStyle(0)
pinfo_mc.SetTextAlign(30+3)
pinfo_mc.SetTextFont(42)
pinfo_mc.AddText("Pb-Pb #sqrt{#it{s}_{NN}} = 8.8 GeV, 0-5%")
pinfo_mc.AddText(f'T = {pt_distr.GetParameter(1)*1000:.2f} #pm {pt_distr.GetParError(1)*1000:.2f} MeV')
pinfo.AddText('T_{gen} = 244.6 MeV')
hist_pt_mc.SetMarkerStyle(20)
hist_pt_mc.SetMarkerColor(ROOT.kBlue)
hist_pt_mc.Draw("e")
pinfo_mc.Draw()
cv_mc.Write()
cv_mc.SaveAs(results_path+"/pt_"+suffix+"_mc.png")
cv_mc.SaveAs(results_path+"/pt_"+suffix+"_mc.pdf")
hist_pt_mc.Write()

#################################################
# compute the mass for each pT bin
#################################################

for bin in range(1, hist_shift.GetNbinsX()+1):
    hist_tmp = hist2d_shift.ProjectionY(f'mass_shift_pt_{ptbin_lw:.1f}_{ptbin_up:.1f}', bin, bin)
    hist_shift.SetBinContent(bin, hist_tmp.GetMean())
    hist_shift.SetBinError(bin, hist_tmp.GetMeanError())
    hist_mass_corr.SetBinContent(bin, hist_mass_corr.GetBinContent(bin)-hist_tmp.GetMean())
hist_shift.Write()
hist2d_shift.Write()
hist_mass.Write()
fit_mass = ROOT.TF1("fit_mass","pol0",0.2,2.0)
hist_mass_corr.Fit(fit_mass,"MR+")
cv_mass = ROOT.TCanvas("cv_mass","cv_mass")
ROOT.gPad.SetLeftMargin(0.15)
pinfo_mass = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
pinfo_mass.SetBorderSize(0)
pinfo_mass.SetFillStyle(0)
pinfo_mass.SetTextAlign(30+3)
pinfo_mass.SetTextFont(42)
pinfo_mass.AddText("Pb-Pb #sqrt{#it{s}_{NN}} = 8.8 GeV, 0-5%")
pinfo_mass.AddText(f'm = {fit_mass.GetParameter(0):.5f} #pm {fit_mass.GetParError(0):.5f} '+'GeV/#it{c}^{2}')
pinfo_mass.AddText('m_{gen} = '+f'{mass} MeV')
hist_mass_corr.SetMarkerStyle(20)
hist_mass_corr.SetMarkerColor(ROOT.kBlue)
hist_mass_corr.GetXaxis().SetRangeUser(0.2,2.0)
hist_min_val = hist_mass_corr.GetBinContent(2)-2*hist_mass_corr.GetBinError(2)
hist_max_val = hist_mass_corr.GetBinContent(2)+3*hist_mass_corr.GetBinError(2)
for iBin in range(2, 12):#hist_mass_corr.GetNbinsX()):
    tmp_min = hist_mass_corr.GetBinContent(iBin)-2*hist_mass_corr.GetBinError(iBin)
    tmp_max = hist_mass_corr.GetBinContent(iBin)+3*hist_mass_corr.GetBinError(iBin)
    
    print("tmp_max(",iBin,"): ",tmp_max)
    print("tmp_min(",iBin,"): ",tmp_min)
    if tmp_max > hist_max_val:
        hist_max_val = tmp_max
    if tmp_min < hist_min_val:
        hist_min_val = tmp_min
hist_mass_corr.GetYaxis().SetRangeUser(hist_min_val,hist_max_val)
hist_mass_corr.Draw("e")

merr_box = ROOT.TBox(0.2, fit_mass.GetParameter(0)-fit_mass.GetParError(0), 2.0, fit_mass.GetParameter(0)+fit_mass.GetParError(0))
#merr_box.SetLineStyle(7)
merr_box.SetLineWidth(1)
merr_box.SetLineColor(ROOT.kRed)
merr_box.SetFillColor(ROOT.kOrange)
merr_box.SetFillStyle(3004)
merr_box.Draw("same")

mgen_line = ROOT.TLine(0.2, mass, 2.0, mass)
mgen_line.SetLineStyle(7)
mgen_line.SetLineColor(ROOT.kBlue)
mgen_line.Draw("same")


leg = ROOT.TLegend(0.2,0.2,.65,0.45)
leg.SetFillStyle(0)
leg.SetMargin(0.2) #separation symbol-text
leg.SetBorderSize(0)
leg.SetTextSize(0.025)
leg.AddEntry(mgen_line, "PDG value", "l")
leg.AddEntry(merr_box, "Measured mass", "fl")
leg.Draw()
pinfo_mass.Draw()
cv_mass.Write()
cv_mass.SaveAs(results_path+"/mass_values_"+suffix+".pdf")
cv_mass.SaveAs(results_path+"/mass_values_"+suffix+".png")
hist_mass_corr.Write()

for bin in range(1, hist_shift.GetNbinsX()+1):
    hist_mass_corr.SetBinError(bin, hist_mass_corr.GetBinError(bin)*ROOT.TMath.Sqrt(nev/full_run))
hist_mass_corr.Fit(fit_mass,"MR+")
hist_mass_corr.SetName("hist_mass_scaled")
cv_mass_scaled = ROOT.TCanvas("cv_mass_scaled","cv_mass_scaled")
pinfo_mass_scaled = ROOT.TPaveText(0.5, 0.65, 0.88, 0.86, "NDC")
pinfo_mass_scaled.SetBorderSize(0)
pinfo_mass_scaled.SetFillStyle(0)
pinfo_mass_scaled.SetTextAlign(30+3)
pinfo_mass_scaled.SetTextFont(42)
pinfo_mass_scaled.AddText("Pb-Pb #sqrt{#it{s}_{NN}} = 8.8 GeV, 0-5%")
pinfo_mass_scaled.AddText(f'm = {fit_mass.GetParameter(0):.7f} #pm {fit_mass.GetParError(0):.7f} '+'GeV/#it{c}^{2}')
pinfo_mass_scaled.AddText('m_{gen} = '+f'{mass} MeV')
hist_mass_corr.SetMarkerStyle(20)
hist_mass_corr.SetMarkerColor(ROOT.kBlue)
hist_mass_corr.Draw("e")
pinfo_mass_scaled.Draw()
cv_mass_scaled.Write()
cv_mass_scaled.SaveAs(results_path+"/mass_"+suffix+"_scaled.pdf")
cv_mass_scaled.SaveAs(results_path+"/mass_"+suffix+"_scaled.png")
hist_mass_corr.Write()
output.Close()

