from itertools import count
import math
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10
import warnings

from pyparsing import col

import aghast
from matplotlib.pyplot import axis
import numpy as np
import uproot
from hipe4ml.model_handler import ModelHandler
import ROOT
from ROOT import TF1, TH1D, TCanvas, TPaveStats, TPaveText, gStyle, THnSparseD, TMath, TFile
from array import array
import pickle
import pandas as pd
from scipy import stats
import os
# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def least_significant_digit(num_list):
    lsd = -1
    for num in num_list:
        if num == 0:
            continue
        num_string = str(num)
        #num_string = str(num_string)
        if '.' in num_string:
            # There's a decimal point. Figure out how many digits are to the right
            # of the decimal point and negate that.
            power =  len(num_string.partition('.')[2])
            print(power)
            lsd_i = int(num_string.partition('.')[2][power-1])*10**(-power)
        else:
            # No decimal point. Count trailing zeros.
            power = len(num_string) - len(num_string.rstrip('0'))
            print(power)
            lsd_i = int(num_string[-power-1])*10**(power)
        if lsd == -1 or lsd > lsd_i:
            lsd = lsd_i
    print(lsd)
    return lsd
        

def get_skimmed_large_data_std_hsp(mass, data_path, pt_bins, preselection='', range=0.04, mass_bins=40, split_sig = False):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('\nStarting standard selections appplication')

    pt_min = pt_bins[0]
    pt_max = pt_bins[-1]
    #minimum_pt_bins = int((pt_max-pt_min)/least_significant_digit(pt_bins))
    
    minimum_pt_bins = int((pt_max-pt_min)/0.01) #least_significant_digit(pt_bins))
    mass_range_min = mass*(1-range)
    mass_range_max = mass*(1+range)

    nbins = array('i', [mass_bins, minimum_pt_bins, 2])
    xmin  = array('d', [mass_range_min, pt_min, -0.5])
    xmax  = array('d', [mass_range_max, pt_max,  1.5])
    nvar = 3 if split_sig else 2
    hsparse = THnSparseD('sparse_m_pt', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});true;counts', nvar, nbins, xmin, xmax)

    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    if preselection != "":
        preselection = " and "+preselection

    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            data_range = f'{ptbin[0]}<pt<{ptbin[1]} and {mass_range_min}<m<{mass_range_max}'
            df_tmp = data.query(data_range+preselection)

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['true'][ind]])
                hsparse.Fill(x)

    return hsparse 

def get_skimmed_large_data_hsp(mass, data_path, pt_bins, training_columns, suffix='', preselection='', range=0.04, mass_bins=40, split_sig = False):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication')

    model_handler_list = []
    eff_score_array_list = []
    tsd_min = 13
    tsd_max = 13
    handlers_path = "../Models/handlers"
    efficiencies_path = "../Results/Efficiencies"
    for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
        pt_index = pt_bins.index(ptbin[0])

        info_string = f'_{ptbin[0]}{ptbin[1]}'
        filename_handler = handlers_path + '/model_handler_' +suffix+ info_string + '.pkl'
        filename_efficiencies = efficiencies_path + '/Eff_Score_' + suffix + info_string + '.npy'

        model_handler_list.append(ModelHandler())
        model_handler_list[pt_index].load_model_handler(filename_handler)

        eff_score_array_list.append(np.load(filename_efficiencies))

        tsd = eff_score_array_list[pt_index][1][-1]
        if tsd < tsd_min:
            tsd_min = tsd

    pt_min = pt_bins[0]
    pt_max = pt_bins[-1]
    minimum_pt_bins = int((pt_max-pt_min)/0.01) #least_significant_digit(pt_bins))
    mass_range_min = mass*(1-range)
    mass_range_max = mass*(1+range)
    nbin_score = 2000
    nbins = array('i', [mass_bins, minimum_pt_bins, nbin_score,    2])
    xmin  = array('d', [mass_range_min,     pt_min,    tsd_min, -0.5])
    xmax  = array('d', [mass_range_max,     pt_max,   tsd_max,   1.5])
    nvar = 4 if split_sig else 3
    hsparse = THnSparseD('sparse_m_pt_s', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});score;true;counts', nvar, nbins, xmin, xmax)


    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    if preselection != "":
        preselection = " and "+preselection


    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            pt_index = pt_bins.index(ptbin[0])
            tsd = eff_score_array_list[pt_index][1][-1]

            data_range = f'{ptbin[0]}<pt<{ptbin[1]} and {mass_range_min}<m<{mass_range_max}'
            df_tmp = data.query(data_range+preselection)
            df_tmp.insert(0, 'score', model_handler_list[pt_index].predict(df_tmp[training_columns]))
            df_tmp = df_tmp.query('score>@tsd')

            for ind in df_tmp.index:
                if nvar == 3:
                    x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind]])
                else:
                    x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind], df_tmp['true'][ind]])
                hsparse.Fill(x)

    return hsparse


def expected_signal_counts(bw, multiplicity, branching_ratio, pt_range, eff, nevents):
    signal = multiplicity * nevents* branching_ratio  * bw.Integral(pt_range[0], pt_range[1], 1e-8) / bw.Integral(0, 10, 1e-8)
    return int(round(signal * eff))


def significance_error(signal, background):
    signal_error = np.sqrt(signal + 1e-10)
    background_error = np.sqrt(background + 1e-10)

    sb = signal + background + 1e-10
    sb_sqrt = np.sqrt(sb)

    s_propag = (sb_sqrt + signal / (2 * sb_sqrt))/sb * signal_error
    b_propag = signal / (2 * sb_sqrt)/sb * background_error

    if signal+background == 0:
        return 0

    return np.sqrt(s_propag * s_propag + b_propag * b_propag)


def expo(x, tau):
    return np.exp(-x / (tau * 0.029979245800))


def h1_preselection_efficiency(ptbins, name='PreselEff'):
    th1 = TH1D(name, ';#it{p}_{T} (GeV/#it{c});Preselection efficiency', len(ptbins) - 1, np.array(ptbins, 'double'))
    th1.SetDirectory(0)

    return th1


def h1_generated(ptbins, name='Generated'):
    th1 = TH1D(name, ';#it{p}_{T} (GeV/#it{c}); Generated', len(ptbins)-1, np.array(ptbins, 'double'))
    th1.SetDirectory(0)

    return th1


def h1_rawcounts(ptbins, name='RawCounts', suffix=''):
    th1 = TH1D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});Raw counts', len(ptbins)-1, np.array(ptbins, 'double'))
    th1.SetDirectory(0)

    return th1


def h1_significance(ptbins, name='Significance', suffix=''):
    th1 = TH1D(f'{name}{suffix}', ';#it{p}_{T} (GeV/#it{c});Significance', len(ptbins)-1, np.array(ptbins, 'double'))
    th1.SetDirectory(0)

    return th1


def h1_invmass(counts, pt_range, name=''):
    ghist = aghast.from_numpy(counts)
    th1 = aghast.to_root(ghist, f'pT{pt_range[0]}{pt_range[1]}_{name}')
    th1.SetDirectory(0)
    return th1

def h1_from_sparse(hnsparse, pt_range, score, name=''):
    n_pt_bins = hnsparse.GetAxis(1).GetNbins()
    pt_min = hnsparse.GetAxis(1).GetBinLowEdge(1)
    pt_max = hnsparse.GetAxis(1).GetBinUpEdge(1)
    step = (pt_max-pt_min)/n_pt_bins/2.
    ptbin_min = hnsparse.GetAxis(1).FindBin(pt_range[0]+step)
    ptbin_max = hnsparse.GetAxis(1).FindBin(pt_range[1]-step)
    if ptbin_max > hnsparse.GetAxis(1).GetNbins():
        ptbin_max = hnsparse.GetAxis(1).GetNbins()
    scorebin_min = hnsparse.GetAxis(2).FindBin(score)
    scorebin_max = hnsparse.GetAxis(2).GetNbins()
    if scorebin_min > scorebin_max:
        scorebin_min = scorebin_max
    hnsparse.GetAxis(1).SetRange(ptbin_min, ptbin_max)
    hnsparse.GetAxis(2).SetRange(scorebin_min, scorebin_max)
    th1 = hnsparse.Projection(0)
    #restore old setting
    hnsparse.GetAxis(1).SetRange(1, hnsparse.GetAxis(1).GetNbins())
    hnsparse.GetAxis(2).SetRange(1, scorebin_max)
    th1.SetName(name)
    width = th1.GetBinWidth(1)*1000 #to MeV
    th1.GetYaxis().SetTitle(r'Counts/%0.1f MeV' % (width))
    th1.SetTitle('')
    th1.SetDirectory(0)
    return th1

def bdt_efficiency_from_sparse(hnsparse, pt_range, name=''):
    hnsparse_clone = hnsparse.Clone()
    hnsparse_clone.GetAxis(3).SetRange(2,2)

    n_pt_bins = hnsparse.GetAxis(1).GetNbins()
    pt_min = hnsparse.GetAxis(1).GetBinLowEdge(1)
    pt_max = hnsparse.GetAxis(1).GetBinUpEdge(1)
    step = (pt_max-pt_min)/n_pt_bins/2.
    ptbin_min = hnsparse_clone.GetAxis(1).FindBin(pt_range[0]+step)
    ptbin_max = hnsparse_clone.GetAxis(1).FindBin(pt_range[1]-step)
    if ptbin_max > hnsparse_clone.GetAxis(1).GetNbins():
        ptbin_max = hnsparse_clone.GetAxis(1).GetNbins()
    hnsparse_clone.GetAxis(1).SetRange(ptbin_min, ptbin_max)

    th1 = hnsparse_clone.Projection(2)
    entries = th1.GetEntries()
    counts = th1.GetBinContent(th1.GetNbinsX())
    eff = counts/entries
    th1.SetBinContent(th1.GetNbinsX(), eff)
    th1.SetBinError(th1.GetNbinsX(), ROOT.TMath.Sqrt(eff*(1-eff)/entries))
    #print("***********************************************")
    ##print("bin: ",ibin)
    #print("counts: ",counts)
    #print("eff: ", eff)
    #print("err: ", ROOT.TMath.Sqrt(eff*(1-eff)/entries))
    for ibin in range(th1.GetNbinsX()-1, 0, -1):
        counts += th1.GetBinContent(ibin)
        eff = counts/entries
        if eff > 1:
            eff = 1
        th1.SetBinContent(ibin, eff)
        th1.SetBinError(ibin, ROOT.TMath.Sqrt(eff*(1-eff)/entries))
    th1.SetName(name)
    th1.GetYaxis().SetTitle('BDT efficiency')
    th1.SetTitle('')
    th1.SetDirectory(0)
    return th1

def bdt_efficiency_train(eff_score_array, name=''):
    th1 = ROOT.TH1D(name,";score;BDT efficiency",1000,eff_score_array[1][-1],eff_score_array[1][0])
    for ibin in range(1, len(eff_score_array[0])+1):
        score_bin = th1.GetXaxis().FindBin(eff_score_array[1][ibin-1])
        th1.SetBinContent(score_bin, eff_score_array[0][ibin-1])
        th1.SetBinError(score_bin, 0)
    eff = eff_score_array[0][-1]
    for ibin in range(1, th1.GetNbinsX()+1):
        if th1.GetBinContent(ibin) > 0 and th1.GetBinContent(ibin)<eff:
            eff = th1.GetBinContent(ibin)
        else:
            th1.SetBinContent(ibin,eff)

    return th1

def h1_from_sparse_std(hnsparse, pt_range, name=''):
    hnsparse_clone = hnsparse.Clone()
    step = 3.0/3000.0/2.
    ptbin_min = hnsparse_clone.GetAxis(1).FindBin(pt_range[0]+step)
    ptbin_max = hnsparse_clone.GetAxis(1).FindBin(pt_range[1]-step)
    if ptbin_max > hnsparse_clone.GetAxis(1).GetNbins():
        ptbin_max = hnsparse_clone.GetAxis(1).GetNbins()
    hnsparse_clone.GetAxis(1).SetRange(ptbin_min, ptbin_max)
    th1 = hnsparse_clone.Projection(0)
    th1.SetName(name)
    width = th1.GetBinWidth(1)*1000 #to MeV
    th1.GetYaxis().SetTitle(r'Counts/%0.1f MeV' % (width))
    th1.SetTitle('')
    th1.SetDirectory(0)
    return th1

def h1_invmass_ov(counts, pt_range, hist_range, bins=40, name=''):
    th1 = TH1D(f'pT{pt_range[0]}{pt_range[1]}', '', bins, hist_range[0], hist_range[1])

    for index in range(0, len(counts)):
        th1.SetBinContent(index+1, counts[index])
        th1.SetBinError(index + 1, math.sqrt(counts[index]))

    th1.SetDirectory(0)

    return th1

def round_to_error(x, error):
    return round(x, -int(floor(log10(abs(error)))))


def get_ptbin_index(th2, ptbin):
    return th2.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))


def get_ctbin_index(th2, ctbin):
    return th2.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))


def fit_hist(
        histo, pt_range, mass, directory, mc_fit_file, peak_width=3, sig_model="gauss", bkg_model="pol2", mass_range=0.04, Eint=17.3, fix_params = False, print = ""):
    hist_range = [mass*(1-mass_range),mass*(1+mass_range)]
    # canvas for plotting the invariant mass distribution
    cv = TCanvas(f'cv_{histo.GetName()}')
    # define the number of parameters depending on the bkg model
    if 'pol' in str(bkg_model):
        n_bkgpars = int(bkg_model[3]) + 1
    elif 'expo' in str(bkg_model):
        n_bkgpars = 2
    else:
        print(f'Unsupported background model {bkg_model}')
    norm_params = []
    if sig_model == 'gauss':
        norm_params.append(0)
        n_sigpars = 3
    elif sig_model == 'd-crystal':
        norm_params.append(4)
        n_sigpars = 7
    elif sig_model == 'exp-gauss':
        norm_params.append(0)
        n_sigpars = 5
    elif sig_model == 'kde':
        kde = mc_fit_file.Get(f'fit/kde_{pt_range[0]:.2f}_{pt_range[1]:.2f}')
        n_sigpars = 1
        norm_params.append(0)
    elif 'sub' in str(sig_model):
        norm_params.append(0)
        n_sigpars = 3
    else:
        if sig_model != 'd-gauss':
            print(f'Unsupported signal model {sig_model}')
            print(f'Model setted to default: d-gauss')
        n_sigpars = 6

    # define the fit function bkg_model + sig_model
    def sig_plus_bkg(x, par):
        sig_val = 0
        tf1_bkg = TF1('tf1_bkg', f'{bkg_model}(0)', 0, 5)
        for param in range(0, n_bkgpars):
            tf1_bkg.SetParameter(param, par[param])
        bkg_val = tf1_bkg.Eval(x[0])
        par_list = []
        for param in range(n_bkgpars, n_bkgpars+n_sigpars):
            par_list.append(par[param])
        
        if sig_model == 'd-gauss':
            fit_sig = TF1('fitSig', 'gausn(0)+gausn(3)', 0, 5)
            for param in range(0, n_sigpars):
                fit_sig.SetParameter(param, par_list[param])
            sig_val = fit_sig.Eval(x[0])
        elif sig_model == 'd-crystal':
            sig_val = double_sided_crystal_ball(x, par_list)
        elif sig_model == 'exp-gauss':
            sig_val = gauss_exp_tails(x, par_list)
        elif sig_model == 'kde':
            sig_val = kde.GetValue(x[0])*par[n_bkgpars]
        else:
            fit_sig = TF1('fitSig', 'gausn(0)', 0, 5)
            for param in range(0, n_sigpars):
                fit_sig.SetParameter(param, par_list[param])
            sig_val = fit_sig.Eval(x[0])
        return bkg_val + sig_val 
    
    if sig_model == "d-crystal" or sig_model=="kde" or sig_model == "exp-gauss":
        fit_tpl = TF1('fitTpl', sig_plus_bkg, 0, 5, n_bkgpars+n_sigpars)
        if sig_model != "kde":
            mc_function = mc_fit_file.Get(f'fit/{sig_model}_{pt_range[0]:.2f}_{pt_range[1]:.2f}')
            for param in range(0, n_sigpars):
                if param not in norm_params:
                    if fix_params:
                        fit_tpl.FixParameter(n_bkgpars + param, mc_function.GetParameter(param))
                    else:
                        fit_tpl.SetParameter(n_bkgpars + param, mc_function.GetParameter(param))
                        fit_tpl.SetParLimits(n_bkgpars + param, mc_function.GetParameter(param)-mc_function.GetParError(param)*3,
                                                            mc_function.GetParameter(param)+mc_function.GetParError(param)*3)
                else:
                    fit_tpl.SetParameter(n_bkgpars + param, 100)
                    fit_tpl.SetParLimits(n_bkgpars + param, 0, 1000)

    elif sig_model == 'gauss' or 'sub' in str(sig_model):
        fit_tpl = TF1('fitTpl', f'{bkg_model}(0) + gausn({n_bkgpars})', 0, 5)
        mc_function = mc_fit_file.Get(f'fit/gauss_{pt_range[0]:.2f}_{pt_range[1]:.2f}')
        for param in range(0, n_sigpars):
            if param not in norm_params:
                if fix_params and not 'sub' in str(sig_model):
                    fit_tpl.FixParameter(n_bkgpars + param, mc_function.GetParameter(param))
                else:
                    fit_tpl.SetParameter(n_bkgpars + param, mc_function.GetParameter(param))  
                    fit_tpl.SetParLimits(n_bkgpars + param, mc_function.GetParameter(param)-mc_function.GetParError(param)*3,
                                                        mc_function.GetParameter(param)+mc_function.GetParError(param)*3)
            else:
                fit_tpl.SetParameter(n_bkgpars + param, 100)
                fit_tpl.SetParLimits(n_bkgpars + param, 0, 1000)

    else:
        mc_function = mc_fit_file.Get(f'fit/{sig_model}_{pt_range[0]:.2f}_{pt_range[1]:.2f}')

        if fix_params:
            fit_tpl = TF1('fitTpl', f'{bkg_model}(0) + [{n_bkgpars+6}]*(gausn({n_bkgpars}) + gausn({n_bkgpars+3}))', 0, 5)
        else:
            fit_tpl = TF1('fitTpl', f'{bkg_model}(0) + gausn({n_bkgpars}) + gausn({n_bkgpars+3})', 0, 5)
            
        for param in range(0, n_sigpars):
            if fix_params:
                fit_tpl.FixParameter(n_bkgpars + param, mc_function.GetParameter(param))
            else:
                if param != 0 and param != 3:
                    fit_tpl.SetParameter(n_bkgpars + param, mc_function.GetParameter(param))
                    fit_tpl.SetParLimits(n_bkgpars + param, mc_function.GetParameter(param)-mc_function.GetParError(param)*3,
                                                        mc_function.GetParameter(param)+mc_function.GetParError(param)*3)
                else:
                    fit_tpl.SetParameter(n_bkgpars + param, 100)
                    fit_tpl.SetParLimits(n_bkgpars + param, 0, 1000)
    
    histo_bkg = histo.Clone("histo_bkg")
    peak_bins = [histo_bkg.GetXaxis().FindBin(mass-peak_width), histo_bkg.GetXaxis().FindBin(mass+peak_width)]
    for bin in range(peak_bins[0], peak_bins[1]+1):
        histo_bkg.SetBinContent(bin, 0)
        histo_bkg.SetBinError(bin, 0)
    bkg_tpl = TF1('bkgTpl', f'{bkg_model}(0)', 0, 5)
    histo_bkg.Fit(bkg_tpl, "RIM0Q","", mass*(1-mass_range), mass*(1+mass_range))
    for param in range(0, n_bkgpars):
        fit_tpl.SetParameter(param, bkg_tpl.GetParameter(param))
    
    # plotting stuff for fit_tpl
    fit_tpl.SetNpx(1000)
    fit_tpl.SetLineWidth(2)
    fit_tpl.SetLineColor(2)
    # plotting stuff for bkg model
    bkg_tpl.SetNpx(1000)
    bkg_tpl.SetLineWidth(2)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.SetLineColor(2)

    ########################################
    # plotting the fits
    ax_titles = ';m (GeV/#it{c}^{2});Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'+'/#it{c}^{2}'

    # invariant mass distribution histo and fit
    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)
    histo.SetTitle(ax_titles)
    histo.SetMaximum(1.5 * histo.GetMaximum())
    
    if sig_model == "kde":
        histo.Fit(fit_tpl, "MQR", "", hist_range[0], hist_range[1])
    else:
        histo.Fit(fit_tpl, "MIQR", "", hist_range[0], hist_range[1])
    if not 'sub' in str(sig_model):
        bkg_tpl.SetParameters(fit_tpl.GetParameters())
        histo.Draw()
        histo.SetDrawOption("e")
        histo.GetXaxis().SetRangeUser(hist_range[0], hist_range[1])
    else:
        histo_bkg.Draw()
        histo_bkg.SetDrawOption("e")
        histo_bkg.GetXaxis().SetRangeUser(hist_range[0], hist_range[1])
    # represent the bkg_model separately
    bkg_tpl.SetLineColor(600)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.Draw("same")

    # get the fit parameters
    signal = 0
    errsignal = 0
    if sig_model == "d-crystal":
        signal = fit_tpl.Integral(mass*(1-mass_range), mass*(1+mass_range)) / histo.GetBinWidth(1)
        background = bkg_tpl.Integral(mass*(1-mass_range), mass*(1+mass_range)) / histo.GetBinWidth(1)
        errsignal = math.sqrt(signal)
        signal -= background
    elif sig_model == 'd-gauss':
        #background = bkg_tpl.Integral(histo.GetXaxis().GetBinLowEdge(1), histo.GetXaxis().GetBinUpEdge(50)) / histo.GetBinWidth(1)
        #signal = histo.GetEntries()-background
        #errsignal = math.sqrt(signal)
        signal = (fit_tpl.GetParameter(n_bkgpars)+fit_tpl.GetParameter(n_bkgpars+3)) / histo.GetBinWidth(1)
        errsignal = math.sqrt(fit_tpl.GetParError(n_bkgpars)**2+fit_tpl.GetParError(n_bkgpars+3)**2) / histo.GetBinWidth(1)
        if fix_params:
            signal *= fit_tpl.GetParameter(n_bkgpars+6)
            errsignal = fit_tpl.GetParError(n_bkgpars+6)*(fit_tpl.GetParameter(n_bkgpars)+fit_tpl.GetParameter(n_bkgpars+3)) / histo.GetBinWidth(1)
        
    elif 'sub' in str(sig_model):
        width = float(sig_model[3:6])
        sigma = fit_tpl.GetParameter(n_bkgpars+2)
        signal = 0
        peak_bins = [histo.GetXaxis().FindBin(mass-width*sigma), histo.GetXaxis().FindBin(mass+width*sigma)]
        for bin in range(peak_bins[0], peak_bins[1]+1):
            signal += histo.GetBinContent(bin)
        peak_edges = [histo.GetXaxis().GetBinLowEdge(peak_bins[0]), histo.GetXaxis().GetBinUpEdge(peak_bins[1])]
        background = bkg_tpl.Integral(peak_edges[0], peak_edges[1]) / histo.GetBinWidth(1)
        errsignal = ROOT.TMath.Sqrt(signal+background)
        signal -= background
    else:
        for param in norm_params:
            signal += fit_tpl.GetParameter(n_bkgpars+param) / histo.GetBinWidth(1)
            errsignal += fit_tpl.GetParError(n_bkgpars+param)**2
        errsignal = math.sqrt(errsignal) / histo.GetBinWidth(1)

    # print fit info on the canvas
    pinfo = TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(30+3)
    pinfo.SetTextFont(42)

    string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{Eint} GeV, centrality {0}-{5}%'
    pinfo.AddText(string)

    string = f'{pt_range[0]:.2f}'+' #leq #it{p}_{T} < '+f'{pt_range[1]:.2f}'+' GeV/#it{c} '
    pinfo.AddText(string)

    string = f'S {signal:.0f} #pm {errsignal:.0f}'
    pinfo.AddText(string)

    pinfo.Draw()
    gStyle.SetOptStat(0)

    st = histo.FindObject('stats')
    if isinstance(st, TPaveStats):
        st.SetX1NDC(0.12)
        st.SetY1NDC(0.62)
        st.SetX2NDC(0.40)
        st.SetY2NDC(0.90)
        st.SetOptStat(0)
    directory.cd()
    histo.Write()
    cv.Write()
    if print != "":
        cv.SaveAs(print+".pdf")
        cv.SaveAs(print+".png")
    
    return (signal, errsignal)

#produce the correspondent mT = sqrt(pT^2 + mass^2)-mass array from the give pT array
def pt_array_to_mt_m0_array(pt_array, mass):
    mt_array = []
    for pt_item in pt_array:
        mt_array.append(math.sqrt(pt_item**2+mass**2)-mass)
    return mt_array

#produce the correspondent mT = sqrt(pT^2 + mass^2)-mass array from the give pT array
def pt_array_to_mt_array(pt_array, mass):
    mt_array = []
    for pt_item in pt_array:
        mt_array.append(math.sqrt(pt_item**2+mass**2))
    return mt_array

def crystal_ball(x, par):
    alpha = par[0]
    n =     par[1]
    meanx = par[2]
    sigma = par[3]
    nn =    par[4]
    p0 =    par[5]
    p1 =    par[6]
    a = TMath.Power((n/math.fabs(alpha)), n) * math.exp(-0.5*alpha*alpha)
    b = n/math.fabs(alpha) - math.fabs(alpha)
 
    arg = (x[0] - meanx)/sigma

    if arg > -1.*alpha:
        fitval = nn * math.exp(-0.5*arg*arg)
    else:
        fitval = nn * a * TMath.Power((b-arg), (-1*n))
      
    return fitval+p0+p1*x[0]

def double_sided_crystal_ball(x, par):
    alphaL = par[0]
    nL     = par[1]
    meanx  = par[2]
    sigma  = par[3]
    nn     = par[4]
    alphaR = par[5]
    nR     = par[6]
    
    a = TMath.Power((nL/TMath.Abs(alphaL)), nL) * TMath.Exp(-0.5*alphaL*alphaL)
    b = nL/TMath.Abs(alphaL) - TMath.Abs(alphaL)
    c = TMath.Power((nR/TMath.Abs(alphaR)), nR) * TMath.Exp(-0.5*alphaR*alphaR)
    d = nR/TMath.Abs(alphaR) - TMath.Abs(alphaR)
    
    arg = (x[0] - meanx)/sigma
    fitval = 0
    
    if arg > -1.*alphaL and arg < alphaR:
        fitval = nn * TMath.Exp(-0.5*arg*arg)
    elif arg <= -1.*alphaL:
        fitval = nn * a * TMath.Power((b-arg), (-1*nL))
    elif arg >= alphaR:
        fitval = nn  * c * TMath.Power((d+arg), (-1*nR))
        
    return fitval


def gauss_exp_tails(x, par):
    N = par[0]
    mu = par[1]
    sig = par[2]
    tau0 = par[3]
    tau1 = par[4]
    u = (x[0] - mu) / sig
    if (u < tau0):
        return N*TMath.Exp(-tau0 * (u - 0.5 * tau0))
    elif (u <= tau1):
        return N*TMath.Exp(-u * u * 0.5)
    else:
        return N*TMath.Exp(-tau1 * (u - 0.5 * tau1))

#integral of the e^{-tau*x} from 0 to infinity
def IntExp(tau):
    return TMath.Exp(-(tau**2)/2.) / TMath.Abs(tau)

#integral of gaussian from -infinity to x
def IntGauss(x):
  rootPiBy2 = TMath.Sqrt(TMath.PiOver2())
  return rootPiBy2 * (TMath.Erf(TMath.Abs(x) / TMath.Sqrt2()))

#save the BDT output distributions for signal and bkg for the train and test set
def save_bdt_output_plot(data, pt_range, suffix = ''):
    fig_path = os.environ['FIGURES']
    info_string = f'_{pt_range[0]}{pt_range[1]}'

    bdt_score_dir = fig_path + '/TrainTest'

    if not os.path.exists(bdt_score_dir):
        os.makedirs(bdt_score_dir)

    cv = TCanvas("cv","cv")
    gStyle.SetOptStat(0)
    data_sig = data[0]
    data_sig["true"] = data[1]
    data_sig_test = data[2]
    data_sig_test["true"] = data[3]
    data_sig = data_sig.query("true > 0.5")
    data_sig_test = data_sig_test.query("true > 0.5")
    stat_sig, pval_sig = stats.ks_2samp(data_sig['score'],data_sig_test['score'])
    nbins = 140
    max = data[0]["score"].max()
    min = data[0]["score"].min()
    counts_s, _ = np.histogram(data_sig['score'], bins=nbins, range=[min, max])
    counts_st, _ = np.histogram(data_sig_test['score'], bins=nbins, range=[min, max])
    hist_s = TH1D('hist_s', '; score; p.d.f', nbins, min, max)
    hist_st = TH1D('hist_st_st', '; score; p.d.f', nbins, min, max)
    for index in range(0, nbins):
        hist_s.SetBinContent(index+1, counts_s[index]/sum(counts_s))
        hist_s.SetBinError(index + 1, math.sqrt(counts_s[index])/sum(counts_s))
        hist_st.SetBinContent(index+1, counts_st[index]/sum(counts_st))
        hist_st.SetBinError(index + 1, math.sqrt(counts_st[index])/sum(counts_st))
    del data_sig
    del data_sig_test
    data_bkg = data[0]
    data_bkg["true"] = data[1]
    data_bkg_test = data[2]
    data_bkg_test["true"] = data[3]
    data_bkg = data_bkg.query("true < 0.5")
    data_bkg_test = data_bkg_test.query("true < 0.5")
    stat_bkg, pval_bkg = stats.ks_2samp(data_bkg['score'],data_bkg_test['score'])
    counts_b, _ = np.histogram(data_bkg['score'], bins=nbins, range=[min, max])
    counts_bt, _ = np.histogram(data_bkg_test['score'], bins=nbins, range=[min, max])
    hist_b = TH1D('hist_b', '; score; p.d.f', nbins, min, max)
    hist_bt = TH1D('hist_bt', '; score; p.d.f', nbins, min, max)
    for index in range(0, nbins):
        hist_b.SetBinContent(index+1, counts_b[index]/sum(counts_b))
        hist_b.SetBinError(index + 1, math.sqrt(counts_b[index])/sum(counts_b))
        hist_bt.SetBinContent(index+1, counts_bt[index]/sum(counts_bt))
        hist_bt.SetBinError(index + 1, math.sqrt(counts_bt[index])/sum(counts_bt))
    del data_bkg
    del data_bkg_test

    hist_s.SetFillColor(ROOT.kBlue)
    hist_s.SetFillStyle(3345)

    hist_b.SetFillColor(ROOT.kRed)
    hist_b.SetFillStyle(3354)

    hist_st.SetMarkerStyle(20)
    hist_st.SetMarkerColor(ROOT.kBlue)

    hist_bt.SetMarkerColor(ROOT.kRed)
    hist_bt.SetMarkerStyle(20)

    max = hist_s.GetMaximum() if hist_s.GetMaximum() > hist_b.GetMaximum() else hist_b.GetMaximum()
    hist_s.GetYaxis().SetRangeUser(10e-5,max*8)
    hist_s.Draw("same hist")
    hist_b.Draw("same hist")
    hist_st.Draw("same hist e")
    hist_bt.Draw("same hist e")

    legend = ROOT.TLegend(0.15, 0.72, 0.55, 0.85)
    legend.SetLineWidth(0)
    legend.SetFillColor(-1)
    legend.AddEntry(hist_s, "Signal training set", "f")
    legend.AddEntry(hist_b, "Background training set", "f")
    legend.AddEntry(hist_st, "Signal training set", "ep")
    legend.AddEntry(hist_bt, "Background training set", "ep")
    legend.Draw()

    pinfo = TPaveText(0.5, 0.7, 0.91, 0.9, "NDC")
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(30+3)
    pinfo.SetTextFont(42)
    string = 'Kolmogorov-Smirnov test'
    pinfo.AddText(string)
    string = f'Signal: statistic = {stat_sig:.3f}, p-value = {pval_sig:.2f} '
    pinfo.AddText(string)
    string = f'Background: statistic = {stat_bkg:.3f}, p-value = {pval_bkg:.2f} '
    pinfo.AddText(string)
    pinfo.Draw()
    cv.SetLogy()
    cv.SaveAs(bdt_score_dir + '/BDT_Score_' + suffix + info_string + '.png')

#save the correlation matricies of the features used for the training + mass for signal and background
def save_correlation_matricies(df_list, columns, suffix=''):
    fig_path = os.environ['FIGURES']
    bdt_eff_dir = fig_path

    labels = ["Signal", "Background"]
    
    for label, df in zip(labels, df_list):
        hist = ROOT.TH2D("corr_"+label, label+";;", len(columns), 0, len(columns), len(columns), 0, len(columns))
        corr = df.corr()
        cv = ROOT.TCanvas("cv", "cv",1600,1600)
        axis_x = hist.GetXaxis()
        axis_y = hist.GetYaxis()
        for col_x in columns:
            index_x = columns.index(col_x) + 1
            axis_x.SetBinLabel(index_x, col_x)
            axis_y.SetBinLabel(index_x, col_x)
            for col_y in columns:
                index_y = columns.index(col_y) + 1
                hist.SetBinContent(index_x, index_y, round(corr.at[col_x, col_y], 3))
        ROOT.gStyle.SetOptStat(0)
        hist.Draw("TEXT COLZ")
        cv.SaveAs(bdt_eff_dir + '/Corr_' + suffix + '_' + label + '.png')
    return

#save the efficiency vs BDT output selection computed on the test set
def save_bdt_efficiency(data, pt_range, suffix=''):
    fig_path = os.environ['FIGURES']
    info_string = f'_{pt_range[0]}{pt_range[1]}'
    bdt_eff_dir = fig_path + '/Efficiency'
    nbins = 1000
    hist_eff = TH1D("hist_eff", "; Score selection; BDT efficiency", nbins, data[2]["score"].min(), data[2]["score"].max())
    #hist_eff.SetMarkerStyle(20)
    df = data[2]
    df['true'] = data[3]
    df = df.query('true > 0.5')
    test_set_size = df.shape[0]
    delta_eff = -1
    err_delta_eff = -1
    for index in range(0, nbins):
        cut = hist_eff.GetXaxis().GetBinLowEdge(index + 1)
        df_tmp = df.query("score > @cut")
        tmp_size = df_tmp.shape[0]
        eff = tmp_size/test_set_size
        hist_eff.SetBinContent(index + 1, eff)
        hist_eff.SetBinError(index + 1, math.sqrt(eff*(1-eff)/test_set_size))
        if ROOT.TMath.Abs(eff-hist_eff.GetBinContent(index)) > delta_eff and index != 0:
            delta_eff = ROOT.TMath.Abs(eff-hist_eff.GetBinContent(index))
            err_delta_eff = ROOT.TMath.Sqrt(hist_eff.GetBinError(index+1)**2+hist_eff.GetBinError(index)**2)
    print("delta eff max: ",delta_eff," +- ",err_delta_eff)
    if not os.path.exists(bdt_eff_dir):
        os.makedirs(bdt_eff_dir)

    cv = TCanvas("cv","cv")
    gStyle.SetOptStat(0)
    hist_eff.Draw()
    cv.SaveAs(bdt_eff_dir + '/Eff_' + suffix + info_string + '.png')
    del df

#returns the integral of the pT distribution function between pt_min and pt_max
# dN/dpt = C*x*exp(-sqrt(x^2+mass^2)/T)
def get_pt_integral(pt_spectra, pt_min = 0, pt_max ="infinity"):
    mass = pt_spectra.GetParameter(1)
    T = pt_spectra.GetParameter(0)
    
    if pt_max == "infinity":
        int_max = 0
    else:
        t_max = ROOT.TMath.Sqrt(pt_max**2+mass**2)/T
        int_max = -T**2*(ROOT.TMath.Exp(-t_max)*(1+t_max))
    
    t_min = ROOT.TMath.Sqrt(pt_min**2+mass**2)/T
    int_min = -T**2*(ROOT.TMath.Exp(-t_min)*(1+t_min))
    return int_max - int_min