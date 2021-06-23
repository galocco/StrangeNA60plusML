import math
import os
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10
import warnings
import aghast
import numpy as np
import pandas as pd
import ROOT
import uproot
import xgboost as xgb
from hipe4ml.model_handler import ModelHandler
from ROOT import TF1, TH1D, TH2D, TH3D, TCanvas, TPaveStats, TPaveText, gStyle, TDatabasePDG, THnSparseD
import re
import xml.etree.cElementTree as ET
from sklearn.utils import shuffle
from array import array
# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def build_tree(xgtree, base_xml_element, var_indices):
    regex_float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    parent_element_dict = {'0':base_xml_element}
    pos_dict = {'0':'s'}
    for line in xgtree.split('\n'):
        if not line: continue
        if ':leaf=' in line:
            #leaf node
            result = re.match(r'(\t*)(\d+):leaf=({0})$'.format(regex_float_pattern), line)
            if not result:
                print(line)
            depth = result.group(1).count('\t')
            inode = result.group(2)
            res = result.group(3)
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar="-1", Cut="0.0e+00", cType="1", res=str(res), rms="0.0e+00", purity="0.0e+00", nType="-99")
        else:
            #\t\t3:[var_topcand_mass<138.19] yes=7,no=8,missing=7
            result = re.match(r'(\t*)([0-9]+):\[(?P<var>.+)<(?P<cut>{0})\]\syes=(?P<yes>\d+),no=(?P<no>\d+)'.format(regex_float_pattern),line)
            if not result:
                print(line)
            depth = result.group(1).count('\t')
            inode = result.group(2)
            var = result.group('var')
            cut = result.group('cut')
            lnode = result.group('yes')
            rnode = result.group('no')
            pos_dict[lnode] = 'l'
            pos_dict[rnode] = 'r'
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar=str(var_indices[var]), Cut=str(cut),
                                             cType="1", res="0.0e+00", rms="0.0e+00", purity="0.0e+00", nType="0")
            parent_element_dict[lnode] = node_elementTree
            parent_element_dict[rnode] = node_elementTree
            
def convert_model(model, input_variables, output_xml):
    NTrees = len(model)
    var_list = input_variables
    var_indices = {}
    
    # <MethodSetup>
    MethodSetup = ET.Element("MethodSetup", Method="BDT::BDT")

    # <Variables>
    Variables = ET.SubElement(MethodSetup, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        name = val[0]
        var_type = val[1]
        var_indices[name] = ind
        Variable = ET.SubElement(Variables, "Variable", VarIndex=str(ind), Type=val[1], 
            Expression=name, Label=name, Title=name, Unit="", Internal=name, 
            Min="0.0e+00", Max="0.0e+00")

    # <GeneralInfo>
    GeneralInfo = ET.SubElement(MethodSetup, "GeneralInfo")
    Info_Creator = ET.SubElement(GeneralInfo, "Info", name="Creator", value="xgboost2TMVA")
    Info_AnalysisType = ET.SubElement(GeneralInfo, "Info", name="AnalysisType", value="Classification")

    # <Options>
    Options = ET.SubElement(MethodSetup, "Options")
    Option_NodePurityLimit = ET.SubElement(Options, "Option", name="NodePurityLimit", modified="No").text = "5.00e-01"
    Option_BoostType = ET.SubElement(Options, "Option", name="BoostType", modified="Yes").text = "Grad"
    
    # <Weights>
    Weights = ET.SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")
    
    for itree in range(NTrees):
        BinaryTree = ET.SubElement(Weights, "BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(itree))
        build_tree(model[itree], BinaryTree, var_indices)
        
    tree = ET.ElementTree(MethodSetup)
    tree.write(output_xml)

def get_skimmed_large_data(mass, multiplicity, bratio, eff, sig_path, bkg_path, event_path, pt_bins, training_columns, mode, split='', suffix='', preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 30, 2000])
    xmin  = array('d', [mass*(1-range), 0., -20])
    xmax  = array('d', [mass*(1+range), 3.,  20])
    hsparse = THnSparseD('sparse_m_pt_s', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});score;counts', 3, nbins, xmin, xmax)

    if mode == 3:
        handlers_path = "../Models/3Body/handlers"
        efficiencies_path = "../Results/3Body/Efficiencies"

    if mode == 2:
        handlers_path = "../Models/2Body/handlers"
        efficiencies_path = "../Results/2Body/Efficiencies"

    background_file = ROOT.TFile(event_path, "read")
    hist_ev = background_file.Get('hNevents')
    n_ev = hist_ev.GetBinContent(1)
    nsig = int(multiplicity*eff*n_ev*bratio)
    background_file.Close()
    executor = ThreadPoolExecutor()
    bkg_tree_name = bkg_path + ":/ntcand"
    bkg_iterator = uproot.iterate(bkg_tree_name, executor=executor, library="pd")
    
    for data in bkg_iterator:
        rename_df_columns(data)
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            info_string = '_{}{}'.format(ptbin[0], ptbin[1])

            filename_handler = handlers_path + '/model_handler_' +suffix+ info_string + split + '.pkl'
            filename_efficiencies = efficiencies_path + '/Eff_Score_' +suffix+ info_string + split + '.npy'

            model_handler = ModelHandler()
            model_handler.load_model_handler(filename_handler)

            eff_score_array = np.load(filename_efficiencies)
            tsd = eff_score_array[1][-1]

            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'

            df_tmp = data.query(data_range + 'and ' + preselection)       
            df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))
            df_tmp = df_tmp.query('score>@tsd')
            
            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind]])
                hsparse.Fill(x)
    
    executor = ThreadPoolExecutor()
    sig_tree_name = sig_path + ":/ntcand"
    sig_iterator = uproot.iterate(sig_tree_name, executor=executor, library="pd")

    counter = 0
    for data in sig_iterator:
        rename_df_columns(data)
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            info_string = '_{}{}'.format(ptbin[0], ptbin[1])

            filename_handler = handlers_path + '/model_handler_' +suffix+ info_string + split + '.pkl'
            filename_efficiencies = efficiencies_path + '/Eff_Score_' +suffix+ info_string + split + '.npy'

            model_handler = ModelHandler()
            model_handler.load_model_handler(filename_handler)

            eff_score_array = np.load(filename_efficiencies)
            tsd = eff_score_array[1][-1]

            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'

            df_tmp = data.query(data_range + ' and ' + preselection)

            df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))
            df_tmp = df_tmp.query('score>@tsd')

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind]])
                hsparse.Fill(x)
                
                counter += 1
                if counter==nsig:
                    return hsparse

    return hsparse    

def get_skimmed_large_data_std(mass, multiplicity, bratio, eff, sig_path, bkg_path, event_path, pt_bins, training_columns, mode, split='', suffix='', preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 30])
    xmin  = array('d', [mass*(1-range), 0.])
    xmax  = array('d', [mass*(1+range), 3.])
    hsparse = THnSparseD('sparse_m_pt', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});counts', 2, nbins, xmin, xmax)

    if mode == 3:
        handlers_path = "../Models/3Body/handlers"
        efficiencies_path = "../Results/3Body/Efficiencies"

    if mode == 2:
        handlers_path = "../Models/2Body/handlers"
        efficiencies_path = "../Results/2Body/Efficiencies"

    background_file = ROOT.TFile(event_path, "read")
    hist_ev = background_file.Get('hNevents')
    n_ev = hist_ev.GetBinContent(1)
    nsig = int(multiplicity*eff*n_ev*bratio)
    background_file.Close()
    executor = ThreadPoolExecutor()
    bkg_tree_name = bkg_path + ":/ntcand"
    bkg_iterator = uproot.iterate(bkg_tree_name, executor=executor, library="pd")
    
    for data in bkg_iterator:
        rename_df_columns(data)
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'

            df_tmp = data.query(data_range + 'and ' + preselection)
            
            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind]])
                hsparse.Fill(x)
    
    executor = ThreadPoolExecutor()
    sig_tree_name = sig_path + ":/ntcand"
    sig_iterator = uproot.iterate(sig_tree_name, executor=executor, library="pd")

    counter = 0
    for data in sig_iterator:
        rename_df_columns(data)
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):

            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'

            df_tmp = data.query(data_range + ' and ' + preselection)

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind]])
                hsparse.Fill(x)
                
                counter += 1
                if counter==nsig:
                    return hsparse

    return hsparse    

def get_skimmed_large_data_full(mass, data_path, pt_bins, training_columns, mode, split='', suffix='', preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 30, 2000])
    xmin  = array('d', [mass*(1-range), 0., -20])
    xmax  = array('d', [mass*(1+range), 3.,  20])
    hsparse = THnSparseD('sparse_m_pt_s', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});score;counts', 3, nbins, xmin, xmax)

    if mode == 3:
        handlers_path = "../Models/3Body/handlers"
        efficiencies_path = "../Results/3Body/Efficiencies"

    if mode == 2:
        handlers_path = "../Models/2Body/handlers"
        efficiencies_path = "../Results/2Body/Efficiencies"

    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    df_applied = pd.DataFrame()

    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            info_string = f'_{ptbin[0]}{ptbin[1]}{split}'

            filename_handler = handlers_path + '/model_handler_' +suffix+ info_string + '.pkl'
            filename_efficiencies = efficiencies_path + '/Eff_Score_' + suffix + info_string + '.npy'

            model_handler = ModelHandler()
            model_handler.load_model_handler(filename_handler)

            eff_score_array = np.load(filename_efficiencies)
            tsd = eff_score_array[1][-1]

            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'
            df_tmp = data.query(data_range+" and "+preselection)
            df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))
            df_tmp = df_tmp.query('score>@tsd')

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind]])
                hsparse.Fill(x)

    return hsparse


def get_skimmed_large_data_std_full(mass, data_path, pt_bins, training_columns, mode, split='', suffix='', preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 30])
    xmin  = array('d', [mass*(1-range), 0.])
    xmax  = array('d', [mass*(1+range), 3.])
    hsparse = THnSparseD('sparse_m_pt', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});counts', 2, nbins, xmin, xmax)

    if mode == 3:
        handlers_path = "../Models/3Body/handlers"
        efficiencies_path = "../Results/3Body/Efficiencies"

    if mode == 2:
        handlers_path = "../Models/2Body/handlers"
        efficiencies_path = "../Results/2Body/Efficiencies"

    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    df_applied = pd.DataFrame()

    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'
            df_tmp = data.query(data_range+" and "+preselection)

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind]])
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
    hnsparse_clone = hnsparse.Clone()
    ptbin_min = hnsparse_clone.GetAxis(1).FindBin(pt_range[0])
    ptbin_max = hnsparse_clone.GetAxis(1).FindBin(pt_range[1])
    if ptbin_max > hnsparse_clone.GetAxis(1).GetNbins():
        ptbin_max = hnsparse_clone.GetAxis(1).GetNbins()
    scorebin_min = hnsparse_clone.GetAxis(2).FindBin(score)
    scorebin_max = hnsparse_clone.GetAxis(2).GetNbins()
    if scorebin_min > scorebin_max:
        scorebin_min = scorebin_max
    hnsparse_clone.GetAxis(1).SetRange(ptbin_min, ptbin_max)
    hnsparse_clone.GetAxis(2).SetRange(scorebin_min, scorebin_max)
    th1 = hnsparse_clone.Projection(0)
    th1.SetName(name)
    th1.SetDirectory(0)
    return th1

def h1_from_sparse_std(hnsparse, pt_range, name=''):
    hnsparse_clone = hnsparse.Clone()
    ptbin_min = hnsparse_clone.GetAxis(1).FindBin(pt_range[0])
    ptbin_max = hnsparse_clone.GetAxis(1).FindBin(pt_range[1])
    if ptbin_max > hnsparse_clone.GetAxis(1).GetNbins():
        ptbin_max = hnsparse_clone.GetAxis(1).GetNbins()
    hnsparse_clone.GetAxis(1).SetRange(ptbin_min, ptbin_max)
    th1 = hnsparse_clone.Projection(0)
    th1.SetName(name)
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
        histo, pt_range, mass, nsigma=3, model="pol2", fixsigma=-1, sigma_limits=None, mode=3, split ='', Eint=17.3, peak_mode=True, gauss=True):
    
    #mass != TDatabasePDG.Instance().GetParticle(333).Mass()
    
    hist_range = [mass*0.97,mass*1.03]
    # canvas for plotting the invariant mass distribution
    cv = TCanvas(f'cv_{histo.GetName()}')

    # define the number of parameters depending on the bkg model
    if 'pol' in str(model):
        n_bkgpars = int(model[3]) + 1
    elif 'expo' in str(model):
        n_bkgpars = 2
    else:
        print(f'Unsupported model {model}')

    # define the fit function bkg_model + gauss/voigt
    if gauss:
        fit_tpl = TF1('fitTpl', f'{model}(0)+gausn({n_bkgpars})', 0, 5)
    else:
        fit_tpl = TF1('fitTpl', f'{model}(0)+TMath::Voigt(x-[{n_bkgpars+1}],[{n_bkgpars+2}],[{n_bkgpars+3}])*[{n_bkgpars}]', 0, 5)

    # redefine parameter names for the bkg_model
    for i in range(n_bkgpars):
        fit_tpl.SetParName(i, f'B_{i}')

    # define parameter names for the signal fit
    fit_tpl.SetParName(n_bkgpars, 'N_{sig}')
    fit_tpl.SetParName(n_bkgpars + 1, '#mu')
    fit_tpl.SetParName(n_bkgpars + 2, '#sigma')
    if not gauss:
        fit_tpl.SetParName(n_bkgpars + 3, '#Gamma')

    max_hist_value = histo.GetMaximum()
    hist_bkg_eval = (histo.GetBinContent(1)+histo.GetBinContent(histo.GetNbinsX()))/2.
    if hist_bkg_eval < 5:
        hist_bkg_eval = 5
    
    #if model=='pol2':
    #    fit_tpl.SetParameter(0, -5400/2)
    #    fit_tpl.SetParameter(1, 29)#94.76)
    #    fit_tpl.SetParameter(2, 5575/2)

    fit_tpl.SetParameter(n_bkgpars, max_hist_value-hist_bkg_eval)
    fit_tpl.SetParLimits(n_bkgpars, 0, max_hist_value+3*hist_bkg_eval)
    fit_tpl.SetParameter(n_bkgpars + 1, mass)
    fit_tpl.SetParLimits(n_bkgpars + 1, mass-0.005, mass+0.005)
    fit_tpl.SetParameter(n_bkgpars + 2, 0.0035)
    fit_tpl.SetParLimits(n_bkgpars + 2, 0.001, 0.006)
    if not gauss:
        fit_tpl.SetParameter(n_bkgpars + 3, 0.00426)
        fit_tpl.SetParLimits(n_bkgpars + 3, 0.000001, 0.005)

    # define signal and bkg_model TF1 separately
    if gauss:
        sigTpl = TF1('fitTpl','gausn(0)', 0, 5)
    else:
        sigTpl = TF1('fitTpl','TMath::Voigt(x-[1],[2],[3])*[0]', 0, 5)

    bkg_tpl = TF1('fitTpl', f'{model}(0)', 0, 5)

    # plotting stuff for fit_tpl
    fit_tpl.SetNpx(300)
    fit_tpl.SetLineWidth(2)
    fit_tpl.SetLineColor(2)
    # plotting stuff for bkg model
    bkg_tpl.SetNpx(300)
    bkg_tpl.SetLineWidth(2)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.SetLineColor(2)

    # define limits for the sigma if provided
    if sigma_limits != None:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.5 *
                             (sigma_limits[0] + sigma_limits[1]))
        fit_tpl.SetParLimits(n_bkgpars + 2, sigma_limits[0], sigma_limits[1])
    # if the mc sigma is provided set the sigma to that value
    elif fixsigma > 0:
        fit_tpl.FixParameter(n_bkgpars + 2, fixsigma)
    # otherwise set sigma limits reasonably
    else:
        fit_tpl.SetParameter(n_bkgpars + 2, 0.0025)
        fit_tpl.SetParLimits(n_bkgpars + 2, 0.0005, 0.0035)

    ########################################
    # plotting the fits
    ax_titles = ';m (GeV/#it{c})^{2};Counts' + f' / {round(1000 * histo.GetBinWidth(1), 2)} MeV'

    # invariant mass distribution histo and fit
    histo.UseCurrentStyle()
    histo.SetLineColor(1)
    histo.SetMarkerStyle(20)
    histo.SetMarkerColor(1)
    histo.SetTitle(ax_titles)
    histo.SetMaximum(1.5 * histo.GetMaximum())
    histo.Fit(fit_tpl, "QRL", "", hist_range[0], hist_range[1])
    histo.SetDrawOption("e")
    histo.GetXaxis().SetRangeUser(hist_range[0], hist_range[1])
    # represent the bkg_model separately
    bkg_tpl.SetParameters(fit_tpl.GetParameters())
    bkg_tpl.SetLineColor(600)
    bkg_tpl.SetLineStyle(2)
    bkg_tpl.Draw("same")
    # represent the signal model separately
    sigTpl.SetParameter(0, fit_tpl.GetParameter(n_bkgpars))
    sigTpl.SetParameter(1, fit_tpl.GetParameter(n_bkgpars+1))
    sigTpl.SetParameter(2, fit_tpl.GetParameter(n_bkgpars+2))
    if not gauss:
        sigTpl.SetParameter(3, fit_tpl.GetParameter(n_bkgpars+3))
    sigTpl.SetLineColor(600)
    # sigTpl.Draw("same")

    # get the fit parameters
    mu = fit_tpl.GetParameter(n_bkgpars+1)
    muErr = fit_tpl.GetParError(n_bkgpars+1)
    sigma = fit_tpl.GetParameter(n_bkgpars+2)
    sigmaErr = fit_tpl.GetParError(n_bkgpars+2)
    bkg = bkg_tpl.Integral(mu - nsigma * sigma, mu +
                           nsigma * sigma) / histo.GetBinWidth(1)
    if peak_mode:
        signal = fit_tpl.GetParameter(n_bkgpars) / histo.GetBinWidth(1)
        errsignal = fit_tpl.GetParError(n_bkgpars) / histo.GetBinWidth(1)
    else:
        signal = 0
        bin_min = histo.GetXaxis().FindBin(mu - nsigma * sigma)
        bin_max = histo.GetXaxis().FindBin(mu + nsigma * sigma)

        for bin in range(bin_min,bin_max+1):
            signal += histo.GetBinContent(bin)
        signal -= bkg
        errsignal = math.sqrt(signal+bkg)
        

    if bkg > 0:
        errbkg = math.sqrt(bkg)
    else:
        errbkg = 0
    # compute the significance
    if signal+bkg > 0:
        signif = signal/math.sqrt(signal+bkg)
        deriv_sig = 1/math.sqrt(signal+bkg)-signif/(2*(signal+bkg))
        deriv_bkg = -signal/(2*(math.pow(signal+bkg, 1.5)))
        errsignif = math.sqrt((errsignal*deriv_sig)**2+(errbkg*deriv_bkg)**2)
    else:
        signif = 0
        errsignif = 0

    # print fit info on the canvas
    pinfo2 = TPaveText(0.5, 0.5, 0.91, 0.9, "NDC")
    pinfo2.SetBorderSize(0)
    pinfo2.SetFillStyle(0)
    pinfo2.SetTextAlign(30+3)
    pinfo2.SetTextFont(42)

    string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{Eint} GeV, centrality {0}-{5}%'
    pinfo2.AddText(string)

    string = f'{pt_range[0]:.3f}'+' #leq #it{p}_{T} < '+f'{pt_range[1]:.3f}'+' GeV/#it{c} '
    pinfo2.AddText(string)

    string = f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {errsignif:.1f} '
    pinfo2.AddText(string)

    string = f'S ({nsigma:.0f}#sigma) {signal:.0f} #pm {errsignal:.0f}'
    pinfo2.AddText(string)

    string = f'B ({nsigma:.0f}#sigma) {bkg:.0f} #pm {errbkg:.0f}'
    pinfo2.AddText(string)

    if bkg > 0:
        ratio = signal/bkg
        string = f'S/B ({nsigma:.0f}#sigma) {ratio:.4f}'

    pinfo2.AddText(string)
    pinfo2.Draw()
    gStyle.SetOptStat(0)

    st = histo.FindObject('stats')
    if isinstance(st, TPaveStats):
        st.SetX1NDC(0.12)
        st.SetY1NDC(0.62)
        st.SetX2NDC(0.40)
        st.SetY2NDC(0.90)
        st.SetOptStat(0)

    histo.Write()
    cv.Write()

    return (signal, errsignal, signif, errsignif, mu, muErr, sigma, sigmaErr)
    return (signal, errsignal, signif, errsignif, sigma, sigmaErr)


def load_mcsigma(pt_range, mode, split=''):
    info_string = f'_{pt_range[0]}{pt_range[1]}{split}'
    sig_path = os.environ['HYPERML_UTILS_{}'.format(mode)] + '/FixedSigma'

    file_name = f'{sig_path}/sigma_array{info_string}.npy'

    return np.load(file_name, allow_pickle=True)


def rename_df_columns(df):
    rename_dict = {}

    for col in df.columns:

        if col.endswith('_f'):
            rename_dict[col] = col[:-2]
    
    df.rename(columns = rename_dict, inplace=True)


def ndarray2roo(ndarray, var):
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1d array'

    name = var.GetName()
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(f'{name}', x ,f'{name}/D')

    for i in ndarray:
        x[0] = i
        tree.Fill()

    array_roo = ROOT.RooDataSet('data', 'dataset from tree', tree, ROOT.RooArgSet(var))
    return array_roo

#TODO: fix this function
def unbinned_mass_fit(data, eff, bkg_model, output_dir, cent_class, pt_range, ct_range, split):
    output_dir.cd()
    # define working variable 
    mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2.975, 3.010, 'GeV/c^{2}')

    # define signal parameters
    hyp_mass = ROOT.RooRealVar('hyp_mass', 'hypertriton mass', 2.989, 2.993, 'GeV/c^{2}')
    width = ROOT.RooRealVar('width', 'hypertriton width', 0.0001, 0.004, 'GeV/c^{2}')

    # define signal component
    signal = ROOT.RooGaussian('signal', 'signal component pdf', mass, hyp_mass, width)

    # define background parameters
    slope = ROOT.RooRealVar('slope', 'exponential slope', -100., 100)

    c0 = ROOT.RooRealVar('c0', 'constant c0', -100., 100.)
    c1 = ROOT.RooRealVar('c1', 'constant c1', -100., 100.)
    c2 = ROOT.RooRealVar('c2', 'constant c2', -100., 100.)

    # define background component depending on background model required
    if bkg_model == 'pol1':
        background = ROOT.RooPolynomial('bkg', 'pol1 bkg', mass, ROOT.RooArgList(c0, c1))
                                        
    if bkg_model == 'pol2':
        background = ROOT.RooPolynomial('bkg', 'pol2 for bkg', mass, ROOT.RooArgList(c0, c1, c2))
        
    if bkg_model == 'expo':
        background = ROOT.RooExponential('bkg', 'expo for bkg', mass, slope)

    # define fraction
    n1 = ROOT.RooRealVar('n1', 'n1 const', 0., 1, 'GeV')

    # define the fit funciton -> signal component + background component
    fit_function = ROOT.RooAddPdf(f'{bkg_model}_gaus', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n1))

    # convert data to RooData               
    roo_data = ndarray2roo(data, mass)

    # fit data
    fit_function.fitTo(roo_data, ROOT.RooFit.Range(2.975, 3.01))

    # plot the fit
    frame = mass.frame(35)

    roo_data.plotOn(frame)
    fit_function.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue))
    fit_function.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed))
    fit_function.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))

    # add info to plot
    nsigma = 3
    mu = hyp_mass.getVal()
    mu_error = hyp_mass.getError()
    sigma = width.getVal()
    sigma_error = width.getError()

    # # getting the chi2 by binning the data
    # hist_tmp = super(roo_data.__class__, roo_data).createHistogram('hist_tmp', mass, ROOT.RooFit.Binning(35))
    # fit_function.chiSquare(mass, histogram)

    # print(type(hist_tmp))
    # exit()

    # # compute significance
    # mass.setRange('signal region',  mu - (nsigma * sigma), mu + (nsigma * sigma))
    # signal_counts = int(round(signal.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal region').getVal() * n1.getVal()*normalization))
    # background_counts = int(round(background.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal region').getVal() * (1 - n1.getVal())*normalization))

    # signif = signal_counts / math.sqrt(signal_counts + background_counts + 1e-10)
    # signif_error = significance_error(signal_counts, background_counts)

    pinfo = ROOT.TPaveText(0.537, 0.474, 0.937, 0.875, 'NDC')
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(30+3)
    pinfo.SetTextFont(42)
    # pinfo.SetTextSize(12)

    decay_label = {
        '': '{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-} + c.c.',
        '_matter': '{}^{3}_{#Lambda}H#rightarrow ^{3}He#pi^{-}',
        '_antimatter': '{}^{3}_{#bar{#Lambda}}#bar{H}#rightarrow ^{3}#bar{He}#pi^{+}',
    }

    string_list = []

    string_list.append(f'ALICE Internal, Pb-Pb 2018 {cent_class[0]}-{cent_class[1]}%')
    string_list.append(decay_label[split] + ', %i #leq #it{p}_{T} < %i GeV/#it{c} ' % (pt_range[0], pt_range[1]))
    string_list.append(f'#mu {mu*1000:.2f} #pm {mu_error*1000:.2f} MeV/c^{2}')
    string_list.append(f'#sigma {sigma*1000:.2f} #pm {sigma_error*1000:.2f} MeV/c^{2}')

    if roo_data.sumEntries()>0:
        string_list.append('#chi^{2} / NDF ' + f'{frame.chiSquare(6 if bkg_model=="pol2" else 5):.2f}')

    # string_list.append(f'Significance ({nsigma:.0f}#sigma) {signif:.1f} #pm {signif_error:.1f}')
    # string_list.append(f'S ({nsigma:.0f}#sigma) {signal_counts} #pm {int(round(math.sqrt(signal_counts)))}')
    # string_list.append(f'B ({nsigma:.0f}#sigma) {background_counts} #pm {int(round(math.sqrt(signal_counts)))}')

    # if background_counts > 0:
    #     ratio = signal_counts / background_counts
    #     string_list.append(f'S/B ({nsigma:.0f}#sigma) {ratio:.2f}')

    for s in string_list:
        pinfo.AddText(s)

    frame.addObject(pinfo)

    sub_dir_name = f'pT{pt_range[0]}{pt_range[1]}_eff{eff:.2f}{split}'
    sub_dir = output_dir.GetDirectory(sub_dir_name)

    if not sub_dir:
        sub_dir = output_dir.mkdir(f'pT{pt_range[0]}{pt_range[1]}_eff{eff:.2f}{split}')

    sub_dir.cd()

    frame.Write(f'frame_model_{bkg_model}')
    hyp_mass.Write(f'hyp_mass_model{bkg_model}')
    width.Write(f'width_model{bkg_model}')

def pt_array_to_mt_m0_array(pt_array, mass):
    mt_array = []
    for pt_item in pt_array:
        mt_array.append(math.sqrt(pt_item**2+mass**2)-mass)
    return mt_array
