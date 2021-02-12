#!/usr/bin/env python3
import argparse
import math
import os
import time
import warnings

import analysis_utils as au
import plot_utils as pu
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import yaml
from analysis_classes import ModelApplication, TrainingAnalysis
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from ROOT import TFile, gROOT, TF1, TDatabasePDG, TH1D, TCanvas

# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-a', '--application', help='Apply ML predictions on data', action='store_true')
parser.add_argument('-c', '--custom', help='Run the custom significance optimisation studies', action='store_true')
parser.add_argument('-split', '--split', help='Run with matter and anti-matter splitted', action='store_true')

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
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

LARGE_DATA = params['LARGE_DATA']
LOAD_LARGE_DATA = params['LOAD_LARGE_DATA']

TRAIN = args.train
OPTIMIZE = args.optimize
APPLICATION = args.application
CUSTOM_SCAN = args.custom
SPLIT_MODE = args.split

if SPLIT_MODE:
    SPLIT_LIST = ['_matter','_antimatter']
else:
    SPLIT_LIST = ['']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
data_sig_path = os.path.expandvars(params['DATA_SIG_PATH'])
data_bkg_path = os.path.expandvars(params['DATA_BKG_PATH'])
event_path = os.path.expandvars(params['EVENT_PATH'])
results_dir = os.environ[f'HYPERML_RESULTS_{N_BODY}']

###############################################################################
start_time = time.time()                          # for performances evaluation
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
if TRAIN:
    for split in SPLIT_LIST:
        ml_analysis = TrainingAnalysis(PDG_CODE, N_BODY, signal_path, bkg_path, split, entrystop=1000000)
        print(f'--- analysis initialized in {((time.time() - start_time) / 60):.2f} minutes ---\n')

        for cclass in CENT_CLASSES:
            ml_analysis.preselection_efficiency(cclass, CT_BINS, PT_BINS, split, suffix = FILE_PREFIX)

            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                    print('\n==================================================')
                    print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)

                    part_time = time.time()

                    # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
                    data = ml_analysis.prepare_dataframe(COLUMNS, cent_class=cclass, ct_range=ctbin, pt_range=ptbin)

                    input_model = xgb.XGBClassifier(verbosity = 0)
                    model_handler = ModelHandler(input_model)
                    
                    model_handler.set_model_params(MODEL_PARAMS)
                    model_handler.set_model_params(HYPERPARAMS)
                    model_handler.set_training_columns(COLUMNS)

                    if OPTIMIZE:
                        model_handler.optimize_params_bayes(
                            data, HYPERPARAMS_RANGE, 'roc_auc', init_points=10, n_iter=10)

                    model_handler.train_test_model(data)
                    print("train test model")
                    print(f'--- model trained and tested in {((time.time() - part_time) / 60):.2f} minutes ---\n')

                    y_pred = model_handler.predict(data[2])
                    data[2].insert(0, 'score', y_pred)

                    eff, tsd = analysis_utils.bdt_efficiency_array(data[3], y_pred, n_points=1000)
                    score_from_eff_array = analysis_utils.score_from_efficiency_array(data[3], y_pred, FIX_EFF_ARRAY)
                    fixed_eff_array = np.vstack((FIX_EFF_ARRAY, score_from_eff_array))
                    
                    col = COLUMNS+['m']
                    SIG_DF = ml_analysis.df_signal[col]
                    BKG_DF = ml_analysis.df_bkg[col]
                    plot_utils.plot_roc(data[3], y_pred)                    
                    plt.savefig(f'../Figures/2Body/roc_curve_{FILE_PREFIX}.pdf')
                    plot_utils.plot_precision_recall(data[3], y_pred)
                    plt.savefig(f'../Figures/2Body/prescision_recall_{FILE_PREFIX}.pdf')
                    plot_utils.plot_distr([SIG_DF, BKG_DF], SIG_DF.columns)
                    plt.savefig(f'../Figures/2Body/plot_distr_{FILE_PREFIX}.pdf')
                    corr_plot = plot_utils.plot_corr([SIG_DF, BKG_DF], SIG_DF.columns)
                    corr_plot[0].savefig(f'../Figures/2Body/plot_corr_Sig_{FILE_PREFIX}.pdf')
                    corr_plot[1].savefig(f'../Figures/2Body/plot_corr_Bkg_{FILE_PREFIX}.pdf')

                    ml_analysis.save_ML_analysis(model_handler, fixed_eff_array, cent_class=cclass,pt_range=ptbin, ct_range=ctbin, split=split, suffix=FILE_PREFIX)
                    ml_analysis.save_ML_plots(model_handler, data, [eff, tsd],cent_class=cclass, pt_range=ptbin, ct_range=ctbin, split=split, suffix=FILE_PREFIX)

        del ml_analysis

    print('')
    print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')

if APPLICATION:
    app_time = time.time()

    file_name = results_dir + f'/{FILE_PREFIX}_results.root'
    results_histos_file = TFile(file_name, 'recreate')

    if (N_BODY==3):
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality','y']
    else:
        application_columns = ['score', 'm', 'ct', 'pt', 'centrality','y']

    sigscan_results = {}    

    for split in SPLIT_LIST:

        if LARGE_DATA:
            if LOAD_LARGE_DATA:
                df_skimmed = pd.read_parquet(os.path.dirname(data_sig_path) + f'/{FILE_PREFIX}skimmed_df.parquet.gzip')
            else:
                df_skimmed = au.get_skimmed_large_data(MULTIPLICITY, BRATIO, EFF, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, PT_BINS, CT_BINS, COLUMNS, application_columns, N_BODY, split, FILE_PREFIX)
                df_skimmed.to_parquet(os.path.dirname(data_sig_path) + f'/{FILE_PREFIX}skimmed_df.parquet.gzip', compression='gzip')

            ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split, df_skimmed)

        else:
            ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, EFF, N_BODY, data_sig_path, data_bkg_path, event_path, CENT_CLASSES, split)
        
        for cclass in CENT_CLASSES:
            # create output structure
            cent_dir_histos = results_histos_file.mkdir(f'{cclass[0]}-{cclass[1]}{split}')

            th2_efficiency = ml_application.load_preselection_efficiency(cclass, split, FILE_PREFIX)

            df_sign = pd.DataFrame()

            for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
                ptbin_index = ml_application.presel_histo.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))

                for ctbin in zip(CT_BINS[:-1], CT_BINS[1:]):
                    ctbin_index = ml_application.presel_histo.GetYaxis().FindBin(0.5 * (ctbin[0] + ctbin[1]))

                    print('\n==================================================')
                    print('centrality:', cclass, ' ct:', ctbin, ' pT:', ptbin, split)
                    print('Application and signal extraction ...', end='\r')

                    mass_bins = 40

                    presel_eff = ml_application.get_preselection_efficiency(ptbin_index, ctbin_index)
                    eff_score_array, model_handler = ml_application.load_ML_analysis(cclass, ptbin, ctbin, split, FILE_PREFIX)

                    if LARGE_DATA:
                        df_applied = ml_application.get_data_slice(cclass, ptbin, ctbin, application_columns)
                    else: 
                        df_applied = ml_application.apply_BDT_to_data(model_handler, cclass, ptbin, ctbin, model_handler.get_training_columns(), application_columns)

                    print("signal:",len(df_applied.query("y > 0.5").index))

                    pt_spectrum = TF1("fpt","x*exp(-TMath::Sqrt(x**2+[0]**2)/[1])",0,100)
                    pt_spectrum.SetParameter(0,mass)
                    pt_spectrum.SetParameter(1,T)

                    sigscan_eff, sigscan_tsd = ml_application.significance_scan(df_applied, presel_eff, eff_score_array, cclass, ptbin, ctbin, pt_spectrum, split, mass_bins, suffix=FILE_PREFIX, sigma_mass = SIGMA, custom=CUSTOM_SCAN)
                    eff_score_array = np.append(eff_score_array, [[sigscan_eff], [sigscan_tsd]], axis=1)

                    sigscan_results[f'ct{ctbin[0]}{ctbin[1]}pt{ptbin[0]}{ptbin[1]}{split}'] = [sigscan_eff, sigscan_tsd]

                    pt_bins = [0,0.5,1,1.5,2.,2.5,3]
                    hist_rec = ml_application.get_pt_hist(pt_bins)
                    if len(PT_BINS) == 2:
                        counts_eff, _ = np.histogram(df_applied.query("y > 0.5 and score > @sigscan_tsd")['pt'], len(pt_bins)-1, range=[pt_bins[0],pt_bins[len(pt_bins)-1]])
                        hist_eff = TH1D('hist_eff', ';#it{p}_{T} (GeV/c);BDT efficiency', len(pt_bins)-1, pt_bins[0], pt_bins[len(pt_bins)-1])

                        for index in range(0, len(pt_bins)-1):
                            eff = counts_eff[index]/hist_rec.GetBinContent(index+1)
                            if eff>1:
                                eff = 1
                            hist_eff.SetBinContent(index + 1, eff)
                            hist_eff.SetBinError(index + 1, math.sqrt(eff*(1-eff)/hist_rec.GetBinContent(index+1)))

                        cv = TCanvas("cv","cv")
                        hist_eff.Draw()
                        cv.SaveAs("eff_"+FILE_PREFIX+"_pt.png")
                        cv.SaveAs("eff_"+FILE_PREFIX+"_pt.pdf")

                    # define subdir for saving invariant mass histograms
                    sub_dir_histos = cent_dir_histos.mkdir(f'ct_{ctbin[0]}{ctbin[1]}') if 'ct' in FILE_PREFIX else cent_dir_histos.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')

                    for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
                        sub_dir_histos.cd()
                        
                        mass_array = np.array(df_applied.query('score>@tsd and m > @mass*0.96 and m < @mass*1.04')['m'].values, dtype=np.float64)

                        counts = np.histogram(mass_array, bins=mass_bins, range=[mass*0.96, mass*1.04])

                        histo_name = f'eff{eff:.2f}'
                        
                        h1_minv = au.h1_invmass(counts, cclass, ptbin, ctbin, name=histo_name)
                        h1_minv.Write()
                                
                    print('Application and signal extraction: Done!\n')

            cent_dir_histos.cd()
            th2_efficiency.Write()

    sigscan_results = np.asarray(sigscan_results)
    filename_sigscan = results_dir + f'/Efficiencies/{FILE_PREFIX}_sigscan.npy'
    np.save(filename_sigscan, sigscan_results)
    print (f'--- ML application time: {((time.time() - app_time) / 60):.2f} minutes ---')
    
    results_histos_file.Close()

    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')
