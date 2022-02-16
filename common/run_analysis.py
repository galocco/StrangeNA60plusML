#!/usr/bin/env python3
import argparse
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
from ROOT import TFile, gROOT, TF1, TDatabasePDG, gSystem
from scipy import stats
#import mplhep

#plt.style.use(mplhep.style.ALICE)
# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)
gROOT.SetBatch()

###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Do the training', action='store_true')
parser.add_argument('-o', '--optimize', help='Run the optimization', action='store_true')
parser.add_argument('-a', '--application', help='Apply ML predictions on data', action='store_true')
parser.add_argument('-c', '--custom', help='Run the custom significance optimisation studies', action='store_true')
parser.add_argument('-checks', '--checks', help='Perform further checks', action='store_true')

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
PDG_CODE = params['PDG']
FILE_PREFIX = params['FILE_PREFIX']
MULTIPLICITY = params['MULTIPLICITY']
BRATIO = params['BRATIO']
EINT = pu.get_sNN(params['EINT'])
T = params['T']
SIGMA = params['SIGMA']
MASS_WINDOW = params['MASS_WINDOW']
NBINS = params['NBINS']
PT_BINS = params['PT_BINS']
COLUMNS = params['TRAINING_COLUMNS']
MODEL_PARAMS = params['XGBOOST_PARAMS']
HYPERPARAMS = params['HYPERPARAMS']
HYPERPARAMS_RANGE = params['HYPERPARAMS_RANGE']

BKG_MODELS = params['BKG_MODELS']

EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
FIX_EFF_ARRAY = np.arange(EFF_MIN, EFF_MAX, EFF_STEP)

PRESELECTION = params['PRESELECTION']

TRAIN = args.train
OPTIMIZE = args.optimize
APPLICATION = args.application
CUSTOM_SCAN = args.custom
CHECKS = args.checks
NEVENTS = params['NEVENTS']

###############################################################################
# define paths for loading data
signal_path = os.path.expandvars(params['MC_PATH'])
bkg_path = os.path.expandvars(params['BKG_PATH'])
    
results_dir = os.environ['RESULTS']+"/"+FILE_PREFIX
if gSystem.AccessPathName(results_dir):
    gSystem.Exec('mkdir '+results_dir)

###############################################################################
start_time = time.time()                          # for performances evaluation
mass = TDatabasePDG.Instance().GetParticle(PDG_CODE).Mass()
if TRAIN:
    ml_analysis = TrainingAnalysis(PDG_CODE, signal_path, bkg_path, 5000000, PRESELECTION)
    print(f'--- analysis initialized in {((time.time() - start_time) / 60):.2f} minutes ---\n')

    ml_analysis.preselection_efficiency(PT_BINS, suffix = FILE_PREFIX)

    col = COLUMNS+['m']
    SIG_DF = ml_analysis.df_signal[col]
    BKG_DF = ml_analysis.df_bkg[col]

    au.save_correlation_matricies([SIG_DF, BKG_DF], SIG_DF.columns.to_list(), FILE_PREFIX)
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        print('\n==================================================')
        print(' pT:', ptbin)

        part_time = time.time()

        # data[0]=train_set, data[1]=y_train, data[2]=test_set, data[3]=y_test
        data = ml_analysis.prepare_dataframe(COLUMNS, pt_range=ptbin, test_size=0.2)

        input_model = xgb.XGBClassifier(verbosity = 0)
        model_handler = ModelHandler(input_model)
        
        model_handler.set_model_params(MODEL_PARAMS)
        model_handler.set_model_params(HYPERPARAMS)
        model_handler.set_training_columns(COLUMNS)

        if OPTIMIZE:
            model_handler.optimize_params_bayes(
                data, HYPERPARAMS_RANGE, 'roc_auc', init_points=10, n_iter=5)

        model_handler.train_test_model(data)
        print("train test model")
        print(f'--- model trained and tested in {((time.time() - part_time) / 60):.2f} minutes ---\n')

        y_pred = model_handler.predict(data[2])
        data[2].insert(0, 'score', y_pred)
        y_pred_test = model_handler.predict(data[0])
        data[0].insert(0, 'score', y_pred_test)

        eff, tsd = analysis_utils.bdt_efficiency_array(data[3], y_pred, n_points=1000)
        score_from_eff_array = analysis_utils.score_from_efficiency_array(data[1], y_pred_test, FIX_EFF_ARRAY)
        fixed_eff_array = np.vstack((FIX_EFF_ARRAY, score_from_eff_array))
        
        plot_utils.plot_roc(data[3], y_pred)                    
        plt.savefig(f'../Figures/roc_curve_{FILE_PREFIX}.pdf')
        plot_utils.plot_precision_recall(data[3], y_pred)
        plt.savefig(f'../Figures/prescision_recall_{FILE_PREFIX}.pdf')
        plot_utils.plot_distr([SIG_DF, BKG_DF], SIG_DF.columns)
        plt.savefig(f'../Figures/plot_distr_{FILE_PREFIX}.pdf')
        corr_plot = plot_utils.plot_corr([SIG_DF, BKG_DF], SIG_DF.columns)
        corr_plot[0].savefig(f'../Figures/plot_corr_Sig_{FILE_PREFIX}.pdf')
        corr_plot[1].savefig(f'../Figures/plot_corr_Bkg_{FILE_PREFIX}.pdf')

        ml_analysis.save_ML_analysis(model_handler, fixed_eff_array, pt_range=ptbin, suffix=FILE_PREFIX)
        ml_analysis.save_feature_importance(model_handler, data, pt_range=ptbin, suffix=FILE_PREFIX)
        au.save_bdt_efficiency(data, ptbin, FILE_PREFIX)
        au.save_bdt_output_plot(data, ptbin, FILE_PREFIX)
    del ml_analysis

    print('')
    print(f'--- training and testing in {((time.time() - start_time) / 60):.2f} minutes ---')

if APPLICATION:
    app_time = time.time()

    data_path = os.path.expandvars(params['DATA_PATH'])

    file_name = results_dir + f'/{FILE_PREFIX}_results.root'
    results_histos_file = TFile(file_name, 'recreate')

    sigscan_results = {}    

    hnsparse = au.get_skimmed_large_data_hsp(mass, data_path, PT_BINS, COLUMNS, FILE_PREFIX, PRESELECTION, MASS_WINDOW, NBINS, CHECKS)
    hnsparse.Write()    
    ml_application = ModelApplication(PDG_CODE, MULTIPLICITY, BRATIO, NEVENTS, hnsparse)
    del hnsparse

    pt_spectrum = TF1("fpt"," x*exp(-TMath::Sqrt(x**2+[0]**2)/[1])", 0, 100)
    pt_spectrum.FixParameter(0, mass)
    pt_spectrum.FixParameter(1, T)

    th1_efficiency = ml_application.load_preselection_efficiency(FILE_PREFIX)
    for ptbin in zip(PT_BINS[:-1], PT_BINS[1:]):
        # define subdir for saving invariant mass histograms
        sub_dir_histos = results_histos_file.mkdir(f'pt_{ptbin[0]}{ptbin[1]}')
        ptbin_index = ml_application.presel_histo.GetXaxis().FindBin(0.5 * (ptbin[0] + ptbin[1]))
        print('\n==================================================')
        print('pT:', ptbin)
        print('Application and signal extraction ...', end='\r')

        presel_eff = ml_application.get_preselection_efficiency(ptbin_index)
        eff_score_array, model_handler = ml_application.load_ML_analysis(ptbin, FILE_PREFIX)
        
        if CHECKS:
            results_histos_file.cd()
            bdt_eff_sparse = au.bdt_efficiency_from_sparse(ml_application.hnsparse, ptbin, f'bdt_eff_{ptbin[0]}_{ptbin[1]}_sparse')
            bdt_eff_sparse.Write()
            bdt_eff_train = au.bdt_efficiency_train(eff_score_array,f'bdt_eff_{ptbin[0]}_{ptbin[1]}_train')
            bdt_eff_train.Write()

        sub_dir_histos.cd()
        sigscan_eff, sigscan_tsd = ml_application.significance_scan(presel_eff, eff_score_array, ptbin, pt_spectrum, suffix=FILE_PREFIX, sigma_mass = SIGMA, custom=CUSTOM_SCAN, mass_range=MASS_WINDOW)
        eff_score_array = np.append(eff_score_array, [[sigscan_eff], [sigscan_tsd]], axis=1)

        for eff, tsd in zip(pd.unique(eff_score_array[0][::-1]), pd.unique(eff_score_array[1][::-1])):
            histo_name = f'eff{eff:.3f}'
            h1_minv = au.h1_from_sparse(ml_application.hnsparse, ptbin, tsd, name=histo_name)
            h1_minv.Write()
            del h1_minv

        print('Application and signal extraction: Done!\n')

    results_histos_file.cd()
    th1_efficiency.Write()
    print (f'--- ML application time: {((time.time() - app_time) / 60):.2f} minutes ---')
    
    results_histos_file.Close()

    print(f'--- analysis time: {((time.time() - start_time) / 60):.2f} minutes ---')

