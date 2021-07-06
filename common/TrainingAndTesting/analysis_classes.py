# this class has been created to generalize the training and to open the file.root just one time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
import uproot
import xgboost as xgb
from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler
from sklearn.model_selection import train_test_split
import math
import analysis_utils as au
import plot_utils as pu
from array import array

class TrainingAnalysis:

    def __init__(self, pdg_code, mc_file_name, bkg_file_name, entrystop=10000000, preselection=''):
        self.mass = ROOT.TDatabasePDG.Instance().GetParticle(pdg_code).Mass()

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT training and testing ')
        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        self.df_signal = uproot.open(mc_file_name)['ntcand'].arrays(library='pd').query(preselection)
        preselection = ' and ' + preselection
        self.df_generated = uproot.open(mc_file_name)['ntgen'].arrays(library='pd')
        self.df_bkg = uproot.open(bkg_file_name)['ntcand'].arrays(library='pd',entry_stop=entrystop).query("true < 0.5" + preselection)

        self.df_signal['y'] = 1
        self.df_bkg['y'] = 0

    def preselection_efficiency(self, pt_bins, save=True, suffix=''):
        cut  =  f'{pt_bins[0]}<=pt<={pt_bins[len(pt_bins)]}'         
            
        pres_histo = au.h1_preselection_efficiency(pt_bins)
        gen_histo = au.h1_generated(pt_bins)

        for pt in self.df_signal.query(cut)['pt']:#.to_records(index=False)
            pres_histo.Fill(pt)
        
        for pt in self.df_generated['pt']:#.to_records(index=False)
            gen_histo.Fill(pt)

        pres_histo.Divide(gen_histo)

        if save:
            path = os.environ['HYPERML_EFFICIENCIES']

            filename = path + f'/PreselEff{suffix}.root'
            t_file = ROOT.TFile(filename, 'recreate')
            
            pres_histo.Write()
            t_file.Close()

        return pres_histo

    def prepare_dataframe(self, training_columns, pt_range, test_size=0.5):
        data_range = f'{pt_range[0]}<pt<{pt_range[1]}'#' and {cent_class[0]}<=centrality<{cent_class[1]}'

        sig = self.df_signal.query(data_range)
        bkg = self.df_bkg.query(data_range)

        if (len(bkg) >= 10*len(sig)):
            bkg = bkg.sample(n=10*len(sig))

        print('\nNumber of signal candidates: {}'.format(len(sig)))
        print('Number of background candidates: {}\n'.format(len(bkg)))

        df = pd.concat([self.df_signal.query(data_range), self.df_bkg.query(data_range)])

        train_set, test_set, y_train, y_test = train_test_split(df[training_columns + ['m']], df['y'], test_size=test_size, random_state=42)

        return [train_set, y_train, test_set, y_test]

    def save_ML_analysis(self, model_handler, eff_score_array, pt_range, suffix=''):
        info_string = f'_{pt_range[0]}{pt_range[1]}'

        models_path = os.environ['HYPERML_MODELS']+'/models'
        handlers_path = os.environ['HYPERML_MODELS']+'/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES']

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        if not os.path.exists(handlers_path):
            os.makedirs(handlers_path)

        filename_handler = handlers_path + '/model_handler_' + suffix + info_string + '.pkl'
        filename_model = models_path + '/BDT' + suffix + info_string + '.model'
        filename_efficiencies = efficiencies_path + '/Eff_Score_'+suffix + info_string + '.npy'
        
        model_handler.dump_model_handler(filename_handler)
        model_handler.dump_original_model(filename_model, xgb_format=True)
        np.save(filename_efficiencies, eff_score_array)

        print('ML analysis results saved.\n')

    def save_ML_plots(self, model_handler, data, eff_score_array, pt_range, suffix=''):
        fig_path = os.environ['HYPERML_FIGURES']
        info_string = f'_{pt_range[0]}{pt_range[1]}'

        bdt_score_dir = fig_path + '/TrainTest'
        bdt_eff_dir = fig_path + '/Efficiency'
        feat_imp_dir = fig_path + '/FeatureImp'

        bdt_score_plot = plot_utils.plot_output_train_test(model_handler, data, bins=100, log=True)
        if not os.path.exists(bdt_score_dir):
            os.makedirs(bdt_score_dir)

        bdt_score_plot.savefig(bdt_score_dir + '/BDT_Score_' + suffix + info_string + '.pdf')

        bdt_eff_plot = plot_utils.plot_bdt_eff(eff_score_array[1], eff_score_array[0])
        if not os.path.exists(bdt_eff_dir):
            os.makedirs(bdt_eff_dir)

        bdt_eff_plot.savefig(bdt_eff_dir + '/BDT_Eff_' + suffix + info_string + '.pdf')

        feat_imp = plot_utils.plot_feature_imp(data[2][model_handler.get_original_model().get_booster().feature_names], data[3], model_handler)
        if not os.path.exists(feat_imp_dir):
            os.makedirs(feat_imp_dir)

        plt.savefig(feat_imp_dir + '/FeatImp_' + suffix + info_string + '.pdf')
        plt.close()

        print('ML plots saved.\n')


class ModelApplication:

    def __init__(self, pdg_code, multiplicity, branching_ratio, event_filename, hsparse):

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT appplication and signal extraction')

        self.mass = ROOT.TDatabasePDG.Instance().GetParticle(pdg_code).Mass()
        if pdg_code == 310: #K0s
            self.lifetime = 89.54# ps
        elif pdg_code == 3122: #Lambda
            self.lifetime = 263.2# ps
        elif pdg_code == 3334: #Omega
            self.lifetime = 82.1# ps
        elif pdg_code == 3312: #Xi
            self.lifetime = 163.9# ps
        else:
            self.lifetime = ROOT.TDatabasePDG.Instance().GetParticle(pdg_code).Lifetime()*1e+12 #lifetime in ps
        self.multiplicity = multiplicity
        self.branching_ratio = branching_ratio
        background_file = ROOT.TFile(event_filename,"read")
        hist_ev = background_file.Get('hNevents')
        self.n_events = hist_ev.GetBinContent(1)
        background_file.Close()     
        self.hnsparse = hsparse
        print('\nNumber of events: ', self.n_events)

        print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')

    def load_preselection_efficiency(self, suffix = ''):
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES']
        filename_efficiencies = efficiencies_path + f'/PreselEff{suffix}.root'

        tfile = ROOT.TFile(filename_efficiencies)

        self.presel_histo = tfile.Get("PreselEff")
        self.presel_histo.SetDirectory(0)

        return self.presel_histo

    def load_ML_analysis(self, pt_range, suffix=''):

        info_string = f'_{pt_range[0]}{pt_range[1]}'

        handlers_path = os.environ['HYPERML_MODELS'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES']

        filename_handler = handlers_path + '/model_handler_' + suffix + info_string + '.pkl'
        filename_efficiencies = efficiencies_path + '/Eff_Score_' + suffix + info_string + '.npy'

        eff_score_array = np.load(filename_efficiencies)

        model_handler = ModelHandler()
        model_handler.load_model_handler(filename_handler)

        return eff_score_array, model_handler

    def get_preselection_efficiency(self, ptbin_index):
        return self.presel_histo.GetBinContent(ptbin_index)

    def load_sigma_array(self, pt_range):

        info_string = '_{}{}{}'.format(pt_range[0], pt_range[1], split)
        sigma_path = os.environ['HYPERML_UTILS']+'/FixedSigma'
        filename_sigma = sigma_path + "/sigma_array" + info_string + '.npy'
        return np.load(filename_sigma)

    def significance_scan(self, pre_selection_efficiency, eff_score_array, pt_range, pt_spectrum, custom = False, suffix = '', sigma_mass = 2, mass_range=0.04):
        print('\nSignificance scan: ...')

        hist_range = [self.mass*(1-mass_range), self.mass*(1+mass_range)]
        peak_range = [self.mass-3*sigma_mass, self.mass+3*sigma_mass]
        
        bdt_efficiency = eff_score_array[0]
        threshold_space = eff_score_array[1]

        expected_signal = []
        significance = []
        significance_error = []
        for index, tsd in enumerate(threshold_space):
            histo_name = f'score{tsd:.2f}'
            h1_minv = au.h1_from_sparse(self.hnsparse, pt_range, tsd, name=histo_name)
            fit_tpl = ROOT.TF1('fitTpl', 'pol1(0)', hist_range[0], hist_range[1])
            peak_bins = [h1_minv.GetXaxis().FindBin(peak_range[0]), h1_minv.GetXaxis().FindBin(peak_range[1])]
            for bin in range(peak_bins[0], peak_bins[1]+1):
                h1_minv.SetBinContent(bin, 0)

            h1_minv.Fit(fit_tpl, "QRL", "")

            exp_signal = au.expected_signal_counts(
                pt_spectrum, self.multiplicity, self.branching_ratio, pt_range, pre_selection_efficiency * bdt_efficiency[index],
                self.n_events)

            exp_background = fit_tpl.Integral(peak_range[0], peak_range[1])/h1_minv.GetBinWidth(1)
            expected_signal.append(exp_signal)

            if (exp_background < 0):
                exp_background = 0

            sig = exp_signal / np.sqrt(exp_signal + exp_background + 1e-10)
            sig_error = au.significance_error(exp_signal, exp_background)

            if custom:
                significance.append(sig)
                significance_error.append(sig_error)
            else:
                significance.append(sig * bdt_efficiency[index])
                significance_error.append(sig_error * bdt_efficiency[index]) 

        max_index = np.argmax(significance)
        max_score = threshold_space[max_index]
        #max_significance = significance[max_index]
        data_range_array = [pt_range[0], pt_range[1]]
        pu.plot_significance_scan_hsp(
            max_index, significance, significance_error, expected_signal, self.hnsparse, threshold_space,
            data_range_array, self.mass, custom, suffix, sigma_mass)

        bdt_eff_max_score = bdt_efficiency[max_index]

        print('Significance scan: Done!')

        # return max_score, bdt_eff_max_score, max_significance
        return bdt_eff_max_score, max_score

def load_mcsigma(cent_class, pt_range, ct_range, split=''):
    info_string = f'_{cent_class[0]}{cent_class[1]}_{pt_range[0]}{pt_range[1]}_{ct_range[0]}{ct_range[1]}'
    sigma_path = os.environ['HYPERML_UTILS'] + '/FixedSigma'

    file_name = f'{sigma_path}/sigma_array{info_string}.npy'

    return np.load(file_name)
