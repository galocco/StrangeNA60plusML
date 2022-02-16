import io
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from array import array
import analysis_utils as au
import ROOT

matplotlib.use('pdf')

###############################################################################
# define custom colors
kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueCT = ROOT.TColor.GetColorTransparent(kBlueC, 0.5)
kRedC = ROOT.TColor.GetColor('#e31a1c')
kRedCT = ROOT.TColor.GetColorTransparent(kRedC, 0.5)
kPurpleC = ROOT.TColor.GetColor('#911eb4')
kPurpleCT = ROOT.TColor.GetColorTransparent(kPurpleC, 0.5)
kOrangeC = ROOT.TColor.GetColor('#ff7f00')
kOrangeCT = ROOT.TColor.GetColorTransparent(kOrangeC, 0.5)
kGreenC = ROOT.TColor.GetColor('#33a02c')
kGreenCT = ROOT.TColor.GetColorTransparent(kGreenC, 0.5)
kMagentaC = ROOT.TColor.GetColor('#f032e6')
kMagentaCT = ROOT.TColor.GetColorTransparent(kMagentaC, 0.5)
kYellowC = ROOT.TColor.GetColor('#ffe119')
kYellowCT = ROOT.TColor.GetColorTransparent(kYellowC, 0.5)


def plot_efficiency_significance(tsd, significance, efficiency, data_range_array):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'

    ax1.set_xlabel('BDT Score')
    ax1.set_ylabel('Significance', color=color)
    ax1.plot(tsd, significance, color=color)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    ax2 = ax1.twinx()

    color = 'tab:red'
    # we already handled the x-label with ax1
    ax2.set_ylabel('BDT efficiency', color=color)
    ax2.plot(tsd, efficiency, color=color)
    ax2.tick_params(axis='y', labelcolor=color, direction='in')

    fig.tight_layout()

    fig_eff_path = os.environ['FIGURES']+'/Significance'
    if not os.path.exists(fig_eff_path):
        os.makedirs(fig_eff_path)

    fig_name = '/sign_eff_pT{}{}.png'.format(
        data_range_array[1],
        data_range_array[2])
    plt.savefig(fig_eff_path + fig_name)
    plt.close()


def plot_significance_scan_hsp(
        max_index, significance, significance_error, expected_signal, hnsparse, score_list, data_range_array,
        mass, custom = False, suffix = '', sigma_mass=0.005):
    if custom:
        label = 'Significance x Efficiency'
    else:
        label = 'Significance'

    max_score = score_list[max_index]

    peak_range = [mass-3*sigma_mass, mass+3*sigma_mass]

    h1_minv = au.h1_from_sparse(hnsparse, data_range_array, score_list[max_index], name="max_sig")
    hist_range = [h1_minv.GetXaxis().GetXmin(), h1_minv.GetXaxis().GetXmax()]
    
    mass_bins = h1_minv.GetNbinsX()

    bkg_tpl_l = ROOT.TF1('fitBkg_l', 'pol1(0)', hist_range[0], peak_range[0])
    bkg_tpl_r = ROOT.TF1('fitBkg_r', 'pol1(0)', peak_range[1], hist_range[1])
    bkg_tpl_l.SetLineColor(ROOT.kGreen+2)
    bkg_tpl_r.SetLineColor(ROOT.kGreen+2)
    fit_tpl = ROOT.TF1('fitTpl','pol1(0)+gausn(2)', peak_range[0], peak_range[1])#

    fit_tpl.SetLineColor(ROOT.kOrange+4)
    peak_bins = [h1_minv.GetXaxis().FindBin(peak_range[0]), h1_minv.GetXaxis().FindBin(peak_range[1])]
    h1_minv.GetYaxis().SetTitle(f'Counts/{1000*((hist_range[1] - hist_range[0])/mass_bins):.1f}'+' MeV/#it{c}^{2}')

    for bin in range(peak_bins[0], peak_bins[1]+1):
        h1_minv.SetBinContent(bin, 0)
        h1_minv.SetBinError(bin, 0)

    h1_minv.Fit(bkg_tpl_l, "QRM", "", hist_range[0], hist_range[1])
    bkg_tpl_l.SetRange(hist_range[0], peak_range[0])
    h1_peak = h1_minv.Clone()
    h1_peak.SetName("peak")

    for par in range(2):
        fit_tpl.SetParameter(par, bkg_tpl_l.GetParameter(par))
        bkg_tpl_r.SetParameter(par, bkg_tpl_l.GetParameter(par))
    fit_tpl.SetParameter(2, expected_signal[max_index]*((hist_range[1] - hist_range[0])/mass_bins))
    fit_tpl.SetParameter(2 + 1, mass)
    fit_tpl.SetParameter(2 + 2, sigma_mass)
    cv_inv = ROOT.TCanvas("cv_iv","cv", 1024, 768)
    cv_inv.cd()
    ROOT.gStyle.SetOptStat(0)
    h1_minv.SetTitle(f'{data_range_array[0]:.2f} #leq'+' #it{p}_{T}'+f' #leq {data_range_array[1]:.2f}'+' GeV/#it{c}  '+f'Cut Score={max_score:0.2f}  Significance={significance[max_index]:.2f}  Raw yield={expected_signal[max_index]:.0f}')
    h1_peak.SetTitle(f'{data_range_array[0]:.2f} #leq'+' #it{p}_{T}'+f' #leq {data_range_array[1]:.2f}'+' GeV/#it{c}  '+f'Cut Score={max_score:0.2f}  Significance={significance[max_index]:.2f}  Raw yield={expected_signal[max_index]:.0f}')
    
    for bin in range(1, h1_minv.GetNbinsX()+1):
        if peak_bins[0] <= bin <= peak_bins[1]:
            val = fit_tpl.Eval(h1_minv.GetBinCenter(bin))
            h1_peak.SetBinContent(bin, val)
            h1_peak.SetBinError(bin, ROOT.TMath.Sqrt(val))
        else:
            h1_peak.SetBinContent(bin, 0.001)
            h1_peak.SetBinError(bin, 0.001)
    
    h1_minv.SetMarkerColor(ROOT.kBlue)
    h1_peak.SetMarkerColor(ROOT.kRed)
    h1_minv.SetMarkerStyle(20)
    h1_peak.SetMarkerStyle(20)
    legend = ROOT.TLegend(0.6,0.6,0.9,0.9)
    legend.AddEntry(bkg_tpl_l,"Background fit","l")
    legend.AddEntry(fit_tpl,"Signal model (Gauss)","l")
    legend.AddEntry(h1_minv,"Data","pe")
    legend.AddEntry(h1_peak,"Pseudo data","pe")
    if h1_peak.GetMaximum() > h1_minv.GetMaximum():
        h1_peak.GetYaxis().SetRangeUser(0.1, h1_peak.GetMaximum()*1.3)
    else:
        h1_peak.GetYaxis().SetRangeUser(0.1, h1_minv.GetMaximum()*1.3)
    h1_peak.Draw("e")
    h1_minv.Draw("e same")
    fit_tpl.Draw("same")
    bkg_tpl_l.Draw("same")
    bkg_tpl_r.Draw("same")
    legend.Draw()
    #significance scan
    cv_sig = ROOT.TCanvas("cv_sig","cv", 1024, 768)
    cv_sig.cd()
    
    score_binning = array('d', score_list[::-1])
    score_err_list = []
    for i in range(len(score_list)):
        score_err_list.append(0)
    
    score_err_binning = array('d', score_err_list)
    sgn_binning = array('d', significance[::-1])
    sgn_err_binning = array('d', significance_error[::-1])

    h1_sign = ROOT.TGraphErrors(len(score_list)-1 ,score_binning, sgn_binning, score_err_binning, score_err_binning)
    h1_sign_err = ROOT.TGraphErrors(len(score_list)-1 ,score_binning, sgn_binning, score_err_binning, sgn_err_binning)

    h1_sign_err.SetTitle(f'{data_range_array[0]:.2f} #leq'+' #it{p}_{T}'+f' #leq {data_range_array[1]:.2f}'+' GeV/#it{c}  '+f'Cut Score={max_score:0.2f}  Significance={significance[max_index]:.2f}  Raw yield={expected_signal[max_index]:.0f}')
    h1_sign_err.GetXaxis().SetTitle("score")
    h1_sign_err.GetXaxis().SetRangeUser(score_list[::-1][0], score_list[::-1][-1])
    h1_sign_err.GetYaxis().SetTitle(label)
    h1_sign.SetLineColor(ROOT.kBlue)
    h1_sign_err.SetLineColor(ROOT.kAzure+8)
    h1_sign_err.SetFillColor(ROOT.kAzure+8)
    h1_sign_err.SetMarkerColor(1)
    h1_sign.SetMarkerColor(1)

    h1_sign_err.Draw("ACP E4")
    h1_sign.Draw("C same")
    #h1_sign_err.GetYAxis().SetRangeUser(-6,6)
    #h1_sign.GetYAxis().SetRangeUser(-6,6)
    fig_name = f'Significance_pT{data_range_array[0]:.2f}{data_range_array[1]:.2f}_{suffix}.png'
    fig_sig_path = os.environ['FIGURES']+'/Significance'
    
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    cv_sig.SaveAs(fig_sig_path + '/' + fig_name)

    fig_name = f'InvMass_pTpT{data_range_array[0]:.2f}{data_range_array[1]:.2f}_{suffix}.png'
    cv_inv.SaveAs(fig_sig_path + '/' + fig_name)

def plot_confusion_matrix(y_true, df, score,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, fig_name='confusion.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # if the score is closer to max then to min it's recognised as signal
    y_pred = [1 if i > score else 0 for i in df['score']]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['Background', 'Signal']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    fig_sig_path = os.environ['FIGURES']+'/Confusion'
    if not os.path.exists(fig_sig_path):
        os.makedirs(fig_sig_path)

    plt.savefig(fig_sig_path + '/' + fig_name)
    plt.close()

    return ax

def get_sNN(e_nucleon):
    if e_nucleon == 158:
        return 17.3
    elif e_nucleon == 80:
        return 12.3
    elif e_nucleon == 40:
        return 8.8
    elif e_nucleon == 30:
        return 7.6
    elif e_nucleon == 20:
        return 6.3
    else:
        print("energy not available")
        return 0

def get_decimal(error):
    decimal = 0
    while error < 1:
        error *= 10
        decimal += 1
    return decimal

def set_to_decimal(value):
    decimal = get_decimal(value)
    return round(value, decimal)

