import math
from concurrent.futures import ThreadPoolExecutor
from math import floor, log10
import warnings
import aghast
import numpy as np
import uproot
from hipe4ml.model_handler import ModelHandler
from ROOT import TF1, TH1D, TCanvas, TPaveStats, TPaveText, gStyle, THnSparseD, TMath
from array import array
# avoid pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_skimmed_large_data_std_hsp(mass, data_path, pt_bins, preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 300])
    xmin  = array('d', [mass*(1-range), 0.])
    xmax  = array('d', [mass*(1+range), 3.])
    hsparse = THnSparseD('sparse_m_pt', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});counts', 2, nbins, xmin, xmax)

    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    if preselection != "":
        preselection = " and "+preselection

    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'
            df_tmp = data.query(data_range+preselection)

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind]])
                hsparse.Fill(x)

    return hsparse 

def get_skimmed_large_data_hsp(mass, data_path, pt_bins, training_columns, suffix='', preselection='', range=0.04):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    nbins = array('i', [40, 3000, 20000])
    xmin  = array('d', [mass*(1-range), 0., -20])
    xmax  = array('d', [mass*(1+range), 3.,  20])
    hsparse = THnSparseD('sparse_m_pt_s', ';mass (GeV/#it{c}^{2});#it{p}_{T} (GeV/#it{c});score;counts', 3, nbins, xmin, xmax)

    handlers_path = "../Models/handlers"
    efficiencies_path = "../Results/Efficiencies"

    executor = ThreadPoolExecutor()
    data_tree_name = data_path + ":/ntcand"
    iterator = uproot.iterate(data_tree_name, executor=executor, library='pd')

    if preselection != "":
        preselection = " and "+preselection

    for data in iterator:
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
            info_string = f'_{ptbin[0]}{ptbin[1]}'

            filename_handler = handlers_path + '/model_handler_' +suffix+ info_string + '.pkl'
            filename_efficiencies = efficiencies_path + '/Eff_Score_' + suffix + info_string + '.npy'

            model_handler = ModelHandler()
            model_handler.load_model_handler(filename_handler)

            eff_score_array = np.load(filename_efficiencies)
            tsd = eff_score_array[1][-1]

            data_range = f'{ptbin[0]}<pt<{ptbin[1]}'
            df_tmp = data.query(data_range+preselection)
            df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))
            df_tmp = df_tmp.query('score>@tsd')

            for ind in df_tmp.index:
                x = array('d', [df_tmp['m'][ind], df_tmp['pt'][ind], df_tmp['score'][ind]])
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
    step = 3.0/3000.0/2.
    ptbin_min = hnsparse_clone.GetAxis(1).FindBin(pt_range[0]+step)
    ptbin_max = hnsparse_clone.GetAxis(1).FindBin(pt_range[1]-step)
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
    width = th1.GetBinWidth(1)*1000 #to MeV
    th1.GetYaxis().SetTitle(r'Counts/%0.1f MeV' % (width))
    th1.SetTitle('')
    th1.SetDirectory(0)
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
        histo, pt_range, mass, nsigma=3, model="pol2", mass_range=0.04, Eint=17.3, peak_mode=True, gauss=True, crystal=False):
    
    #mass != TDatabasePDG.Instance().GetParticle(333).Mass()
    
    hist_range = [mass*(1-mass_range),mass*(1+mass_range)]
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
    if crystal:
        par = 7
        #fit_tpl = TF1('fitTpl', crystal_ball, 0, 5, par)
        fit_tpl = TF1('fitTpl', f'{model}(0)+gausn({n_bkgpars})+gausn({n_bkgpars+3})', 0, 5)
    else:
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
    if crystal:
        #fit_tpl.SetParameters(1, 2.5, mass, histo.GetRMS()/2., histo.GetMaximum())
        fit_tpl.SetParameter(n_bkgpars, 100)
        fit_tpl.SetParLimits(n_bkgpars, 0, 5000)
        fit_tpl.SetParameter(n_bkgpars + 3, 100)
        fit_tpl.SetParLimits(n_bkgpars + 3, 0, 5000)
        fit_tpl.SetParameter(n_bkgpars + 1, mass)
        fit_tpl.SetParLimits(n_bkgpars + 1, mass-0.005, mass+0.005)
        fit_tpl.SetParameter(n_bkgpars + 4, mass)
        fit_tpl.SetParLimits(n_bkgpars + 4, mass-0.005, mass+0.005)
        fit_tpl.SetParameter(n_bkgpars + 2, 0.0035)
        fit_tpl.SetParLimits(n_bkgpars + 2, 0.001, 0.006)
        fit_tpl.SetParameter(n_bkgpars + 5, 0.0035)
        fit_tpl.SetParLimits(n_bkgpars + 5, 0.001, 0.006)
        #fit_tpl.SetParameter(1, 1.5)
        #fit_tpl.SetParLimits(1, 0.75,3)
        #fit_tpl.SetParameter(0, 1)
        #fit_tpl.SetParLimits(0, 0.5,1)
        #fit_tpl.SetParameter(1, 2.5)
        #fit_tpl.SetParameter(2, mass)
        #fit_tpl.FixParameter(n_bkgpars+2, mass)
        #fit_tpl.FixParameter(n_bkgpars+5, mass)
        #fit_tpl.SetParLimits(n_bkgpars+2, mass-0.0002,mass+0.0002)
        #fit_tpl.SetParLimits(n_bkgpars+5, mass-0.0002,mass+0.0002)
        #fit_tpl.SetParameter(n_bkgpars+3, 0.002)
        #fit_tpl.SetParameter(n_bkgpars+6, 0.002)
        #fit_tpl.SetParLimits(n_bkgpars+3, 0.001, 0.01)
        #fit_tpl.SetParLimits(n_bkgpars+6, 0.001, 0.01)
        #fit_tpl.SetParameter(3, histo.GetRMS()/2.)
        #fit_tpl.SetParameter(4, histo.GetMaximum())
        #fit_tpl.SetParameter(5, -2000)
        #fit_tpl.SetParameter(6, 2000)
    else:
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
    fit_tpl.SetParameter(n_bkgpars + 2, 0.0025)
    fit_tpl.SetParLimits(n_bkgpars + 2, 0.0005, 0.0035)

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
        if crystal:
            signal = (fit_tpl.GetParameter(n_bkgpars) + fit_tpl.GetParameter(n_bkgpars+3)) / histo.GetBinWidth(1)
            errsignal = TMath.Sqrt(fit_tpl.GetParError(n_bkgpars)*fit_tpl.GetParError(n_bkgpars)+fit_tpl.GetParError(n_bkgpars+3)*fit_tpl.GetParError(n_bkgpars+3)) / histo.GetBinWidth(1)
        else:
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

def rename_df_columns(df):
    rename_dict = {}

    for col in df.columns:

        if col.endswith('_f'):
            rename_dict[col] = col[:-2]
    
    df.rename(columns = rename_dict, inplace=True)

def pt_array_to_mt_m0_array(pt_array, mass):
    mt_array = []
    for pt_item in pt_array:
        mt_array.append(math.sqrt(pt_item**2+mass**2)-mass)
    return mt_array

def crystal_ball(x, par):
    # Crystal Ball function fit
    alpha = par[0]
    n =     par[1]
    meanx = par[2]
    sigma = par[3]
    nn =    par[4]
    p0 =    par[5]
    p1 =    par[6]
    #print("alpha: ",alpha)
    #print("n: ",n)
    #print("meanx: ",meanx)
    #print("sigma: ",sigma)
    #print("nn: ",nn)
    #print("p0: ",p0)
    #print("p1: ",p1)
    a = TMath.Power((n/math.fabs(alpha)), n) * math.exp(-0.5*alpha*alpha)
    b = n/math.fabs(alpha) - math.fabs(alpha)
 
    arg = (x[0] - meanx)/sigma

    if arg > -1.*alpha:
        fitval = nn * math.exp(-0.5*arg*arg)
    else:
        fitval = nn * a * TMath.Power((b-arg), (-1*n))
      
    return fitval+p0+p1*x[0]
