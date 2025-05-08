import ROOT
import plot_utils as pu

EINT = pu.get_sNN(40)

particle_list = [
                "K0S_L5_E40",
                "LAMBDA_L5_E40",
                "ANTILAMBDA_L5_E40",
                "XI_L5_E40",
                "ANTIXI_L5_E40",
                "OMEGA_L5_E40",
                ]

label_list = [
                "K^{0}_{S}",
                "#Lambda^{0}",
                "#bar{#Lambda}^{0}",
                "#Xi^{-}",
                "#bar{#Xi}^{+}",
                "#Omega^{-}+#bar{#Omega}^{+}",
                ]

fit_list = [
    "d-gauss_pol1",
    "d-gauss_pol1",
    "d-gauss_pol1",
    "d-gauss_pol1",
    "gauss_pol1",
    "gauss_pol1",
]

color_list = [
    ROOT.kRed,
    ROOT.kBlue,
    ROOT.kGreen,
    ROOT.kBlack,
    ROOT.kMagenta,
    ROOT.kCyan,
]

marker_list = [
    20,
    21,
    22,
    23,
    24,
    25,
]

ROOT.gStyle.SetOptStat(0)


cv_BS = ROOT.TCanvas("cv_BS","cv_BS")

# Set smaller top and right margins
cv_BS.SetTopMargin(0.05)   # Default is ~0.1
cv_BS.SetRightMargin(0.04) # Default is ~0.1

# Optional: Set left and bottom if needed
cv_BS.SetLeftMargin(0.13)
cv_BS.SetBottomMargin(0.12)

cv_Sgn = ROOT.TCanvas("cv_Sgn","cv_Sgn")

# Set smaller top and right margins
cv_Sgn.SetTopMargin(0.05)   # Default is ~0.1
cv_Sgn.SetRightMargin(0.04) # Default is ~0.1

# Optional: Set left and bottom if needed
cv_Sgn.SetLeftMargin(0.13)
cv_Sgn.SetBottomMargin(0.12)

cv_Sgn_ev = ROOT.TCanvas("cv_Sgn_ev","cv_Sgn_ev")

# Set smaller top and right margins
cv_Sgn_ev.SetTopMargin(0.05)   # Default is ~0.1
cv_Sgn_ev.SetRightMargin(0.04) # Default is ~0.1

# Optional: Set left and bottom if needed
cv_Sgn_ev.SetLeftMargin(0.13)
cv_Sgn_ev.SetBottomMargin(0.12)


hist_BS = {}
hist_Sgn = {}
hist_Sgn_ev = {}
file = {}
min_BS = 10**10
max_BS = -5
min_Sgn = 10**10
max_Sgn = -5
min_Sgn_ev = 10**10
max_Sgn_ev = -5


legend = ROOT.TLegend(0.73,0.7,1.03,0.95)
legend.SetTextSize(0.05)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetNColumns(2)

for particle, fit, color, marker,label  in zip(particle_list, fit_list, color_list, marker_list, label_list):
    file[particle] = ROOT.TFile(f"../Results/{particle}_analysis/{particle}_analysis_results_BS.root","read")

    hist_BS[particle] = file[particle].Get(f"hist_BS_{fit}")
    hist_Sgn[particle] = file[particle].Get(f"hist_Sgn_{fit}")
    hist_Sgn_ev[particle] = file[particle].Get(f"hist_Sgn_ev_{fit}")

    hist_BS[particle].SetMarkerColor(color)
    hist_Sgn[particle].SetMarkerColor(color)
    hist_Sgn_ev[particle].SetMarkerColor(color)

    hist_BS[particle].SetLineColor(color)
    hist_Sgn[particle].SetLineColor(color)
    hist_Sgn_ev[particle].SetLineColor(color)

    hist_BS[particle].SetMarkerStyle(marker)
    hist_Sgn[particle].SetMarkerStyle(marker)
    hist_Sgn_ev[particle].SetMarkerStyle(marker)

    legend.AddEntry(hist_BS[particle], label, "lpe")

    for i in range(1,hist_BS[particle].GetNbinsX()+1):
        bin_value = hist_BS[particle].GetBinContent(i)
        if  bin_value < min_BS:
            min_BS = bin_value 
        if  bin_value > max_BS:
            max_BS = bin_value 

    for i in range(1,hist_Sgn[particle].GetNbinsX()+1):
        bin_value = hist_Sgn[particle].GetBinContent(i)
        if  bin_value < min_Sgn:
            min_Sgn = bin_value 
        if  bin_value > max_Sgn:
            max_Sgn = bin_value 

    for i in range(1,hist_Sgn_ev[particle].GetNbinsX()+1):
        bin_value = hist_Sgn_ev[particle].GetBinContent(i)
        if  bin_value < min_Sgn_ev:
            min_Sgn_ev = bin_value 
        if  bin_value > max_Sgn_ev:
            max_Sgn_ev = bin_value 

    cv_BS.cd()
    hist_BS[particle].Draw("same ep")

    cv_Sgn.cd()
    hist_Sgn[particle].Draw("same ep")

    cv_Sgn_ev.cd()
    hist_Sgn_ev[particle].Draw("same ep")


# print fit info on the canvas
pinfo2 = ROOT.TPaveText(0.1, 0.79, 0.95, 0.94, "NDC")
pinfo2.SetBorderSize(0)
pinfo2.SetFillStyle(0)
pinfo2.SetTextAlign(11)
pinfo2.SetTextFont(42)
pinfo2.SetTextSize(0.048)
string = 'Pb-Pb #sqrt{s_{NN}} = '+f'{EINT} GeV, centrality 0-5%'
pinfo2.AddText(string)
string = '6e+11 ions on target, 15% target int. length'
pinfo2.AddText(string)

cv_BS.cd()
legend.Draw("same")
pinfo2.Draw()

cv_Sgn.cd()
legend.Draw("same")
pinfo2.Draw()

cv_Sgn_ev.cd()
legend.Draw("same")
pinfo2.Draw()

hist_BS[particle_list[0]].GetYaxis().SetRangeUser(min_BS/1.3,max_BS*3)
hist_Sgn[particle_list[0]].GetYaxis().SetRangeUser(min_Sgn/1.3,max_Sgn*3)
hist_Sgn_ev[particle_list[0]].GetYaxis().SetRangeUser(min_Sgn_ev/1.3,max_Sgn_ev*3)


# Set title size (the axis title, not the histogram title)
hist_BS[particle_list[0]].GetXaxis().SetTitleSize(0.05)  # Default is ~0.04
hist_BS[particle_list[0]].GetYaxis().SetTitleSize(0.05)
# Set label size (numbers on the axis)
hist_BS[particle_list[0]].GetXaxis().SetLabelSize(0.04)  # Default is ~0.035
hist_BS[particle_list[0]].GetYaxis().SetLabelSize(0.04)
# Set title size (the axis title, not the histogram title)
hist_Sgn[particle_list[0]].GetXaxis().SetTitleSize(0.05)  # Default is ~0.04
hist_Sgn[particle_list[0]].GetYaxis().SetTitleSize(0.05)
# Set label size (numbers on the axis)
hist_Sgn[particle_list[0]].GetXaxis().SetLabelSize(0.04)  # Default is ~0.035
hist_Sgn[particle_list[0]].GetYaxis().SetLabelSize(0.04)
# Set title size (the axis title, not the histogram title)
hist_Sgn_ev[particle_list[0]].GetXaxis().SetTitleSize(0.05)  # Default is ~0.04
hist_Sgn_ev[particle_list[0]].GetYaxis().SetTitleSize(0.05)
# Set label size (numbers on the axis)
hist_Sgn_ev[particle_list[0]].GetXaxis().SetLabelSize(0.04)  # Default is ~0.035
hist_Sgn_ev[particle_list[0]].GetYaxis().SetLabelSize(0.04)

cv_BS.SetLogy()
cv_Sgn.SetLogy()
cv_Sgn_ev.SetLogy()


cv_BS.Update()
cv_Sgn.Update()
cv_Sgn_ev.Update()


cv_BS.SaveAs("/home/giacomo/StrangeNA60plusML/Results/BS.pdf")
cv_Sgn.SaveAs("/home/giacomo/StrangeNA60plusML/Results/ExpSgnStrangeParticles.pdf")
cv_Sgn_ev.SaveAs("/home/giacomo/StrangeNA60plusML/Results/ExpSgnStrangeParticles.pdf_ev.pdf")

cv_BS.SaveAs("/home/giacomo/StrangeNA60plusML/Results/BS.png")
cv_Sgn.SaveAs("/home/giacomo/StrangeNA60plusML/Results/ExpSgnStrangeParticles.png")
cv_Sgn_ev.SaveAs("/home/giacomo/StrangeNA60plusML/Results/ExpSgnStrangeParticles.pdf_ev.png")



