import ROOT
import uproot
import pandas as pandas



self.df_signal = uproot.open(mc_file_name)['ntcand'].arrays(library='pd').query(preselection)
