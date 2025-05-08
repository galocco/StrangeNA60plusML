
python3 run_analysis.py -t -a ../Config/OMEGA_L5_E40.yaml

python3 run_analysis.py -t -a ../Config/ANTIXI_L5_E40.yaml
python3 run_analysis.py -t -a ../Config/ANTILAMBDA_L5_E40.yaml
python3 run_analysis.py -t -a ../Config/XI_L5_E40.yaml

python3 run_analysis.py -t -a ../Config/K0S_L5_E40.yaml
python3 fit_mc.py -dg -g ../Config/K0S_L5_E40.yaml
python3 fit.py ../Config/K0S_L5_E40.yaml
python3 scale_plot.py ../Config/K0S_L5_E40.yaml

python3 run_analysis.py -t -a ../Config/ANTILAMBDA_L5_E40.yaml
python3 fit_mc.py -dg -g ../Config/ANTILAMBDA_L5_E40.yaml
python3 fit.py ../Config/ANTILAMBDA_L5_E40.yaml
python3 scale_plot.py ../Config/ANTILAMBDA_L5_E40.yaml
python3 BSratio.py ../Config/ANTILAMBDA_L5_E40.yaml


python3 fit_mc.py -dg -g ../Config/OMEGA_L5_E40.yaml
python3 fit.py ../Config/OMEGA_L5_E40.yaml
python3 scale_plot.py ../Config/OMEGA_L5_E40.yaml

python3 fit_mc.py -dg -g ../Config/ANTIXI_L5_E40.yaml
python3 fit.py ../Config/ANTIXI_L5_E40.yaml
python3 scale_plot.py ../Config/ANTIXI_L5_E40.yaml



python3 fit_mc.py -dg -g ../Config/ANTILAMBDA_L5_E40.yaml
python3 fit.py ../Config/ANTILAMBDA_L5_E40.yaml
python3 scale_plot.py ../Config/ANTILAMBDA_L5_E40.yaml

python3 fit_mc.py -dg -g ../Config/XI_L5_E40.yaml
python3 fit.py ../Config/XI_L5_E40.yaml
python3 scale_plot.py ../Config/XI_L5_E40.yaml

python3 scale_plot.py ../Config/OMEGA_L5_E40.yaml
python3 scale_plot.py ../Config/XI_L5_E40.yaml
python3 scale_plot.py ../Config/K0S_L5_E40.yaml
python3 scale_plot.py ../Config/ANTILAMBDA_L5_E40.yaml
python3 scale_plot.py ../Config/LAMBDA_L5_E40.yaml
python3 scale_plot.py ../Config/ANTIXI_L5_E40.yaml

python3 BSratio.py ../Config/OMEGA_L5_E40.yaml
python3 BSratio.py ../Config/XI_L5_E40.yaml
python3 BSratio.py ../Config/K0S_L5_E40.yaml
python3 BSratio.py ../Config/ANTILAMBDA_L5_E40.yaml
python3 BSratio.py ../Config/LAMBDA_L5_E40.yaml
python3 BSratio.py ../Config/ANTIXI_L5_E40.yaml

python3 showSignificance.py 

python3 new_evt_mix.py ../Config/PHI_L5_E40.yaml 