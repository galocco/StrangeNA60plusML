#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TDatabasePDG.h"
#include "TCanvas.h"
using namespace std;
/*
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
*/
float get_pt_integral(float mass, float T, float pt_min = 0,float pt_max = -1){
    float t_max = 0;
    float int_max = 0;
    if (pt_max == -1){
        int_max = 0;
    }
    else{
        t_max = TMath::Sqrt(pt_max*pt_max+mass*mass)/T;
        //std::cout<<"t_max: "<<t_max<<"\n";
        int_max = -T*T*(TMath::Exp(-t_max)*(1+t_max));
        //std::cout<<"int_max: "<<int_max<<"\n";
    }
    float t_min = TMath::Sqrt(pt_min*pt_min+mass*mass)/T;
    float int_min = -T*T*(TMath::Exp(-t_min)*(1+t_min));
    //std::cout<<"t_min: "<<t_min<<"\n";
    //std::cout<<"int_min: "<<int_min<<"\n";
    return int_max - int_min;
}

void check_multiplicity(TString signal_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntSig_L5_E40_train.root", TString data_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntBkg_K0S_L5_E40_data.root", int pdg_code = 310, TString suffix="K0S_L5_E40"){
    TFile* signal_file = new TFile(signal_path.Data(),"read");
    TFile* data_file = new TFile(data_path.Data(),"read");
    std::cout<<pdg_code<<"\n";
    float mass = TDatabasePDG::Instance()->GetParticle(pdg_code)->Mass();
    TF1* pt_distr = new TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",0,3);
    float T = 0.2446;
    float bratio = 0.489;//phi
    float multiplicity = 2.55;
    float n_ev = 9000000;
    float pt_max = 3;
    if(TMath::Sqrt(pdg_code)==310){//Kaon
        bratio =  0.692;
        multiplicity = 40;
        T = 0.22745;
    }
    else if(pdg_code==3122){//Lambda
        std::cout<<"Lambda\n";
        bratio = 0.639;
        multiplicity = 0.68;        
        T = 0.301;
    }
    else if(pdg_code==3312){//Xi
        bratio =  0.638;
        multiplicity = 2.96;
        T = 0.222;
        pt_max = 2.5;
    }
    else if(pdg_code==-3312){//Xi
        bratio =  0.638;
        multiplicity = 0.13;
        T = 0.255;
    }
    else if(TMath::Sqrt(pdg_code)==3334){//omega
        bratio =  0.433;
        multiplicity = 0.14;
        T = 0.218;
    }
    pt_distr->SetParameter(1,T);
    pt_distr->SetParLimits(1,T*0.6,T*1.2);
    pt_distr->FixParameter(2,mass);
    float p1 = get_pt_integral(mass, T, 0,3);
    float p2 = get_pt_integral(mass, T, 0,-1);
    float pt_corr = p1/p2;
    std::cout<<"multiplicity = "<<multiplicity<<"\n";
    std::cout<<"bratio = "<<bratio<<"\n";
    std::cout<<"T = "<<T<<"\n";
    std::cout<<"n_ev = "<<n_ev<<"\n";
    TTree* tree_rec = (TTree*) signal_file->Get("ntcand");
    TTree* tree_gen = (TTree*) signal_file->Get("ntgen");
    TTree* tree_data = (TTree*) data_file->Get("ntcand");

    float n_rec = (float)tree_rec->GetEntries(); 
    float n_gen = (float)tree_gen->GetEntries(); 
    float n_data = (float)tree_data->GetEntries();
    Float_t eff = n_rec/n_gen;

    printf("efficiency: %f \n",eff);
    Float_t pt, m, true_cand,cosp;
    TH1D* hist_eff = new TH1D("hist_eff",";;",30,0,pt_max);
    TH1D* hist_gen = new TH1D("hist_gen",";;",30,0,pt_max);

    tree_rec->SetBranchAddress("pt",&pt);
    tree_rec->SetBranchAddress("cosp",&cosp);
    for(int i =0; i<n_rec; i++){
        tree_rec->GetEntry(i);
        if (cosp < 0.9999)
            continue;
        hist_eff->Fill(pt);
    }

    tree_gen->SetBranchAddress("pt",&pt);
    for(int i =0; i<n_gen; i++){
        tree_gen->GetEntry(i);
        hist_gen->Fill(pt);
    }

    for(int bin=1;bin<=hist_gen->GetNbinsX();bin++){
        if(hist_gen->GetBinContent(bin) ==0){
            hist_eff->SetBinContent(bin, 1);
            hist_gen->SetBinContent(bin, 1);
        }
    }
    hist_eff->Divide(hist_gen);
    TH1D* hist_data = new TH1D("hist_data",";;",30,0,pt_max);
    tree_data->SetBranchAddress("pt",&pt);
    tree_data->SetBranchAddress("m",&m);
    tree_data->SetBranchAddress("true",&true_cand);
    tree_data->SetBranchAddress("cosp",&cosp);
    float counter = 0.01;
    for(int i =0; i<n_data; i++){
        if (i/n_data > counter){
            std::cout<<i/n_data*100<<"%\n";
            counter += 0.01;
        }
        tree_data->GetEntry(i);
        if (cosp < 0.9999)
            continue;
        if(true_cand > 0.5)
            hist_data->Fill(pt);
    }
    float rec_part = 0;
    for(int bin=1;bin<=hist_data->GetNbinsX();bin++){
        hist_data->SetBinContent(bin, hist_data->GetBinContent(bin)/hist_eff->GetBinContent(bin));
        hist_data->SetBinError(bin, hist_data->GetBinError(bin)/hist_eff->GetBinContent(bin));
        rec_part += hist_data->GetBinContent(bin);
        std::cout<<bin<<" - "<<hist_data->GetBinContent(bin)<<"\n";

    }
    hist_data->Fit(pt_distr, "MRI+");
    float corr_part = rec_part/n_ev;//bratio;
    std::cout<<"rec part: "<<rec_part<<"\n";
    std::cout<<"cor part: "<<corr_part<<"\n";
    float err_corr = TMath::Sqrt(rec_part)/n_ev;//bratio;
    printf("multiplicity = %f +- %f\n", (corr_part/pt_corr)/bratio, err_corr/pt_corr);
    printf("multiplicity gen = %f\n", multiplicity);
    TFile* results = new TFile(Form("params_check_%s.root",suffix.Data()),"recreate");
    hist_eff->Write();
    hist_data->Write();
    hist_gen->Fit(pt_distr, "MR+");
    hist_gen->Write();
    signal_file->Close();
    data_file->Close();
}

void check_features(TString signal_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntSig_L5_E40_train.root", TString data_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntBkg_K0S_L5_E40_data.root", int pdg_code = 310, TString suffix="K0S_L5_E40", bool signal_only=false){
    TFile* signal_file = new TFile(signal_path.Data(),"read");
    TFile* data_file = new TFile(data_path.Data(),"read");
    TFile* results = new TFile(Form("check_results_%s.root",suffix.Data()),"recreate");
    
    Float_t mass = TDatabasePDG::Instance()->GetParticle(pdg_code)->Mass();
    Float_t mass_min = mass*0.92;
    Float_t mass_max = mass*1.08;
    TTree* tree_rec = (TTree*) signal_file->Get("ntcand");
    TTree* tree_data = (TTree*) data_file->Get("ntcand");

    float n_rec = (float)tree_rec->GetEntries(); 
    float n_data = (float)tree_data->GetEntries(); 
    int nbody = 2;
    if(pdg_code ==3312)
        nbody = 3;
    const int nvars3b = 15;
    const int nvars2b = 8;
    int nvars = nvars2b;
    if(nbody==3)
        nvars = nvars3b;
    Float_t var[nvars3b];
    
    //                            m       cosp    pt   dca   d0prod  dist  arm  qt
    Float_t min_2b[nvars2b] = {mass_min, 0.9999,  0,    0,  -0.001,    0,  -1,  0};
    Float_t max_2b[nvars2b] = {mass_max,   1,     3,  0.02,  0.001,   30,   1,  1};

    //                            m        cosp cospD  pt   dca  dcaD  dist   distD     bxy    bxyD  arm  armD  qt  qtD         mD
    Float_t min_3b[nvars3b] = {mass_min, 0.9999,  -1,   0,     0,    0,   -1,     -1,  -0.01, -0.01,   -1,  -1,  0,  0, 1.115683-0.01};
    Float_t max_3b[nvars3b] = {mass_max,      1,   1,   3,  0.02, 0.02,   30,     30,   0.01,  0.01,    1,   1,  1,  1, 1.115683+0.01};
    Int_t nbin[nvars3b] =     {60,           20, 400,  20,    20,   20,   30,     30,     20,    20,   20,  20, 20, 20,                 60};

    TString vars_list[nvars3b] = {"m","cosp","cospD","pt","dca","dcaD","dist","distD","bxy","bxyD","arm","armD","qt","qtD","mD"};
    TString vars_list2b[nvars2b] = {"m","cosp","pt","dca","d0prod","dist","arm","qt"};

    TH2F *hArmPodSig = new TH2F("hArmPodSig", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);
    TH2F *hArmPodDSig = new TH2F("hArmPodDSig", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);
    TH2F *hArmPodBkg = new TH2F("hArmPodBkg", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);
    TH2F *hArmPodDBkg = new TH2F("hArmPodDBkg", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);
    TH2F *hArmPodSkg = new TH2F("hArmPodSkg", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);
    TH2F *hArmPodDSkg = new TH2F("hArmPodDSkg", ";#alpha;#it{p}_{T} (GeV/#it{c});counts", 1000, -1, 1, 1000, 0, 2);

    TH1D* hist_list[nvars];
    TH1D* hist_data_sig[nvars];
    TH1D* hist_data_bkg[nvars];
    TCanvas* cv_list[nvars];
    //////////////////////////////////////////////////////////////////////
    for(int i = 0; i<nvars ;i++){
        if(nbody==3){
            tree_rec->SetBranchAddress(vars_list[i].Data(),&var[i]);
            hist_list[i] = new TH1D(Form("hist_%s_sig",vars_list[i].Data()),Form(";%s; pdf",vars_list[i].Data()),nbin[i],min_3b[i],max_3b[i]);
            cv_list[i] = new TCanvas(Form("cv_%s",vars_list[i].Data()),Form("cv_%s",vars_list[i].Data()));

        }
        else{
            tree_rec->SetBranchAddress(vars_list2b[i].Data(),&var[i]);
            hist_list[i] = new TH1D(Form("hist_%s_sig",vars_list2b[i].Data()),Form(";%s; pdf",vars_list2b[i].Data()),nbin[i],min_2b[i],max_2b[i]);
            cv_list[i] = new TCanvas(Form("cv_%s",vars_list2b[i].Data()),Form("cv_%s",vars_list2b[i].Data()));

        }
        hist_list[i]->SetLineColor(kGreen);
    }
    for(int i =0; i<n_rec; i++){
        //std::cout<<i<<"\n";
        if(i%1000000==0)
            std::cout<<"process = "<<i/n_rec*100<<"%\n";
        tree_rec->GetEntry(i);
        if (var[1] < 0.999)
            continue;
        for(int i = 0; i<nvars ;i++){
            hist_list[i]->Fill(var[i]);
            if(nbody == 3){
                hArmPodSig->Fill(var[10],var[12]);
                hArmPodDSig->Fill(var[11],var[13]);
            }
        }
    }
    results->cd();
    //////////////////////////////////////////////////////////////////////
    for(int i = 0; i<nvars ;i++){
        if(nbody==3){
            tree_data->SetBranchAddress(vars_list[i].Data(),&var[i]);
            hist_data_sig[i] = new TH1D(Form("hist_%s_data_sig",vars_list[i].Data()),Form(";%s; pdf",vars_list[i].Data()),nbin[i],min_3b[i],max_3b[i]);
            hist_data_bkg[i] = new TH1D(Form("hist_%s_data_bkg",vars_list[i].Data()),Form(";%s; pdf",vars_list[i].Data()),nbin[i],min_3b[i],max_3b[i]);
        }
        else{
            tree_data->SetBranchAddress(vars_list2b[i].Data(),&var[i]);
            hist_data_sig[i] = new TH1D(Form("hist_%s_data_sig",vars_list2b[i].Data()),Form(";%s; pdf",vars_list2b[i].Data()),nbin[i],min_2b[i],max_2b[i]);
            hist_data_bkg[i] = new TH1D(Form("hist_%s_data_bkg",vars_list2b[i].Data()),Form(";%s; pdf",vars_list2b[i].Data()),nbin[i],min_2b[i],max_2b[i]);
        }
        hist_data_sig[i]->SetLineColor(kBlue);
        hist_data_bkg[i]->SetLineColor(kRed);
    }
    float true_cand = 0;
    tree_data->SetBranchAddress("true" ,&true_cand);
    float reduce = 1;
    if(nbody==2)
        reduce=10;
    printf("n_data: %f\n",n_data);
    printf("n_rec: %f\n",n_rec);
    for(int i =0; i<n_data; i++){
        std::cout<<"i: "<<i<<"\n";
        if(i%10000000==0)
            std::cout<<"process = "<<i/n_data*1*100<<"%\n";
        tree_data->GetEntry(i);
        //if(var[14] < 1.115683-0.001*5 || var[14] > 1.115683+0.001*5) continue;
        //if (var[1] < 0.9999)
        //    continue;
        for(int i = 0; i<nvars ;i++){
            if(true_cand > 0.5){
                hist_data_sig[i]->Fill(var[i]);
                if(nbody==3){
                    hArmPodSkg->Fill(var[10],var[12]);
                    hArmPodDSkg->Fill(var[11],var[13]);
                }
            }
            else{
                hist_data_bkg[i]->Fill(var[i]);
                if(nbody==3){
                    hArmPodBkg->Fill(var[10],var[12]);
                    hArmPodDBkg->Fill(var[11],var[13]);
                }
            }
        }
    }
    
    results->cd();
    for(int i = 0; i<nvars ;i++){
        Float_t max_range_tmp = 0;
        Float_t max_range = 0;
        Float_t min_range = 0;
        //hist_list[i]->Scale(1./hist_list[i]->GetEntries());
        hist_list[i]->Write();
        //hist_data_sig[i]->Scale(1./hist_data_sig[i]->GetEntries());
        hist_data_sig[i]->Write();
        max_range = hist_data_sig[i]->GetMaximum()*1.2;
        max_range_tmp = hist_data_bkg[i]->GetMaximum()*1.2;
        max_range = max_range_tmp > max_range ? max_range_tmp : max_range;
        max_range_tmp = hist_list[i]->GetMaximum()*1.2;
        max_range = max_range_tmp > max_range ? max_range_tmp : max_range;
        cv_list[i]->cd();
        hist_data_sig[i]->GetYaxis()->SetRangeUser(min_range, max_range);
        hist_data_sig[i]->Draw("");
        hist_list[i]->Draw("same");
        if(!signal_only){
            hist_data_bkg[i]->Scale(1./hist_data_bkg[i]->GetEntries());
            hist_data_bkg[i]->Write();
            hist_data_bkg[i]->Draw("same");
        }
        cv_list[i]->Write();
    }
    if(nbody==3){
        hArmPodSkg->Write();
        hArmPodDSkg->Write();
        hArmPodSig->Write();
        hArmPodDSig->Write();
        if(!signal_only){
            hArmPodBkg->Write();
            hArmPodDBkg->Write();
        }
    }
    signal_file->Close();
    data_file->Close();
    results->Close();
}

