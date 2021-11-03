#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TF1.h"
#include "TDatabasePDG.h"
using namespace std;

float get_pt_integral(float mass,float pt_min = 0,float pt_max = -1){
    float T = 227.45;
    float t_max = 0;
    float int_max = 0;
    if (pt_max == -1){
        int_max = 0;
    }
    else{
        t_max = TMath::Sqrt(pt_max*pt_max+mass*mass)/T;
        std::cout<<"t_max: "<<t_max<<"\n";
        int_max = -T*T*(TMath::Exp(-t_max)*(1+t_max));
        std::cout<<"int_max: "<<int_max<<"\n";
    }
    float t_min = TMath::Sqrt(pt_min*pt_min+mass*mass)/T;
    float int_min = -T*T*(TMath::Exp(-t_min)*(1+t_min));
    std::cout<<"t_min: "<<t_min<<"\n";
    std::cout<<"int_min: "<<int_min<<"\n";
    return int_max - int_min;
}

void check_multiplicity(TString signal_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntSig_L5_E40_train.root", TString data_path = "/home/giacomo/StrangeNA60plusML/Data/K0S_L5_E40/fntBkg_K0S_L5_E40_data.root", int pdg_code = 310, TString suffix="K0S_L5_E40"){
    TFile* signal_file = new TFile(signal_path.Data(),"read");
    TFile* data_file = new TFile(data_path.Data(),"read");
    
    float mass = TDatabasePDG::Instance()->GetParticle(pdg_code)->Mass();
    TF1* pt_distr = new TF1("pt_distr", "[0]*x*exp(-TMath::Sqrt(x**2+[2]**2)/[1])",0,3);
    pt_distr->SetParameter(1,0.22745);
    pt_distr->SetParLimits(1,0.227*0.6,0.227*1.2);
    pt_distr->FixParameter(2,mass);
    float pt_corr = get_pt_integral(mass, 0,4)/get_pt_integral(mass,0,-1);
    std::cout<<"pt_corr = "<<pt_corr<<"\n";
    TTree* tree_rec = (TTree*) signal_file->Get("ntcand");
    TTree* tree_gen = (TTree*) signal_file->Get("ntgen");
    TTree* tree_data = (TTree*) data_file->Get("ntcand");

    float n_rec = (float)tree_rec->GetEntries(); 
    float n_gen = (float)tree_gen->GetEntries(); 
    float n_data = (float)tree_data->GetEntries(); 
    printf("%f\n",n_rec);
    Float_t eff = n_rec/n_gen;

    printf("efficiency: %f \n",eff);
    int n_ev = 9000000;
    float bratio = 0.489;
    Float_t pt, m, true_cand,cosp;
    TH1D* hist_eff = new TH1D("hist_eff",";;",60,-1,3);
    TH1D* hist_gen = new TH1D("hist_gen",";;",60,-1,3);

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
    TH1D* hist_data = new TH1D("hist_data",";;",60,-1,3);
    tree_data->SetBranchAddress("pt",&pt);
    tree_data->SetBranchAddress("m",&m);
    tree_data->SetBranchAddress("true",&true_cand);
    tree_data->SetBranchAddress("cosp",&cosp);
    for(int i =0; i<n_data; i++){
        tree_data->GetEntry(i);
        if (cosp < 0.999999)
            continue;
        if(true_cand > 0.5)
            hist_data->Fill(pt);
    }
    int rec_part = 0;
    for(int bin=1;bin<=hist_data->GetNbinsX();bin++){
        hist_data->SetBinContent(bin, hist_data->GetBinContent(bin)/hist_eff->GetBinContent(bin));
        hist_data->SetBinError(bin, hist_data->GetBinError(bin)/hist_eff->GetBinContent(bin));
        rec_part += hist_data->GetBinContent(bin);

    }
    hist_data->Fit(pt_distr, "MR+");
    float corr_part = rec_part/n_ev/bratio;
    float err_corr = TMath::Sqrt(rec_part)/n_ev/bratio;
    printf("multiplicity = %f +- %f\n",corr_part/pt_corr,err_corr/pt_corr);
    printf("multiplicity gen = %f\n", 39.15);
    TFile* results = new TFile(Form("params_check_%s.root",suffix.Data()),"recreate");
    hist_eff->Write();
    hist_data->Write();
    hist_gen->Fit(pt_distr, "MR+");
    hist_gen->Write();
    signal_file->Close();
    data_file->Close();
}

TH1D* load_histogram(TString hist_name, TString file_name, TString tree_name){
    TFile* file = new TFile(file_name.Data(),"read");
    TTree* tree = (TTree*) file->Get(tree_name.Data());
    TH1D* hist = new TH1D(hist_name.Data(),";;",60,0,3);
    Float_t pt, cosp;
    float n_entries = (float)tree->GetEntries();
    tree->SetBranchAddress("pt",&pt);
    tree->SetBranchAddress("cosp",&cosp);
    for(int i=0; i<n_entries; i++){
        tree->GetEntry(i);
        if(cosp > 0.9999)
            hist->Fill(pt);
    } 
    return hist;
}