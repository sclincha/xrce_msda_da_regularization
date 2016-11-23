
#
__author__ = 'sclincha'
#
#"""
#Domain Regularization ....





import numpy as np
import scipy.sparse as sp
import domain_adaptation_baseline
import dataset_utils
import cPickle
import amazon_exp
import sys,pickle
import string,os
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
import string
import pickle
import msda_exp
#from msda_exp import msda_domainreg_exp
from sklearn.feature_extraction.text import TfidfTransformer

################################################################################
###   SGE SETTINGS #############################################################
##source /usr/local/grid/XRCE/common/settings.sh
##DATASETSIDS=range(12,24)
#DATASETSIDS=range(12,24)
DATASETSIDS=range(12,14)
#DATASET='AMT'
DATASET='20NG'
##FEAT=[2]
##NOISE=[0.9]
##CHANGE EXPSET in EXPSET=20 in script file

exec_path="/home/sclincha/AAT/src/DenoisingAutoencoders/exp_domain_reg.sh"
##resdir_path='/opt/scratch/MLS/usr/sclincha/domainreg_exp_20ng'
##resdir_path='/opt/scratch/MLS/usr/sclincha/domainreg_exp'
#resdir_path='/opt/scratch/MLS/usr/sclincha/dr_target_reg_cv_10e4_feat'
resdir_path='/home/sclincha/AAT/src/DenoisingAutoencoders/dr_target_20NG_cv'
#resdir_path='/opt/scratch/MLS/usr/sclincha/tmp/'

#exp_utils.make_grid(exec_path,resdir_path,DATASETSIDS,NOISE,FEAT)

def param_argument(dataset):
    return str(dataset)

def qsub_cmd(dataset,res_dir_path,exec_name):
    params   =param_argument(dataset)
    exp_name = 'DA_DR'+string.replace(params,':','-')

    #res_dir_name=res_dir_path+'/'+dataset+'_feat'
    #os.system('mkdir -p '+ res_dir_name)
    #os.mkdir(res_dir_name)
    #Remove Standard Output
    #-o /dev/null -e /dev/null
    #cmd_str='qsub -M stephane.clinchant@xrce.xerox.com -m a -cwd -N ' +exp_name+ ' -l vf=2G,h_vmem=12G,p=2 ' \
    #
    #                                                                              '-v PARAM='+"'"+params+"'"+',OUTDIR='+"'"+res_dir_path+"'"+'  '+exec_name

    #-o /opt/scratch/MLS/usr/sclincha/sge_logs/ -e /opt/scratch/MLS/usr/sclincha/sge_logs/
    cmd_str='qsub  -o /opt/scratch/MLS/usr/sclincha/sge_logs/ -e /opt/scratch/MLS/usr/sclincha/sge_logs/ -m a -cwd -N ' +exp_name+ ' -l vf=16G,h_vmem=32G,p=4 ' \
                                                                                  '-v DATA='+  "'"+DATASET+"'"+',PARAM='+"'"+params+"'"+',OUTDIR='+"'"+res_dir_path+"'"+'  '+exec_name
    print cmd_str
    os.system(cmd_str)


def make_grid():
    for dataset in DATASETSIDS:
        qsub_cmd(dataset,resdir_path,exec_path)

def print_results(res_path):
    DATANAME =pickle.load(open('datasets_id.pickle'))
    for i in DATASETSIDS:
        try:
            z=np.loadtxt(res_path+str(i))
            print DATANAME[i],z
        except Exception as e:
            print(e)


def print_results_latex(res_path,DID=DATASETSIDS,did_pickle='datasets_id.pickle'):
    DATANAME =pickle.load(open(did_pickle))
    Z=None
    for i in DID:
        try:
            z=np.loadtxt(res_path+str(i))
            if Z is not None:
                Z = np.vstack([Z,z])
            else:
                Z=z
            msda_acc =100*z[0]
            dr_acc   =100*z[1]
            if type(DATANAME[i])==str:
                #20NG Case
                taskname = DATANAME[i].replace('vs','')
                #print taskname
                tok=string.split(taskname,'_')
                del tok[1]
                #print tok
                TT=tok
                print string.join(TT,'&'),'& %1.2f'% msda_acc,'&','%1.2f'%dr_acc,'\\\\'
            else:
                print string.join(DATANAME[i][:2],'&'),'& %1.2f'% msda_acc,'&','%1.2f'%dr_acc,'\\\\'
        except Exception as e:
            print(e)
    print Z
    print Z.mean(axis=0)


def print_results_notebooks(res_path):
    DATANAME =pickle.load(open('datasets_id.pickle'))
    Z=None
    for i in DATASETSIDS:
        try:
            z=np.loadtxt(res_path+str(i))
            if Z is not None:
                Z = np.vstack([Z,z])
            else:
                Z=z
            msda_acc =z[0]
            dr_acc   =z[1]
            print '|',string.join(DATANAME[i],'|'),'| %1.3f'%msda_acc,'|','%1.3f'%dr_acc,'|'
        except Exception as e:
            print(e)
    print Z
    print Z.mean(axis=0)




if __name__ == '__main__':
    """
    python exp_run.py DatasetID:noise:feat_type outdir .
    """
    #vectorize_datasets() #This has been run once
    EXPSET = sys.argv[1]
    params=sys.argv[2]
    out_dir     =sys.argv[3]

    print params,out_dir
    parameters = string.split(params,':')

    datasetid = int(parameters[0])
    #noise     = float(parameters[1])
    #feat_type = int(parameters[2])

    #Create the target directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)




    ALPHAS=[0.1,1,50,100,150,200,300]
    ETAS=[0.01,0.1,1,10]
    #ETAS=[0.1]
    max_word=5000
    domain_feat='bow'

    if EXPSET=='AMT':
        DATASETSID=pickle.load(open('amt_datasets_id.pickle'))
        #Xs,Ys,Xt,Yt,dico = dataset_utils.get_da_datasets(DATASETSID[datasetid][0],DATASETSID[datasetid][1],DATASETSID[datasetid][2])
        #TODO Do exp with 5000 Features
        Xs,Ys,Xt,Xtest,Ytest,dico = amazon_exp.get_da_dataset_test(DATASETSID[datasetid][0],DATASETSID[datasetid][1],DATASETSID[datasetid][2],max_words=max_word,feat_type=2)

    elif EXPSET=="20NG":
        Xs,Ys,Xt,Yt,dico_countVect = dataset_utils.get_da_twenty_datasets(datasetid)

        #TODO
        tfidf_trans = TfidfTransformer()
        tfidf_trans.fit(sp.vstack([Xs,Xt]))
        Xs = tfidf_trans.transform(Xs)
        Xt =tfidf_trans.transform(Xt)


        dico={}
        for word,wid in dico_countVect.vocabulary_.iteritems():
            dico[wid]=word

        Xtest=Xt.copy()
        Ytest=Yt



    da_corpus=domain_adaptation_baseline.DACorpus(Xs,Ys,Xt,None)

    y=msda_exp.MsdaDae(da_corpus,LogisticRegression)
    y.fit()
    acc_msda=y.score(Xtest,Ytest)


    target_reg=True
    x=msda_exp.MsdaDomReg(da_corpus,LogisticRegression,alphas=ALPHAS,etas=ETAS,domain_classifier_feat=domain_feat,orthogonal_reg=False,target_reg=target_reg)
    if EXPSET=="20NG":
        x.cross_valid=False
        x.default_clf=LogisticRegression(C=1)

    x.fit()
    acc_domreg = x.score(Xtest,Ytest)
    print "Cross Val Source"
    print x.source_cv_accuracy


    scores =np.array([acc_msda,acc_domreg])
    outname = os.path.join(out_dir,params)
    np.savetxt(outname,scores)

    outname_params=os.path.join(out_dir,params+'.param')
    f=open(outname_params,'w')
    f.write('ALPHAS:'+str(ALPHAS)+'\n')
    f.write('ETAS:'+str(ETAS)+'\n')
    f.write('max_words:'+str(max_word)+'\n')
    f.write('Domain_Feat:'+str(domain_feat)+'\n')
    f.write(str(x.source_cv_accuracy)+'\n')
    f.close()
















