
__author__ = 'sclincha'


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


def print_results_latex(res_path,DID=range(12,24),did_pickle='amt_datasets_id.pickle'):
    """
    Print the result in latex format
    :param res_path:
    :param DID:
    :param did_pickle:
    :return:
    """
    DATANAME =pickle.load(open(did_pickle,'r'))
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
    #max_word=10000
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
















