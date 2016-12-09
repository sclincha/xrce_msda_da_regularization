__author__ = 'sclincha'

import sys
import numpy as np
import string
import sys
sys.path.append('..')
import scipy.sparse as sp
from scipy.io import loadmat
import dataset_utils
import exp_run
import sklearn
import domain_adaptation_baseline
from sklearn.feature_extraction.text import TfidfTransformer

#Do that with multiprocessing to speed up the thing

import multiprocessing
from multiprocessing import Pool
import string
import pickle
import denoising_autoencoders
import itertools
import pdb
from termweight import term_weighting
import commands #for listing all image datasets
import amazon_exp

from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
import sklearn.svm
from sklearn.svm import LinearSVC,SVC
import scipy

import signal
def res_to_latex(Res):
    """
    Export Res of Tuple to latex table
    """
    for r in Res:
        print '%i & %1.3f & %1.3f & %1.2f \\\\' % (r[0],r[3],r[4],r[4]-r[3])

def msda_classifier(Xs,Ys,Xt,Yt,noise=0.9,feat_type=0,score='AUC',clf=None,layer_func=np.tanh,self_learning=False):

    tfidf_trans = TfidfTransformer()

    if feat_type==8:
        Xsn=term_weighting(Xs,feat_type)
        Xtn=term_weighting(Xt,feat_type)
    elif feat_type==2:
        Xsn=tfidf_trans.fit_transform(Xs)
        Xtn=tfidf_trans.transform(Xt)
    else:
        Xsn=Xs
        Xtn=Xt

    #If not Classifier are given, we cross-validate one
    if not(clf):
        clf_cv= domain_adaptation_baseline.cross_validate_classifier(Xsn, Ys, sklearn.linear_model.LogisticRegression, n_jobs=5)
    else:
        clf_cv= clf
        clf_cv =clf.fit(Xsn,Ys)

    no_transfer_acc=clf_cv.score(Xtn,Yt)


    proba = clf_cv.predict_proba(Xtn)

    nclasses=proba.shape[1]
    multiclass = nclasses>2

    if not(multiclass):
        Py_d = proba[:,1]
        vect_prob=np.zeros((Xt.shape[0],1),dtype='f')
        vect_prob[:,0]=Py_d[:]
        #Xt_augment=domain_adaptation_baseline.append_features(Xt,vect_prob)
        Xt_augment=domain_adaptation_baseline.append_features(Xtn,vect_prob)
    else:
        #TODO Try to do it Per Class
        Xt_augment=domain_adaptation_baseline.append_features(Xtn,proba)

    if self_learning:
        Ytpred =clf_cv.predict(Xtn)
        clf_cv= domain_adaptation_baseline.cross_validate_classifier(Xtn, Ytpred, sklearn.linear_model.LogisticRegression)
        no_transfer_acc =clf_cv.score(Xtn,Yt)




    '''
    log_proba = np.log( clf_cv.predict_proba(Xtn) +0.0000001)
    #log_proba = clf_cv.predict_log_proba(Xtn)
    Py_d = log_proba[:,1] -np.log(0.5)
    vect_prob=np.zeros((Xt.shape[0],1),dtype='f')
    vect_prob[:,0]=Py_d[:]
    Xt_augment=domain_adaptation_baseline.append_features(Xtn,vect_prob)
    '''

    hw, W = denoising_autoencoders.mDA(Xt_augment.T, noise, 0.05, layer_func=layer_func)
    h=hw.T

    if not(multiclass):
        #TODO This is dangerous if I swap label 0 and 1 as decision, no ?
        m_score =sklearn.metrics.accuracy_score(Yt,h[:,-1]>0.5)
        #m_score = sklearn.metrics.accuracy_score(Yt,(h[:,-1]-np.log(0.5))>0)

        model_AUC=sklearn.metrics.roc_auc_score(Yt,h[:,-1])
        baseline_AUC=sklearn.metrics.roc_auc_score(Yt,Py_d)
        print "AUC",baseline_AUC,model_AUC

        if score=='AUC':
            return (baseline_AUC,model_AUC)
        else:
            return (no_transfer_acc,m_score)
    else:
        hy_reconstruction=h[:,-nclasses:]
        y_pred  = np.argmax(hy_reconstruction,axis=1)
        m_score = sklearn.metrics.accuracy_score(Yt,y_pred)
        if score=='AUC':
            raise NotImplementedError
        else:
            return (no_transfer_acc,m_score)


def msda_classifier_with_testfeat(Xs,Ys,Xt,Yt,Xt_features,noise=0.9,feat_type=2,target_feat_type=2,score='AUC',
                                  clf=None,layer_func=np.tanh,self_learning=False):

    #Different Feature Weighting for source and target ....

    tfidf_trans = TfidfTransformer()

    if feat_type==8:
        Xsn=term_weighting(Xs,feat_type)
        Xtn=term_weighting(Xt,feat_type)
    elif feat_type==2:
        Xsn=tfidf_trans.fit_transform(Xs)
        Xtn=tfidf_trans.transform(Xt)
    else:
        Xsn=Xs
        Xtn=Xt

    Xtfeatn = Xt_features
    if target_feat_type==2:
        Xtfeatn = term_weighting(Xtfeatn,2)

    #If not Classifier are given, we cross-validate one
    if not(clf):
        clf_cv= domain_adaptation_baseline.cross_validate_classifier(Xsn, Ys, sklearn.linear_model.LogisticRegression, n_jobs=5)
    else:
        clf_cv= clf
        clf_cv =clf.fit(Xsn,Ys)

    no_transfer_acc=clf_cv.score(Xtn,Yt)


    proba = clf_cv.predict_proba(Xtn)

    nclasses=proba.shape[1]
    multiclass = nclasses>2

    if not(multiclass):
        Py_d = proba[:,1]
        vect_prob=np.zeros((Xtfeatn.shape[0],1),dtype='f')
        vect_prob[:,0]=Py_d[:]
        #Xt_augment=domain_adaptation_baseline.append_features(Xt,vect_prob)
        Xt_augment=domain_adaptation_baseline.append_features(Xtfeatn,vect_prob)
    else:
        #TODO Try to do it Per Class
        Xt_augment=domain_adaptation_baseline.append_features(Xtfeatn,proba)

    if self_learning:
        Ytpred =clf_cv.predict(Xtn)
        clf_cv= domain_adaptation_baseline.cross_validate_classifier(Xtfeatn, Ytpred, sklearn.linear_model.LogisticRegression)
        no_transfer_acc =clf_cv.score(Xtfeatn,Yt)



    '''
    log_proba = np.log( clf_cv.predict_proba(Xtn) +0.0000001)
    #log_proba = clf_cv.predict_log_proba(Xtn)
    Py_d = log_proba[:,1] -np.log(0.5)
    vect_prob=np.zeros((Xt.shape[0],1),dtype='f')
    vect_prob[:,0]=Py_d[:]
    Xt_augment=domain_adaptation_baseline.append_features(Xtn,vect_prob)
    '''

    hw, W = denoising_autoencoders.mDA(Xt_augment.T, noise, 0.05, layer_func=layer_func)
    h=hw.T

    if not(multiclass):
        #TODO This is dangerous if I swap label 0 and 1 as decision, no ?
        m_score =sklearn.metrics.accuracy_score(Yt,h[:,-1]>0.5)
        #m_score = sklearn.metrics.accuracy_score(Yt,(h[:,-1]-np.log(0.5))>0)

        model_AUC=sklearn.metrics.roc_auc_score(Yt,h[:,-1])
        baseline_AUC=sklearn.metrics.roc_auc_score(Yt,Py_d)
        print "AUC",baseline_AUC,model_AUC

        if score=='AUC':
            return (baseline_AUC,model_AUC)
        else:
            return (no_transfer_acc,m_score)
    else:
        hy_reconstruction=h[:,-nclasses:]
        y_pred  = np.argmax(hy_reconstruction,axis=1)
        m_score = sklearn.metrics.accuracy_score(Yt,y_pred)
        if score=='AUC':
            raise NotImplementedError
        else:
            return (no_transfer_acc,m_score)




import pdb


def msda_classifier_with_scores(Xt,Yt,St,use_pred=False,noise=0.9,score='AUC',clf=None,layer_func=np.tanh):
    Xtn=Xt

    #no_transfer_acc=clf_cv.score(Xtn,Yt)
    Y_pred = np.argmax(St,axis=1)


    no_transfer_acc = sklearn.metrics.accuracy_score(Yt,Y_pred)


    proba = St
    pred_features =np.zeros(proba.shape,dtype='f')
    print Y_pred[:10]

    for i,cindx in enumerate(Y_pred):
        pred_features[i,cindx]=1.0

    print pred_features[:10,:]


    nclasses=proba.shape[1]
    multiclass = nclasses>2

    if not(multiclass):
        Py_d = proba[:,1]
        vect_prob=np.zeros((Xt.shape[0],1),dtype='f')
        vect_prob[:,0]=Py_d[:]
        #Xt_augment=domain_adaptation_baseline.append_features(Xt,vect_prob)
        if use_pred:
            raise Exception('Use Pred not implemented for binary cases')
        else:
            Xt_augment=domain_adaptation_baseline.append_features(Xtn,vect_prob)
    else:
        #TODO Try to do it Per Class
        if use_pred:
            Xt_augment=domain_adaptation_baseline.append_features(Xtn,pred_features)
        else:
            Xt_augment=domain_adaptation_baseline.append_features(Xtn,proba)



    '''
    log_proba = np.log( clf_cv.predict_proba(Xtn) +0.0000001)
    #log_proba = clf_cv.predict_log_proba(Xtn)
    Py_d = log_proba[:,1] -np.log(0.5)
    vect_prob=np.zeros((Xt.shape[0],1),dtype='f')
    vect_prob[:,0]=Py_d[:]
    Xt_augment=domain_adaptation_baseline.append_features(Xtn,vect_prob)
    '''

    hw, W = denoising_autoencoders.mDA(Xt_augment.T, noise, 0.05, layer_func=layer_func)
    h=hw.T

    if not(multiclass):
        #TODO This is dangerous if I swap label 0 and 1 as decision, no ?
        m_score =sklearn.metrics.accuracy_score(Yt,h[:,-1]>0.5)
        #m_score = sklearn.metrics.accuracy_score(Yt,(h[:,-1]-np.log(0.5))>0)

        model_AUC=sklearn.metrics.roc_auc_score(Yt,h[:,-1])
        baseline_AUC=sklearn.metrics.roc_auc_score(Yt,Py_d)
        print "AUC",baseline_AUC,model_AUC

        if score=='AUC':
            return (baseline_AUC,model_AUC)
        else:
            return (no_transfer_acc,m_score)
    else:
        hy_reconstruction=h[:,-nclasses:]
        y_pred  = np.argmax(hy_reconstruction,axis=1)
        m_score = sklearn.metrics.accuracy_score(Yt,y_pred)
        if score=='AUC':
            raise NotImplementedError
        else:
            return (no_transfer_acc,m_score)





def msda_source_clf_job(paramid):


    parameters = string.split(paramid,':')

    datasetid = int(parameters[0])
    noise     = float(parameters[1])
    feat_type = int(parameters[2])

    DATASETSID=pickle.load(open('amt_datasets_id.pickle'))


    Xs,Ys,Xt,Yt,dico = amazon_exp.get_da_dataset_transductive(DATASETSID[datasetid][0],DATASETSID[datasetid][1],DATASETSID[datasetid][2],max_words=10000)

    #Xs,Ys,Xt_adapt,Xt,Yt,dico = amazon_exp.get_da_dataset_test(DATASETSID[datasetid][0],DATASETSID[datasetid][1],DATASETSID[datasetid][2])
    #get_da_dataset_test

    print Xs.shape
    print Xt.shape
    print Yt.shape

    #score='AUC'
    score='ACC'
    (baseline_score,model_score)= msda_classifier(Xs,Ys,Xt,Yt,noise=noise,feat_type=0,score=score,self_learning=False)
    return (datasetid,noise,feat_type,baseline_score,model_score)




def msda_source_clf_job_20NG(paramid):
    parameters = string.split(paramid,':')

    datasetid = int(parameters[0])
    noise     = float(parameters[1])
    feat_type = int(parameters[2])

    Xs,Ys,Xt,Yt,dico_countVect = dataset_utils.get_da_twenty_datasets_private(datasetid)

 
    tfidf_trans = TfidfTransformer()
    tfidf_trans.fit(Xs)
    Xs = tfidf_trans.transform(Xs)
    Xt =tfidf_trans.transform(Xt)


    dico={}
    for word,wid in dico_countVect.vocabulary_.iteritems():
        dico[wid]=word

    Xtest=Xt.copy()
    Ytest=Yt


    score='ACC'
    (baseline_score,model_score)= msda_classifier(Xs,Ys,Xt,Yt,noise=noise,feat_type=0,score=score,self_learning=False)
    return (datasetid,noise,feat_type,baseline_score,model_score)




def msda_source_clf_job_20NG_testfeatures(paramid):
    parameters = string.split(paramid,':')

    datasetid = int(parameters[0])
    noise     = float(parameters[1])
    feat_type = int(parameters[2])

    Xs,Ys,Xt,Yt,Xt_f,vectorizer,target_vectorizer=dataset_utils.get_da_twenty_datasets_test_feature(datasetid)
    #Xs,Ys,Xt,Yt,dico_countVect = dataset_utils.get_da_twenty_datasets_private(datasetid)

    #TODO
    #tfidf_trans = TfidfTransformer()
    #tfidf_trans.fit(sp.vstack([Xs,Xt]))
    #Xs = tfidf_trans.transform(Xs)
    #Xt =tfidf_trans.transform(Xt)







    print Xs.shape
    print Xt.shape
    print Xt_f.shape

    score='ACC'
    (baseline_score,model_score)=msda_classifier_with_testfeat(Xs,Ys,Xt,Yt,Xt_f,noise=0.9,feat_type=0,target_feat_type=2,score=score,layer_func=np.tanh,self_learning=False)

    #(baseline_score,model_score)= msda_classifier(Xs,Ys,Xt,Yt,noise=noise,feat_type=0,score=score,self_learning=False)
    return (datasetid,noise,feat_type,baseline_score,model_score)

def msda_source_clf_job_AMT_testfeatures(paramid):
    parameters = string.split(paramid,':')

    datasetid = int(parameters[0])
    noise     = float(parameters[1])
    feat_type = int(parameters[2])

    parameters = string.split(paramid,':')

    datasetid = int(parameters[0])
    noise     = float(parameters[1])
    feat_type = int(parameters[2])

    DATASETSID=pickle.load(open('/home/sclincha/AAT/src/DenoisingAutoencoders/datasets_id.pickle'))

    sname,tname=DATASETSID[datasetid][0],DATASETSID[datasetid][1]
    Xs,Ys,Xt,Yt,Xt_f,vectorizer,target_vectorizer=amazon_exp.get_da_dataset_transductive_with_target_feature(sname,tname,DATASETSID[datasetid][2],max_words=10000,feat_type=2)


    #Xs,Ys,Xt,Yt,dico_countVect = dataset_utils.get_da_twenty_datasets_private(datasetid)
    #TODO
    #tfidf_trans = TfidfTransformer()
    #tfidf_trans.fit(sp.vstack([Xs,Xt]))
    #Xs = tfidf_trans.transform(Xs)
    #Xt =tfidf_trans.transform(Xt)

    print Xs.shape
    print Xt.shape
    print Xt_f.shape

    score='ACC'
    (baseline_score,model_score)=msda_classifier_with_testfeat(Xs,Ys,Xt,Yt,Xt_f,noise=0.9,feat_type=0,target_feat_type=2,score=score,layer_func=np.tanh,self_learning=False)

    #(baseline_score,model_score)= msda_classifier(Xs,Ys,Xt,Yt,noise=noise,feat_type=0,score=score,self_learning=False)
    return (datasetid,noise,feat_type,baseline_score,model_score)



def msda_ssl(X,Y,noise=0.9,layer_func=lambda x:x):
    lr=sklearn.linear_model.LogisticRegression()

    rs = sklearn.cross_validation.ShuffleSplit(X.shape[0],train_size=100,test_size=X.shape[0]-100)
    IDX=[(ta_idx,te_idx) for ta_idx,te_idx in rs ]
    train_idx = IDX[0][0]
    test_idx = IDX[0][1]

    lr.fit(X[train_idx],Y[train_idx])
    lr.score(X[test_idx],Y[test_idx])

    proba =lr.predict_proba(X[test_idx])
    Py_d = proba[:,1]
    vect_prob=np.zeros((X[test_idx].shape[0],1),dtype='f')
    vect_prob[:,0]=Py_d[:]
    Xt_augment=domain_adaptation_baseline.append_features(X[test_idx],vect_prob)
    hw, W = denoising_autoencoders.mDA(Xt_augment.T, noise, 0.05, layer_func=layer_func)
    h=hw.T
    m_score =sklearn.metrics.accuracy_score(Y[test_idx],h[:,-1]>0.5)
    #baseline_score=lr.score(X[test_idx],Y[test_idx])
    basleine_score=sklearn.metrics.accuracy_score(Y[test_idx],vect_prob>0.5)

    model_AUC=sklearn.metrics.roc_auc_score(Y[test_idx],h[:,-1])
    baseline_AUC=sklearn.metrics.roc_auc_score(Y[test_idx],Py_d)

    print "AUC",model_AUC,baseline_AUC
    print "ACC",m_score,basleine_score,

    return  basleine_score,m_score





if __name__=="__main__":

    #outname = sys.argv[1]
    #score   = sys.argv[2]

    paramid='1:0.9:0'
    #outname="msda_clf"
    outname="msda_clf_20NG_proba_ACC_AMT_testfeat"

    noise=str(0.9)
    feat =str(2)
    #param_list =[str(i)+':'+noise+':'+feat for i in xrange(28)]
    #param_list =[str(i)+':'+noise+':'+feat for i in xrange(24)]

    #param_list =[str(i)+':'+noise+':'+feat for i in xrange(10)]
    amt_param_list =[str(i)+':'+noise+':'+feat for i in xrange(12,24)]
    #debug_list =[str(i)+':'+noise+':'+feat for i in xrange(2)]


    #res=msda_source_clf_job(paramid)
    #print res

    pool = Pool(12)


    try:
        #res= pool.map(msda_source_clf_job, amt_param_list)
        #res= pool.map(msda_source_clf_job_20NG, param_list)
        #res= pool.map(msda_source_clf_job_20NG_testfeatures, param_list)
        res= pool.map(msda_source_clf_job_AMT_testfeatures, amt_param_list)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit(1)


    print "Res",res
    fo=open(outname,'w')
    pickle.dump(res,fo)
    fo.close()

    for r in res:
        #print '%i,%1.1f,%i,%1.4f,%1.4f' % r
        print '%i:\t%1.4f,%1.4f' % (r[0],r[-2],r[-1])

