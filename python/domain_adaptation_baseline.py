__author__ = 'sclincha'

import numpy as np
from scipy import sparse as spmatrix
import scipy.sparse as sp
import sklearn
import termweight

from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from sklearn.metrics import classification_report



def logeps(x, eps=1e-8):
    return np.log(x + eps)


class DACorpus():

    def __init__(self,Xs,Ys,Xt,Yt):
        self.Xs = Xs   #Source Examples
        self.Ys = Ys   #Target Examples
        self.Xt = Xt   #Target Examples for Adaptation
        self.Yt = Yt   #Target Labels if any

    def has_target_labels(self):
        if self.Yt is not None:
            (self.Yt>-1).sum()>0
        else:
            return False

    def get_labelled_target_index(self):
        if self.Yt is not None:
            Ytindex = self.Yt>-1
            return Ytindex
        else:
            return []

    def get_labelled_instances(self,transform_binary_label='True'):
        if transform_binary_label:
            min_Y_label = self.Ys.min()
            assert(min_Y_label==0)
            Ysr = 2*self.Ys -1
        else:
            Ysr =np.array(self.Ys)

        Ytindex=self.get_labelled_target_index()

        Xt_labelled = self.Xt[Ytindex]
        if transform_binary_label:
            Ytr = 2*self.Yt[Ytindex] -1
        else:
            Ytr =self.Yt[Ytindex]

        Ylabelled_all = np.hstack([Ysr,Ytr])

        if sp.issparse(self.Xs):
            Xlabelled_all =sp.vstack([self.Xs,Xt_labelled])
        else:
            Xlabelled_all =np.vstack([self.Xs,Xt_labelled])

        return Xlabelled_all,Ylabelled_all



#Ad Estimator and transformat
class DomAdapEstimator(object):
    """How do you do semi-supervised domain adaptation"""
    def __init__(self,Xs,Ys,Xt,Yt,source_clf):
        self.Xs = Xs
        self.Ys = Ys
        self.Xt = Xt
        self.Yt=Yt
        self.clf=source_clf

    def fit(self):
        #ByDefault
        #Cross Validate Classifier
        self.clf.fit(self.Xs,self.Ys)

    def transform(self,X):
        return X

    def predict(self,Xtest):
        #Transform Test Data
        #Predict ...
        Xr=self.transform(Xtest)
        return self.clf.predict(Xr)

    def score(self,Xtest,Ytest,print_report=False):
        Ypred=self.predict(Xtest)
        if print_report:
            print(classification_report(Ytest,Ypred))
        accuracy = sklearn.metrics.accuracy_score(Ytest,Ypred)
        return accuracy


class LSIDAE(DomAdapEstimator):

    def __init__(self,Xs,Ys,Xt,Yt,source_clf,lsi_rank,feat_type=5,
                 l2norm=True,use_singular_values=True):
        super(LSIDAE, self).__init__(Xs,Ys,Xt,Yt,source_clf)
        self.k=lsi_rank
        self.V=None
        self.S=None
        self.feat_type=feat_type
        self.use_singular_values=use_singular_values
        self.l2norm=True
        self.use_original_features=False

    def fit(self):
        #Cross Validate Classifier
        ndocs_source = self.Xs.shape[0]
        ndocs_target = self.Xt.shape[0]

        if sp.issparse(self.Xs):
            X_all =sp.vstack([self.Xs,self.Xt])
        else:
            X_all = np.vstack([self.Xs,self.Xt])


        X=termweight.term_weighting(X_all, self.feat_type)
        if sp.issparse(X):
            U,S,V = LSI(X, self.k)
            self.V=V
            self.S=S
            if self.use_singular_values:
                Un= np.dot(U,np.diag(np.sqrt(S)))
            else:
                Un=U

        else:
            #Randomized SVD Here
            svd = TruncatedSVD(n_components=self.k)
            Un = svd.fit_transform(X)

        if self.l2norm:
            sklearn.preprocessing.normalize(Un,'l2',copy=False)

        Us = Un[:ndocs_source,:]
        Ut = Un[ndocs_source:,:]
        #Train Semi-Supervised
        if self.use_original_features is True:
            Xs_n = X_all[:ndocs_source,:]
            Us = append_features(Xs_n,Us,gamma=0.5)

        #return no_transfer(Us,self.Ys,Ut,Yt,clf_class)
        self.clf.fit(Us,self.Ys)

    def transform(self,X):
        if self.V is not None:
            Ul =safe_sparse_dot(X,self.V,dense_output=True)
            if self.l2norm:
                sklearn.preprocessing.normalize(Ul,'l2',copy=False)
            return Ul
        else:
            raise Exception("LSI Latent Vectors not initialized")




def domain_mutual_information(Xsource,Xtarget):
    """
    Implement the domain specific mutual information measure
    cf: Cross-Domain Sentiment Classification via Spectral Feature Alignment.pdf
    :param Xsource:
    :param Xtarget:
    :return:
    """
    Xs_bin = sp.csr_matrix(Xsource)
    Xs_bin.data = np.ones(Xs_bin.data.shape)
    DF_source = 0.5 + np.array(Xs_bin.sum(axis=0)).squeeze()

    Xt_bin = sp.csr_matrix(Xtarget)
    Xt_bin.data = np.ones(Xt_bin.data.shape)
    DF_target = 0.5 +np.array(Xt_bin.sum(axis=0)).squeeze()

    DF_total = DF_source +DF_target

    nd = float(Xsource.shape[0] +Xtarget.shape[0])
    nd_source = Xsource.shape[0]
    nd_target = Xtarget.shape[0]

    P_w = DF_total / nd
    P_notw = 1.0 - P_w

    P_source = nd_source / (nd)
    P_target = 1.0 - P_source

    P_source_w = DF_source / nd  # Fraction  of document with word w in source
    P_target_w = DF_target / nd  # Fraction  of document with word w in target

    P_source_notw = (nd_source - DF_source) / nd

    P_target_notw = (nd_target - DF_target) / nd

    '''
      double G =
        P_v_Du * log (P_v_Du / (P_Du * P_v)) +
        P_Nv_Du * log (P_Nv_Du / (P_Du * P_Nv)) +
        P_v_NDu * log (P_v_NDu / (P_NDu * P_v)) +
        P_Nv_NDu * log (P_Nv_NDu / (P_NDu * P_Nv));
    '''

    G = P_source_w * ( logeps(P_source_w) - logeps(P_w) - logeps(P_source))
    G += P_target_w * ( logeps(P_target_w) - logeps(P_w) - logeps(P_target))
    # I removed this part to try to accound for x>0 in the paper
    #G += P_source_notw * (logeps(P_source_notw) - logeps(P_notw) - logeps(P_source) )
    #G += P_target_notw * ( logeps(P_target_notw) - logeps(P_target) - logeps(P_notw) )


    return G




def spectral_feature_alignement(Xsource,Xtarget,num_di_words=500,nk=100,normalize=True,norm="l2"):

    ndocs_source = Xsource.shape[0]
    ndocs_target = Xtarget.shape[0]

    G=domain_mutual_information(Xsource,Xtarget)
    sindx = np.argsort(G)

    domain_independant_index  = sindx[:num_di_words]
    domain_specific_index = sindx[num_di_words:]

    X_all = sp.vstack([Xsource,Xtarget])
    #Correlation Matrix

    F=X_all.T
    if normalize:
        F = sklearn.preprocessing.normalize(F,norm)

    Q=F * F.T
    nw=Q.shape[0]


    M  = Q[domain_specific_index,:]
    M1 = M[:,domain_independant_index]


    A=sp.bmat([[None,M1],[M1.T,None]])

    d=A.sum(axis=1)
    D=sp.diags(np.asarray(1/np.sqrt(d)).squeeze(),0)
    L=D*A*D

    U =sp.linalg.eigs(L,nk)
    Wreal = np.real(U[1])

    #Ureal = np.real(U[0])
    #x[domain_specific_index]*Wreal[:nw_num_diwords]

    Wds = Wreal[:(nw-num_di_words),]
    #TODO  normalize Wds Or Xds ?
    sklearn.preprocessing.normalize(Wds)
    Xds = X_all[:,domain_specific_index]*Wds

    Xs_ds =  Xds[:ndocs_source,]
    Xt_ds =  Xds[ndocs_source:,]

    #Xs_sfa = sp.hstack([Xsource,Xs_ds])
    #Xt_sfa = sp.hstack([Xtarget,Xt_ds])

    return Xs_ds,Xt_ds


def append_features(Xoriginal,X_newfeatures,gamma=1.0):
    if sp.issparse(Xoriginal):
        Xnf = sp.hstack([Xoriginal, gamma*X_newfeatures])
        return Xnf
    else:
        #Dense Matrix
        Xnf = np.hstack( [Xoriginal,gamma*X_newfeatures])
        return Xnf





def no_transfer(Xs,Ys,Xt,Yt,clf_class=LogisticRegression):
    """
    Baseline for no transfer technique
    :param Xs: source dataset
    :param Ys: labels
    :param Xt: target dataset
    :param Yt: labels
    :param clf_class: classifier class : LogisticRegresion |LinearSVC
    :return:
    """
    clf = cross_validate_classifier(Xs,Ys,clf_class)
    Y_pred = clf.predict(Xt)
    print classification_report(Yt,Y_pred)

    accuracy = sklearn.metrics.accuracy_score(Yt,Y_pred)
    return accuracy



def lsi_transfer(Xs,Ys,Xt,Yt,clf_class=LogisticRegression,feat_type=2,lsi_rank=100,use_original_features=False,use_singular_values=True,l2normalization=False):

    ndocs_source = Xs.shape[0]
    ndocs_target = Xt.shape[0]

    if sp.issparse(Xs):
        X_all =sp.vstack([Xs,Xt])
    else:
        X_all = np.vstack([Xs,Xt])


    X=termweight.term_weighting(X_all, feat_type)
    if sp.issparse(X):
        U,S,V = LSI(X, lsi_rank)
        if use_singular_values:
            Un= np.dot(U,np.diag(np.sqrt(S)))
        else:
            Un=U

    else:
        #Randomized SVD Here
        svd = TruncatedSVD(n_components=lsi_rank)
        Un = svd.fit_transform(X)

    if l2normalization:
        Un =sklearn.preprocessing.normalize(Un)

    Us = Un[:ndocs_source,:]
    Ut = Un[ndocs_source:,:]

    if use_original_features is True:
        Xs_n = X_all[:ndocs_source,:]
        Xt_n = X_all[ndocs_source:,:]

        Us = append_features(Xs_n,Us,gamma=0.5)
        Ut = append_features(Xt_n,Ut,gamma=0.5)

    return no_transfer(Us,Ys,Ut,Yt,clf_class)




def sfa_transfer(Xs,Ys,Xt,Yt,clf_class=LogisticRegression,feat_type=2,nclusters=100,num_di_words=1000):

    Xsn = termweight.term_weighting(Xs, feat_type)
    Xtn = termweight.term_weighting(Xt, feat_type)
    Xs_sfa,Xt_sfa=spectral_feature_alignement(Xsn,Xtn,num_di_words,nk=nclusters,normalize=True)

    Xs_tfidf = termweight.term_weighting(Xs, feat_type)
    Xt_tfidf = termweight.term_weighting(Xt, feat_type)

    Xs_all =append_features(Xs_tfidf,Xs_sfa,gamma=0.5)
    Xt_all =append_features(Xt_tfidf,Xt_sfa,gamma=0.5)

    acc =no_transfer(Xs,Ys,Xt,Yt,clf_class=clf_class)
    acc_sfa =no_transfer(Xs_all,Ys,Xt_all,Yt,clf_class=clf_class)

    return acc,acc_sfa




def cross_validate_classifier(X,Y,clf_class,score='accuracy',ncv=5,n_jobs=1,verbose=0):
    """
    Cross Validate SVM or Logistic Regression
    :param X:
    :param Y:
    :param clf_class:
    :param score:
    :return:
    """
    tuned_parameters = [{'C': [0.0001,0.001,0.1,1,10]}]

    if verbose:
        print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(clf_class(), tuned_parameters, cv=ncv, scoring=score,n_jobs=n_jobs)
    clf.fit(X, Y)
    if verbose:
        print(clf.best_params_)
        print(clf.grid_scores_)
    return clf


def LSI(X, topk, norm=None):
    """
    Example
    U,S,V= dataset_utils.LSI(X,20,'l2')
    """
    [U, S, Vt] = spmatrix.linalg.svds(X, topk)
    if norm:
        Un = preprocessing.normalize(U, norm=norm)
        V = Vt.T
        Vn = preprocessing.normalize(V, norm=norm)
        return Un, S, Vn
    else:
        return U, S, Vt.T