
__author__ = 'sclincha'
"""
Functions from Information Retrieval weighting and similarities
"""

import numpy as np
import scipy.sparse as spmatrix
import pdb
from sklearn.utils.extmath import safe_sparse_dot
import sklearn

import bisect


def row_norms(X):
    """
    Return the row norms of sparse matrix array
    """
    return np.squeeze(np.asarray(X.sum(axis=1)))


def col_norms(X):
    """
    Return the row norms of sparse matrix array
    """
    return np.squeeze(np.asarray(X.sum(axis=0)))


def tfn2_norm(Xdw, c=1.0):
    '''
    Implement Amati DFR Term Frequency Normalization 2
    X is supposed to be sparse matrix ndocsxnwords of word count to get an appropriate
    '''
    #It has been normalized
    ndocs = Xdw.shape[0]
    doc_length = Xdw.sum(axis=1)

    avgdl = np.mean(doc_length)
    renorm = np.log(1 + c * avgdl / (doc_length + 0.5))

    e = np.array(renorm).squeeze()
    Dn = spmatrix.diags(e, 0)
    #Y=Dn*X
    #return Y
    return Dn


def idf_weighting(X):
    """
    Compute IDF
    Y=X*Dw
    """
    (ndocs, nw) = X.shape
    #Copy and Binarize data
    Xc = spmatrix.csc_matrix(X)
    Xc.data = Xc.data / Xc.data  #Binarize

    Doc_Frequency = 0.5 + Xc.sum(axis=0)
    IDF = np.log(ndocs) - np.log(np.squeeze(np.asarray(Doc_Frequency)))
    Dw = spmatrix.diags(IDF, 0)
    #Y=X*Dw
    return Dw

def doc_frequency(Xdw):
    Xcsr = Xdw.T.tocsr()
    Xcsr.data = np.ones(Xcsr.data.shape)

    DF = np.array(Xcsr.sum(axis=1)).squeeze()
    return DF


def loglog_transform(Xcount,c,type):
    """
    :param Xcount:
    :return:
    """
    (nd,nfeat)=Xcount.shape
    dl = np.asarray(Xcount.sum(axis=1))
    dl=dl.squeeze()
    #print dl
    avg_dl=dl.mean()
    r_j     = np.bincount(Xcount.indices) /float(nd)
    [I,J,K] =spmatrix.find(Xcount)

    eps = 1e-8
    eta = 1.2 #for the q-logarithm
    tfn = K*np.log(1.0+c*avg_dl/dl[I])
    if type==1:
        K2= np.log( (r_j[J] +tfn) / r_j[J])

    elif type==2:
        ee =tfn/(1+tfn)
        K2 = -np.log( (eps + np.power(r_j[J],ee)-r_j[J] ) / (1.0+ eps -r_j[J]) )

    elif type==3:
        Z  = r_j[J]/(r_j[J] + tfn)
        K2 = -1.0/(1-eta)*(np.power(Z,1-eta) -1.0)

    Xtrans = spmatrix.coo_matrix( (K2,(I,J)),shape=(nd,nfeat))

    return Xtrans.tocsr()

def filterMatrix(Xdw, df_filter=10):
    """
    Filter columns of the Xdocword matrix
    :param Xdw: Bag-of-Word Matrix: rows are documents, cols words (Xdw is an abreviation of Xdocword)
    :param df_filter: the document frequency filter under which words are ignored
    :return: a coo_matrix (cf scipy.sparse)
    """
    #Preprocess Sparse Matrix
    Xcsr = Xdw.T.tocsr()
    Xcsr.data = np.ones(Xcsr.data.shape)

    DF = np.array(Xcsr.sum(axis=1)).squeeze()
    windx = np.nonzero(DF > df_filter)[0].tolist()

    row = []
    col = []
    for wid in windx:
        #Slow Way
        #xw = Xcsr[wid]
        #row += [wid]*xw.nnz
        #col += xw.indices.tolist()

        colind = Xcsr.indices[Xcsr.indptr[wid]:Xcsr.indptr[wid + 1]].tolist()
        row += [wid] * len(colind)
        col += colind
        #pdb.set_trace()
    val = [1.0] * len(row)

    Xcoo = spmatrix.coo_matrix((val, (row, col)), shape=Xcsr.shape)
    return Xcoo






def term_weighting(Xcount,feat_type,c=1.0):
    """

    :param Xcount: Matrices of doc x terms of term count
    :return: a term weighted matrix
    """

    if feat_type ==0: #TF
        Xtfn = Xcount.astype('d')
        return Xtfn

    elif feat_type ==1: # Binary
        Xtfn = Xcount.astype('d')
        Xtfn = spmatrix.csr_matrix(Xtfn)
        Xtfn.data = np.ones(Xtfn.data.shape)
        Xtfn =sklearn.preprocessing.normalize(Xtfn,norm='l2')
        return Xtfn

    elif feat_type==2:  #TFIDF Classique
        tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(norm='l2')

        #idf = ir_wsim.idf_weighting(Xcount)
        #Xidf= X * idf
        #Xtfn =sklearn.preprocessing.normalize(Xidf,norm='l2')
        #TODO Add LOGLOG Transformer as type
        Xtfn = tfidf_transformer.fit_transform(Xcount)
        return Xtfn

    elif feat_type ==3: #LogLogistic
        Xtfn = loglog_transform(Xcount,c,type=1)
        return Xtfn
    elif feat_type ==4: #SPL ...
        Xtfn = loglog_transform(Xcount,c,type=2)
        return Xtfn
    elif feat_type ==5: #Q-LOGARITHM
        Xtfn = loglog_transform(Xcount,c,type=3)
        return Xtfn
    elif feat_type ==6: #tf_DFR*idf
        idf = idf_weighting(X)
        dl  = tfn2_norm(X,1.0)
        Xtfn=  (dl*X) * idf
        return Xtfn
    elif feat_type==7: #QLN_+ norm L2
        Xtfn = loglog_transform(Xcount,c,type=3)
        Xtfn = sklearn.preprocessing.normalize(Xtfn,norm='l2')
        return Xtfn
    elif feat_type==8: #TF with L2 norm
        Xcount = Xcount.astype('d')
        Xtfn =sklearn.preprocessing.normalize(Xcount,norm='l2')
        return Xtfn

    else:
        raise Exception('Unknown feature type:'+str(feat_type))