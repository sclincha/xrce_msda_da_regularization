__author__ = 'sclincha'

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
import scipy.linalg.blas
import logging
import pdb

def transform_test(Xtest,W,layer_func=np.tanh,use_bias=True):
    """
    Project Test Data with the learn W and layer function
    :param Xtest: np.array or scipy sparse matrix of shape: nfeatures times nobjects
    :param W: The autoencoder matrix
    :param layer_func: the layer function to encode non linearity
    :return:
    """
    (n_features,n_obs)=Xtest.shape

    if use_bias:
        if sp.issparse(Xtest):
            new_row =sp.csr_matrix(np.ones((1,n_obs)))
            Xb=sp.vstack( [Xtest, new_row])
        else:
            #Xb=np.asfortranarray(np.append(X,np.ones((1,n_obs)),axis=0)) #This operation creates a copy of the original array
            Xb=np.append(Xtest,np.ones((1,n_obs)),axis=0) #This operation creates a copy of the original array


        if sp.issparse(Xtest):
            hx=safe_sparse_dot(W,Xb,dense_output=True)
        else:
            hx =np.dot(W,Xb)
        hx=layer_func(hx)

    else:

        if sp.issparse(Xtest):
            hx=safe_sparse_dot(W,Xtest,dense_output=True)
        else:
            hx =np.dot(W,Xtest)
        hx=layer_func(hx)


    return hx



#TODO factor this in the main code to add the option ...
def mDA_without_bias(X,p,reg_lambda=1e-2,layer_func=np.tanh,Xr=None):
    """

    :param X:
    :param p:
    :param reg_lambda:
    :param layer_func:
    :param Xr:
    :return:
    """
    #Extend observations to add bias features

    (n_features,n_obs)=X.shape
    #print("X_matrix has shape",n_features,n_obs)

    if Xr is not None:
        (r,n_obs_r) =Xr.shape
        Xrdense = Xr.toarray()


    q =(1.0-p)*np.ones((n_features,1))
    #q[-1]=1.0


    if sp.issparse(X):
        S =safe_sparse_dot(X,X.T,dense_output=True)
        #S=np.asarray(S)
    else:
        S = np.dot(X,X.T) #See if we can optimize that in python
        #S = scipy.linalg.blas.dgemm(alpha=1.0, a=Xb, b=Xb, trans_b=True)

    Q = (S)*np.dot(q,q.T)

    #This changes the diagonal of Q
    #TODO Check the faster alternatives
    #Q.flat[::n_features+2] = q*np.diag(S)
    np.fill_diagonal(Q,q*np.diag(S))
    #Q[-1,-1]=S[-1,-1]

    if Xr is not None:
        raise NotImplementedError
    else:
        P = np.tile(q.T,(n_features,1)) * S


    reg = reg_lambda*np.eye(n_features)


    Wt =np.linalg.solve((Q+reg).T,P.T)

    #print "Wt",Wt.shape
    W=Wt.T

    #if filter_W_option>0:
    #    W = filter_W(W,filter_W_option,topk=topk)

    if sp.issparse(X):
        hx=safe_sparse_dot(W,X,dense_output=True)
    else:
        hx =np.dot(W,X)
    hx=layer_func(hx)

    return hx,W

def mDA_domain_regularization(X,p,eta,C,D,IC_inverse,reg_lambda=1e-2,layer_func=np.tanh,Xr=None):
    """

    :param X:
    :param p:
    :param reg_lambda:
    :param layer_func:
    :param Xr:
    :return:
    """



    (n_features,n_obs)=X.shape
    if Xr is not None:
        (r,n_obs_r) =Xr.shape
        Xrdense = Xr.toarray()

    q =(1.0-p)*np.ones((n_features,1))
    #q[-1]=1.0


    if sp.issparse(X):
        S =safe_sparse_dot(X,X.T,dense_output=True)
        #S=np.asarray(S)
    else:
        S = np.dot(X,X.T) #See if we can optimize that in python
        #S = scipy.linalg.blas.dgemm(alpha=1.0, a=Xb, b=Xb, trans_b=True)

    Q = (S)*np.dot(q,q.T)
    np.fill_diagonal(Q,q*np.diag(S))


    if Xr is not None:
        #This fails if Xr is sparse, so convert it first  #Element wise Arrays #
        # P = Xrdense.dot(Xb.T)*np.tile(q.T,(r,1))
        #P = safe_sparse_dot(Xrdense,Xb.T,dense_output=True)*np.tile(q.T,(r,1))
        raise NotImplementError
    else:
        P = np.tile(q.T,(n_features,1)) * S


    reg = reg_lambda*np.eye(n_features)



    if len(D.shape)==1:
        Dv=np.zeros((D.shape[0],1))
        Dv[:,0]=D
    else:
        Dv=D

    #For single category
    P2 =  P -eta*(1.0-p)*safe_sparse_dot(X,Dv,dense_output=True).dot(C.T) #
    #P2 =  P -eta*(1.0-p)*Ptmp
    #Preg=P2
    Preg= np.dot(P2,IC_inverse)
    #Preg= np.dot(IC_inverse.T,P2)


    #Wt =np.linalg.solve((Q+reg).T,Preg.T)
    # Q is symmetric Q.T =Q P is as well
    W =np.linalg.solve(Q.T+reg,Preg)


    #print "Wt",Wt.shape
    W=W.T

    #if filter_W_option>0:
    #    W = filter_W(W,filter_W_option,topk=topk)

    if sp.issparse(X):
        hx=safe_sparse_dot(W,X,dense_output=True)
    else:
        hx =np.dot(W,X)
    hx=layer_func(hx)

    return hx,W




def mDA(X, p, reg_lambda, layer_func=np.tanh, Xr=None, filter_W_option=0, topk=50):
    """

    :param X:
    :param p:
    :param reg_lambda:
    :param layer_func:
    :param Xr:
    :return:
    """
    #Extend observations to add bias features

    (n_features,n_obs)=X.shape
    #print("X_matrix has shape",n_features,n_obs)

    if Xr is not None:
        (r,n_obs_r) =Xr.shape
        Xrdense = Xr.toarray()

    if sp.issparse(X):
        new_row =sp.csr_matrix(np.ones((1,n_obs)))
        Xb=sp.vstack( [X, new_row])
    else:
        Xb=np.append(X,np.ones((1,n_obs)),axis=0) #This operation creates a copy of the original array

    q =(1.0-p)*np.ones((n_features+1,1))
    q[-1]=1.0


    if sp.issparse(X):
        S =safe_sparse_dot(Xb,Xb.T,dense_output=True)
        #S=np.asarray(S)
    else:
        S = np.dot(Xb,Xb.T) #See if we can optimize that in python
        #S = scipy.linalg.blas.dgemm(alpha=1.0, a=Xb, b=Xb, trans_b=True)

    Q = (S)*np.dot(q,q.T)

    #This changes the diagonal of Q
    Q.flat[::n_features+2] = q*np.diag(S)
    #np.fill_diagonal(Q,q*np.diag(S))
    Q[-1,-1]=S[-1,-1]

    if Xr is not None:
        P = safe_sparse_dot(Xrdense,Xb.T,dense_output=True)*np.tile(q.T,(r,1))
    else:
        P = np.tile(q.T,(n_features,1)) * S[0:-1]


    reg = reg_lambda*np.eye(n_features+1)
    reg[-1,-1]=0.0

    Wt =np.linalg.solve((Q+reg).T,P.T)

    #print "Wt",Wt.shape
    W=Wt.T

    if filter_W_option>0:
        W = filter_W(W,filter_W_option,topk=topk)

    if sp.issparse(X):
        hx=safe_sparse_dot(W,Xb,dense_output=True)
    else:
        hx =np.dot(W,Xb)
    hx=layer_func(hx)

    return hx,W


def filter_W(W,filter_code,topk=50):
    """
    Filter the Autoencoder weight matrix for experimental purposes:
    Keep the top words
    Removes negatives values
    :param W:
    :return:
    """
    nw=W.shape[0]
    Wfilter = np.array(W)
    if filter_code<=0:
        #No filter
        return Wfilter
    elif filter_code==1:
        #Keep nb_max per_line
        for wi in xrange(nw):
            sindx=np.argsort(W[wi,:])
            mnk =min(topk,sindx.shape[0])
            null_indx = sindx[:-mnk] #This may fail
            Wfilter[wi,null_indx]=0
        return  Wfilter
    elif filter_code==2:
        Wfilter = np.maximum(Wfilter,0)
        return Wfilter
    elif filter_code==3:
        #Keep Only Negative
        Wfilter = np.minimum(Wfilter,0)
        return Wfilter
    else:
        raise Exception('Unknown filter type')



def expectations_PQ(X,p,reg_lambda,layer_func=np.tanh,Xr=None):
    """

    :param X:
    :param p:
    :param reg_lambda:
    :param layer_func:
    :param Xr:
    :return:
    """


    (n_features,n_obs)=X.shape

    if Xr is not None:
        (r,n_obs_r) =Xr.shape
        Xrdense = Xr.toarray()

    if sp.issparse(X):
        new_row =sp.csr_matrix(np.ones((1,n_obs)))
        Xb=sp.vstack( [X, new_row])
    else:
        Xb=np.append(X,np.ones((1,n_obs)),axis=0) #This operation creates a copy of the original array

    q =(1.0-p)*np.ones((n_features+1,1))
    q[-1]=1.0


    if sp.issparse(X):
        S =safe_sparse_dot(Xb,Xb.T,dense_output=True)

    else:
        S = np.dot(Xb,Xb.T)

    Q = (S)*np.dot(q,q.T)

    #This changes the diagonal of Q
    #TODO Check the faster alternatives
    Q.flat[::n_features+2] = q*np.diag(S)
    #np.fill_diagonal(Q,q*np.diag(S))
    Q[-1,-1]=S[-1,-1]

    if Xr is not None:
        #This fails if Xr is sparse, so convert it first  #Element wise Arrays #
        # P = Xrdense.dot(Xb.T)*np.tile(q.T,(r,1))
        P = safe_sparse_dot(Xrdense,Xb.T,dense_output=True)*np.tile(q.T,(r,1))
    else:
        P = np.tile(q.T,(n_features,1)) * S[0:-1]


    reg = reg_lambda*np.eye(n_features+1)
    reg[-1,-1]=0.0

    return P,Q+reg




def layer_function(hw,layer_type):
    """
    Implements different layer type
    :param hw:
    :param layer_type:
    :return:
    """
    if layer_type==1:
        return np.tanh(hw)

    elif layer_type==2:
        return 1.0/(1.0+np.exp(-hw))

    elif layer_type==3: #Rectified Linear Units

        pos_index = hw >0
        [I,J] = np.nonzero(pos_index)
        K = hw[pos_index]
        H=sp.coo_matrix( (K,(I,J)),shape=hw.shape)
        return H.tocsr()


def get_most_frequent_features(Xdw,topw):
    """
    Return the most frequent features
    :param Xdw: Matrix with document in rows, and word in columns
    :param topw: Number of words to select
    :return:
    """
    Xcsr = Xdw.T.tocsr()
    Xcsr.data = np.ones(Xcsr.data.shape)

    DF = np.array(Xcsr.sum(axis=1)).squeeze()
    sorted_index = np.argsort(DF)

    topk=min(topw,sorted_index.shape)

    return sorted_index[-topk:]


def mSDA(X,noise,nb_layers=5,layer_func=np.tanh,Xr=None,reg_lambda=1e-5):
    """
    This stacks multiples layers of features
    :param X:
    :param noise:
    :param nb_layers:
    :param layer_func:
    :return:
    """
    prevhx=X
    allhx=[]
    Ws=[]

    for layer in range(nb_layers):
        if layer==0 and Xr is not None:
            hx, W = mDA(prevhx, noise, reg_lambda, layer_func, Xr=Xr)
        else:
            hx, W = mDA(prevhx, noise, reg_lambda, layer_func)

        prevhx=hx
        Ws.append(W)
        allhx.append(hx)

    return allhx,Ws

