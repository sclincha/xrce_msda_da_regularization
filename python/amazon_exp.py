import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import scipy.sparse as sp
import scipy
import sklearn.cross_validation

import dataset_utils
import denoising_autoencoders
from termweight import term_weighting
from denoising_autoencoders import layer_function, get_most_frequent_features
import domain_adaptation_baseline
import termweight

import sklearn.feature_extraction.text
import itertools
import pickle



_amazon_datanames=['dvd','books','kitchen','electronics']

def make_ID_datasets_DA(outname='amt_datasets_id.pickle'):
    AD=list(itertools.product(_amazon_datanames,_amazon_datanames,['small','all']))
    AD=filter(lambda x : x[0]!=x[1],AD)
    AD=filter(lambda x : x[0]!=x[1],AD)

    AD=sorted(AD,key= lambda x : x[2])

    DATASETSID={}
    i=0
    for da in AD:
        DATASETSID[i]=da
        i+=1

    f=open(outname,'w')
    pickle.dump(DATASETSID,f)
    f.close()

    return DATASETSID


def count_list_to_sparse_matrix(X_list,dico):
    """
    Transform a list of count in a sparse scipy matrix
    :param X_list:
    :param dico:
    :return:
    """
    ndocs = len(X_list)
    voc_size = len(dico.keys())


    X_spmatrix= sp.lil_matrix((ndocs,voc_size))
    for did,counts in X_list:
        for wid,freq in counts:
            X_spmatrix[did,wid]=freq

    return X_spmatrix.tocsr()


def get_dataset_path(domain_name,exp_type):
    """
    Construct file name
    :param domain_name:
    :param exp_type:
    :return:

    dvd_name   = './dataset/amazon_reviews/processed_acl/dvd/all.review'
    books_name = './dataset/amazon_reviews/processed_acl/books/all.review

    """
    prefix ='./dataset/amazon/processed_acl/'
    if exp_type=='small':
        fname='labelled.review'
    elif exp_type=='all':
        fname='all.review'
    elif exp_type=="test":
        fname='unlabeled.review'

    return os.path.join(prefix,domain_name,fname)



def dataset_small_0():
    """
    Return the DVD-Books dataset
    :return:
    """
    dvd_name = get_dataset_path('dvd','small')
    books_name = get_dataset_path('books','small')

    dataset_list = [dvd_name,books_name]
    datasets,dico = dataset_utils.parse_processed_amazon_dataset(dataset_list)
    L_dvd,Y_dvd = datasets[dvd_name]
    L_books,Y_books = datasets[books_name]

    X_dvd = count_list_to_sparse_matrix(L_dvd,dico)
    X_books = count_list_to_sparse_matrix(L_books,dico)

    return X_dvd,Y_dvd,X_books,Y_books,dico



def get_domain_dataset(domain_name,type='small'):
    """
    Return a single domain dataset from amazon
    :param domain_name:
    :param type:
    :return:
    """
    domain_path = get_dataset_path(domain_name,type)
    datasets,dico = dataset_utils.parse_processed_amazon_dataset([domain_path])

    L_s,Y_s = datasets[domain_path]
    X_s = count_list_to_sparse_matrix(L_s,dico)

    return X_s,Y_s,dico



def get_dataset(source_name,target_name,type='small',max_words=10000):

    if isinstance(type,tuple) and len(type)==2:
        source_path = get_dataset_path(source_name,type[0])
        target_path = get_dataset_path(target_name,type[1])
    else:
        source_path = get_dataset_path(source_name,type)
        target_path = get_dataset_path(target_name,type)

    dataset_list = [source_path,target_path]
    print source_path,target_path

    datasets,dico = dataset_utils.parse_processed_amazon_dataset(dataset_list,max_words=max_words)

    print datasets.keys()

    L_s,Y_s = datasets[source_path]
    L_t,Y_t = datasets[target_path]

    X_s = count_list_to_sparse_matrix(L_s,dico)
    X_t = count_list_to_sparse_matrix(L_t,dico)

    return X_s,Y_s,X_t,Y_t,dico

def get_da_dataset_test(source_name,target_name,type='small',max_words=10000,feat_type=2):
    #Take Small in source and Target
    #Take Big source and small on target

    source_path = get_dataset_path(source_name,type)
    target_path = get_dataset_path(target_name,'small') #Take the two thousands target in any case
    test_path   = get_dataset_path(target_name,'test')


    #Training Sets: source and Adaptation
    dataset_list = [source_path,target_path]
    print source_path,target_path

    datasets,dico = dataset_utils.parse_processed_amazon_dataset(dataset_list,max_words=max_words)
    print datasets.keys()

    #Parse the test set with the current dictionary
    test_datasets = dataset_utils.parse_testset_amazon_dataset([test_path],dico)
    L_s,Y_s = datasets[source_path]
    L_t,Y_t = datasets[target_path]
    #We do not need the labels here in fact
    L_test,Y_test = test_datasets[test_path]

    #I should filter words here and add feature types ....
    Xs = count_list_to_sparse_matrix(L_s,dico)
    Xt = count_list_to_sparse_matrix(L_t,dico)
    Xtest = count_list_to_sparse_matrix(L_test,dico)

    if feat_type==2:
        tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(norm='l2')
        tfidf_transformer.fit(sp.vstack([Xs,Xt]))

        Xs = tfidf_transformer.transform(Xs)
        Xt = tfidf_transformer.transform(Xt)
        Xtest = tfidf_transformer.transform(Xtest)


    return Xs,Y_s,Xt,Xtest,Y_test,dico



def mda_exp(Xs,Ys,Xt,Yt,clf_class=LogisticRegression,noise=0.9,feat_type=2,layer_func=lambda x : layer_function(x,3),
            filter_W_option=0,topk=50,cross_valid=True,use_Xr=True,use_bias=True):
    #Stack Dataset Together
    ndocs_source = Xs.shape[0]
    ndocs_target = Xt.shape[0]

    X_all =sp.vstack([Xs,Xt])
    word_selected = get_most_frequent_features(X_all,5000)

    if feat_type>0:
        X_all = term_weighting(X_all,feat_type=feat_type)

    Xdw_most_frequent=X_all[:,word_selected]



    acc_bow = domain_adaptation_baseline.no_transfer(X_all[:ndocs_source,:],Ys,X_all[ndocs_source:,:],Yt)
    #acc_bow=-1
    #print "BOW Baseline",acc_bow
    if use_Xr:
        hw,W = denoising_autoencoders.mDA(X_all.T, noise, 1e-2, layer_func=layer_func, Xr=Xdw_most_frequent.T, filter_W_option=filter_W_option, topk=topk)
    else:
        if use_bias:
            hw,W = denoising_autoencoders.mDA(X_all.T, noise, 1e-2, layer_func=layer_func, filter_W_option=filter_W_option, topk=topk)
        else:
            print("Without Bias ....")
            hw,W = denoising_autoencoders.mDA_without_bias(X_all.T,noise,1e-2,layer_func=layer_func)


    accuracy = evaluate_mda_features(hw,Ys,Yt,ndocs_source,clf_class,cross_valid=cross_valid)

    return acc_bow,accuracy



def evaluate_mda_features(hw,Ys,Yt,ndocs_source,clf,cross_valid=True,n_jobs=1):
    X_all_dafeatures=hw.T

    Xs_mda = X_all_dafeatures[:ndocs_source,:]
    Xt_mda = X_all_dafeatures[ndocs_source:,:]
    if cross_valid:
        clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, Ys, clf, n_jobs=n_jobs)
    else:
        clf.fit(Xs_mda,Ys)

    Y_pred = clf.predict(Xt_mda)
    print classification_report(Yt,Y_pred)

    accuracy = sklearn.metrics.accuracy_score(Yt,Y_pred)

    return accuracy




def lsi_mda(Xs,Ys,Xt,Yt,clf_class=LogisticRegression,noise=0.9,feat_type=2,layer_func=lambda x : layer_function(x,1),lsi_rank=100):
    #First MDA and then LSI
    #Stack Dataset Together
    ndocs_source = Xs.shape[0]
    ndocs_target = Xt.shape[0]

    X_all =sp.vstack([Xs,Xt])
    X= term_weighting(X_all,feat_type)


    word_selected = get_most_frequent_features(X_all,5000)

    Xdw_most_frequent=X_all[:,word_selected]

    hx,_=denoising_autoencoders.mDA(X.T, noise, layer_func=layer_func, Xr=Xdw_most_frequent.T, reg_lambda=1e-2)
    X_all_dafeatures=hx.T




    Xs_mda = X_all_dafeatures[:ndocs_source,:]
    Xt_mda = X_all_dafeatures[ndocs_source:,:]

    return domain_adaptation_baseline.lsi_transfer(Xs_mda,Ys,Xt_mda,Yt,clf_class,feat_type=0,lsi_rank=lsi_rank)




def make_matlab_dataset(type='small',outname="amazon_small_10p4_features.mat",feat_type=0,max_words=10000):
    dvd_name          = get_dataset_path('dvd',type)
    books_name        = get_dataset_path('books',type)
    electronics_name  =  get_dataset_path('electronics',type)
    kitchen_name      =  get_dataset_path('kitchen',type)

    dataset_list = [dvd_name,books_name,electronics_name,kitchen_name]
    datasets,dico = dataset_utils.parse_processed_amazon_dataset(dataset_list,max_words=max_words)

    L_dvd,Y_dvd = datasets[dvd_name]
    L_books,Y_books = datasets[books_name]
    L_elec,Y_elec = datasets[electronics_name]
    L_kit,Y_kit = datasets[kitchen_name]


    X_dvd = count_list_to_sparse_matrix(L_dvd,dico)
    X_books = count_list_to_sparse_matrix(L_books,dico)
    X_elec = count_list_to_sparse_matrix(L_elec,dico)
    X_kit = count_list_to_sparse_matrix(L_kit,dico)

    if feat_type>0:
        X_dvd   = termweight.term_weighting(X_dvd, feat_type)
        X_books = termweight.term_weighting(X_books, feat_type)
        X_elec  = termweight.term_weighting(X_elec, feat_type)
        X_kit  = termweight.term_weighting(X_kit, feat_type)

    A={"X_dvd" :X_dvd,
        "X_boo":X_books,
        "X_ele":X_elec,
        "X_kit":X_kit,
        "Y_dvd":Y_dvd,
        "Y_boo":Y_books,
        "Y_ele":Y_elec,
        "Y_kit":Y_kit}

    scipy.io.savemat(outname,A)


def make_matlab_all_dataset():
    make_matlab_dataset(type='all',outname="amazon_all_10p4_features.mat")


def make_matlab_small_dataset():
    make_matlab_dataset(type='small',outname="amazon_small_10p4_features.mat")


def make_matlab_small_tfidf():
    make_matlab_dataset(type='small',outname="amazon_small_tfidf_features.mat",feat_type=2)

















