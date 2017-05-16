__author__ = 'sclincha'

# -*- coding: utf-8 -*-


import pickle
import string

import numpy as np
import scipy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary as gensim_dico
import amazon_exp


def getFeatureMapping_sklearn(vectorizer):
    """
    Get the Id to Word and Word to Id dictionary from a vectorize
    :param vectorizer:
    :return:
    """
    Voc=vectorizer.get_feature_names()
    dicoWord={}
    Word2Id={}
    I = range(len(Voc))
    for i in I:
        dicoWord[i]=Voc[i]
        Word2Id[Voc[i]]=i

    return dicoWord,Word2Id





def parse_processed_amazon_dataset(FNames,max_words=10000):
    """
    :param fnames: List of filenames to be processed processed
    :return: a dictionary of token to id and a Matrix
    List of (X,Y),(X,Y)
    """


    datasets={}
    dico=gensim_dico()
    print("Parsing",FNames)
    #First pass on document to build dictionary
    for fname in FNames:
        f=open(fname)
        for l in f:
            #print l
            tokens=string.split(l,sep=' ')
            label_string =tokens[-1]

            tokens_list=[]
            for tok in tokens[:-1]:
                ts,tfreq =string.split(tok,':')
                freq=int(tfreq)
                tokens_list += [ts]*freq

            _=dico.doc2bow(tokens_list,allow_update=True)

        f.close()

    ### Preprocessing_options
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    for fname in FNames:
        print fname
        X=[]
        Y=[]
        docid=-1
        f=open(fname)
        for l in f:
            #print l
            tokens=string.split(l,sep=' ')
            label_string =tokens[-1]
            tokens_list=[]
            for tok in tokens[:-1]:
                ts,tfreq =string.split(tok,':')
                freq=int(tfreq)
                tokens_list += [ts]*freq

            count_list=dico.doc2bow(tokens_list,allow_update=False)

            docid+=1

            X.append((docid,count_list))

            #Preprocess Label
            ls,lvalue = string.split(label_string,':')
            #lv=-1
            #print ls,lvalue
            if ls=="#label#":
                if lvalue.rstrip()=='positive':
                    lv=1
                    Y.append(lv)
                elif lvalue.rstrip()=='negative':
                    lv=0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')
            

            #Y.append(lv)

        datasets[fname]=(X,np.array(Y))
        print 'Yvalues'
        print np.bincount(Y)
        f.close()
        del f

    return datasets,dico

def parse_testset_amazon_dataset(FNames,dico):
    """
    Parse the test set with the current dictionary
    """
    datasets={}
    for fname in FNames:
        print fname
        X=[]
        Y=[]
        docid=-1
        f=open(fname)
        for l in f:
            #print l
            tokens=string.split(l,sep=' ')
            label_string =tokens[-1]
            tokens_list=[]
            for tok in tokens[:-1]:
                ts,tfreq =string.split(tok,':')
                freq=int(tfreq)
                tokens_list += [ts]*freq

            count_list=dico.doc2bow(tokens_list,allow_update=False)
            docid+=1

            X.append((docid,count_list))

            #Preprocess Label
            ls,lvalue = string.split(label_string,':')
            #lv=-1
            #print ls,lvalue
            if ls=="#label#":
                if lvalue.rstrip()=='positive':
                    lv=1
                    Y.append(lv)
                elif lvalue.rstrip()=='negative':
                    lv=0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')

        datasets[fname]=(X,np.array(Y))
        print 'Yvalues'
        print np.bincount(Y)
        f.close()
        del f
    return datasets

def get_comp_vs_sci(fit_all=True):
    #TODO Rewrite for source in SourceCateg ....
    #I can make it more generic
    source_categories = ['comp.graphics','comp.os.ms-windows.misc','sci.crypt', 'sci.electronics']
    target_categories= ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'sci.med', 'comp.windows.x','sci.space']


    source_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=source_categories)
    target_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=target_categories)


    vectorizer = CountVectorizer(min_df=3, stop_words="english",max_features=10000)
    if fit_all:
        vectorizer.fit(source_data.data+target_data.data)
    else:
        vectorizer.fit(source_data.data)

    Y_source_atom = source_data.target
    I2S, S2I = getFeatureMapping_sklearn(vectorizer)
    source_category_mapping = dict([(c, i) for (i, c) in enumerate(source_data.target_names)])


    Y_source =np.zeros(Y_source_atom.shape)
    c1 =source_category_mapping['comp.graphics']
    c2 =source_category_mapping['comp.os.ms-windows.misc']

    Y_source[Y_source_atom==c1]=1
    Y_source[Y_source_atom==c2]=1


    Y_target_atom = target_data.target
    target_category_mapping = dict([(c, i) for (i, c) in enumerate(target_data.target_names)])


    Y_target =np.zeros(Y_target_atom.shape)
    c1 =target_category_mapping['comp.sys.ibm.pc.hardware']
    c2 =target_category_mapping['comp.sys.mac.hardware']
    c3 =target_category_mapping['comp.windows.x']

    Y_target[Y_target_atom==c1]=1
    Y_target[Y_target_atom==c2]=1
    Y_target[Y_target_atom==c3]=1


    Xsource=vectorizer.transform(source_data.data)
    Xtarget=vectorizer.transform(target_data.data)

    return Xsource,Y_source,Xtarget,Y_target,vectorizer



def get_rec_vs_talk():
    source_categories_list=[ ['rec.autos','rec.motorcycles'],['talk.politics.guns','talk.politics.misc']]
    target_categories_list=[ ['rec.sport.baseball','rec.sport.hockey'],['talk.politics.mideast','talk.religion.misc']]

    return get_20NG_DomainAdaptation(source_categories_list,target_categories_list)

def get_sci_vs_talk():
    source_categories_list=[ ['sci.electronics','sci.med'],['talk.politics.misc','talk.religion.misc']]
    target_categories_list=[ ['sci.crypt','sci.space'],['talk.politics.guns','talk.politics.mideast']]

    return get_20NG_DomainAdaptation(source_categories_list,target_categories_list)


def get_rec_vs_sci():
    source_categories_list=[ ['sci.space','sci.med'],['rec.autos','rec.sport.baseball']]
    target_categories_list=[ ['sci.crypt','sci.electronics'],['rec.sport.hockey','rec.motorcycles']]

    return get_20NG_DomainAdaptation(source_categories_list,target_categories_list)

def get_comp_vs_rec():
    source_categories_list=[ ['comp.graphics','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware'],['rec.sport.hockey','rec.motorcycles']]
    target_categories_list=[ ['comp.os.ms-windows.misc','comp.windows.x'],['rec.autos','rec.sport.baseball']]

    return get_20NG_DomainAdaptation(source_categories_list,target_categories_list)

def get_comp_vs_talk():
    source_categories_list=[ ['comp.graphics','comp.windows.x','comp.sys.mac.hardware'],['talk.politics.mideast','talk.politics.misc']]
    target_categories_list=[ ['comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware'],['talk.politics.guns','talk.politics.misc']]

    return get_20NG_DomainAdaptation(source_categories_list,target_categories_list)



def get_20NG_DomainAdaptation(source_categories_list,target_categories_list,fit_all=True):
    #source_categories = ['comp.graphics','comp.os.ms-windows.misc','sci.crypt', 'sci.electronics']
    #target_categories= ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'sci.med', 'comp.windows.x','sci.space']

    source_categories = source_categories_list[0] +source_categories_list[1]
    target_categories = target_categories_list[0] +target_categories_list[1]

    source_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=source_categories)
    target_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=target_categories)


    vectorizer = CountVectorizer(min_df=3, stop_words="english",max_features=10000)
    if fit_all:
        vectorizer.fit(source_data.data+target_data.data)
    else:
        vectorizer.fit(source_data.data)

    Y_source_atom = source_data.target
    I2S, S2I = getFeatureMapping_sklearn(vectorizer)
    source_category_mapping = dict([(c, i) for (i, c) in enumerate(source_data.target_names)])


    Y_source =np.zeros(Y_source_atom.shape)
    for source in source_categories_list[0]: #Source Positive
        ci = source_category_mapping[source]
        Y_source[Y_source_atom==ci]=1


    Y_target_atom = target_data.target
    target_category_mapping = dict([(c, i) for (i, c) in enumerate(target_data.target_names)])


    Y_target =np.zeros(Y_target_atom.shape)
    for target in target_categories_list[0]: #Target Positive
        ct=target_category_mapping[target]
        Y_target[Y_target_atom==ct]=1

    #c1 =target_category_mapping['comp.sys.ibm.pc.hardware']
    #c2 =target_category_mapping['comp.sys.mac.hardware']
    #c3 =target_category_mapping['comp.windows.x']
    #Y_target[Y_target_atom==c1]=1
    #Y_target[Y_target_atom==c2]=1
    #Y_target[Y_target_atom==c3]=1


    Xsource=vectorizer.transform(source_data.data)
    Xtarget=vectorizer.transform(target_data.data)

    return Xsource,Y_source,Xtarget,Y_target,vectorizer







def make_comp_vs_sci():
    Xsource,Y_source,Xtarget,Y_target,vectorizer = get_comp_vs_sci()
    save_da_dataset(Xsource,Y_source,Xtarget,Y_target,'20NG_comp_vs_sci.mat')
    pickle.dump(vectorizer,open('20NG_comp_vs_sci_vectorizer.pickle','w'))


def save_da_dataset(Xsource,Ysource,Xtarget,Ytarget,fname):
    A={"X_source" :Xsource,
        "X_target":Xtarget,
        "Y_source":Ysource,
        "Y_target":Ytarget,
    }

    scipy.io.savemat(fname,A)






def get_da_datasets(src_name,target_name,type,max_words=10000):
    """
    Get the Domain adapation Datasets
    get_da_datsets("dvd","books",'small')
    get_da_datsets("comp","sci")
    :param src_name:
    :param target_name:
    :param type:
    :return:
    """
    if src_name=="sci" or target_name=="sci":
        Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_sci()
        if src_name=="sci":
            #comp vs sci
            return Xsource,Y_source,Xtarget,Y_target,vectorizer
        else:
            return Xtarget,Y_target,Xsource,Y_source,vectorizer

    elif src_name=="rec" or target_name=="rec":
        Xsource,Y_source,Xtarget,Y_target,vectorizer=get_rec_vs_talk()
        if src_name=="rec":
            #comp vs sci
            return Xsource,Y_source,Xtarget,Y_target,vectorizer
        else:
            return Xtarget,Y_target,Xsource,Y_source,vectorizer
    else:
        X_s,Y_s,X_t,Y_t,dico = amazon_exp.get_dataset(src_name,target_name,type=type,max_words=max_words)
        return X_s,Y_s,X_t,Y_t,dico




def make_ID_datasets_DA_twenty(outname='twentyda_datasets_id.pickle'):

    DATASETSID={
        0: "comp_vs_sci",
        1: "sci_vs_comp",
        2: "rec_vs_talk",
        3: "talk_vs_rec",
        4: "rec_vs_sci",
        5: "sci_vs_rec",
        6: "sci_vs_talk",
        7: "talk_vs_sci",
        8: "comp_vs_rec",
        9: "rec_vs_comp",
        10: "comp_vs_talk",
        11: "talk_vs_comp",
    }

    f=open(outname,'w')
    pickle.dump(DATASETSID,f)
    f.close()

    return DATASETSID


def get_da_twenty_datasets(datasetid):
    """
    """
    if datasetid ==0:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_sci()

    elif datasetid==1:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_sci()

    elif datasetid ==2:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_rec_vs_talk()

    elif datasetid==3:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_rec_vs_talk()

    elif datasetid ==4:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_rec_vs_sci()

    elif datasetid==5:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_rec_vs_sci()

    elif datasetid ==6:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_sci_vs_talk()

    elif datasetid==7:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_sci_vs_talk()

    elif datasetid ==8:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_rec()

    elif datasetid==9:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_rec()

    elif datasetid ==10:
         Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_talk()

    elif datasetid==11:
         Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_talk()

    else:
        raise Exception('Invalid Dataset ID')

    return Xsource,Y_source,Xtarget,Y_target,vectorizer

#######################################################################################################################
# Limited Access to source data .....

def get_comp_vs_sci_categories_list():
    source_categories_list = [ ['comp.graphics','comp.os.ms-windows.misc'],['sci.crypt', 'sci.electronics']]
    target_categories_list= [['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.windows.x'], ['sci.med','sci.space']]
    return source_categories_list,target_categories_list

def get_rec_vs_talk_categories_list():
    source_categories_list=[ ['rec.autos','rec.motorcycles'],['talk.politics.guns','talk.politics.misc']]
    target_categories_list=[ ['rec.sport.baseball','rec.sport.hockey'],['talk.politics.mideast','talk.religion.misc']]
    return source_categories_list,target_categories_list

    return source_categories_list,target_categories_list
def get_sci_vs_talk_categories_list():
    source_categories_list=[ ['sci.electronics','sci.med'],['talk.politics.misc','talk.religion.misc']]
    target_categories_list=[ ['sci.crypt','sci.space'],['talk.politics.guns','talk.politics.mideast']]
    return source_categories_list,target_categories_list

def get_rec_vs_sci_categories_list():
    source_categories_list=[ ['sci.space','sci.med'],['rec.autos','rec.sport.baseball']]
    target_categories_list=[ ['sci.crypt','sci.electronics'],['rec.sport.hockey','rec.motorcycles']]
    return source_categories_list,target_categories_list

def get_comp_vs_rec_categories_list():
    source_categories_list=[ ['comp.graphics','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware'],['rec.sport.hockey','rec.motorcycles']]
    target_categories_list=[ ['comp.os.ms-windows.misc','comp.windows.x'],['rec.autos','rec.sport.baseball']]
    return source_categories_list,target_categories_list


def get_comp_vs_talk_categories_list():
    source_categories_list=[ ['comp.graphics','comp.windows.x','comp.sys.mac.hardware'],['talk.politics.mideast','talk.politics.misc']]
    target_categories_list=[ ['comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware'],['talk.politics.guns','talk.politics.misc']]
    return source_categories_list,target_categories_list




def get_da_twenty_datasets_private(datasetid):
    if datasetid ==0:
        slist,tlist = get_comp_vs_sci_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid==1:
        tlist,slist = get_comp_vs_sci_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid ==2:
        slist,tlist = get_rec_vs_talk_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid==3:
        tlist,slist = get_rec_vs_talk_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid ==4:
        slist,tlist = get_rec_vs_sci_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid==5:
        tlist,slist = get_rec_vs_sci_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid ==6:
        slist,tlist = get_sci_vs_talk_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)

    elif datasetid==7:
        tlist,slist = get_sci_vs_talk_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)
         #Xtarget,Y_target,Xsource,Y_source,vectorizer=get_sci_vs_talk()

    elif datasetid ==8:
        slist,tlist = get_comp_vs_rec_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)
         #Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_rec()

    elif datasetid==9:
        tlist,slist = get_comp_vs_rec_categories_list()
        return get_20NG_DomainAdaptation(slist,tlist,fit_all=False)
         #Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_rec()

    #elif datasetid ==10:
    #     Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_talk()

    #elif datasetid==11:
    #     Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_talk()

    else:
        raise Exception('Invalid Dataset ID')





def get_20ng_daname(datasetid):
    if datasetid ==0:
        return "comp","sci"
    elif datasetid==1:
        return 'sci','comp'

    elif datasetid ==2:
        return 'rec','talk'

    elif datasetid==3:
        return 'talk','rec'

    elif datasetid ==4:
        return 'rec','sci'

    elif datasetid==5:
        return 'sci','rec'

    elif datasetid ==6:
        return 'sci','talk'

    elif datasetid==7:
        return 'talk','sci'
    elif datasetid ==8:
        return 'comp','rec'
    elif datasetid==9:
        return 'rec','comp'
    else:
        raise Exception('Invalid ID')




def get_20NG_DomainAdaptation_TestFeatures(source_categories_list,target_categories_list):
    #source_categories = ['comp.graphics','comp.os.ms-windows.misc','sci.crypt', 'sci.electronics']
    #target_categories= ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'sci.med', 'comp.windows.x','sci.space']

    source_categories = source_categories_list[0] +source_categories_list[1]
    target_categories = target_categories_list[0] +target_categories_list[1]

    source_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=source_categories)
    target_data = fetch_20newsgroups(data_home='./dataset/sklearn_data', subset='all',remove=('headers', 'footers', 'quotes'),categories=target_categories)


    vectorizer = CountVectorizer(min_df=3, stop_words="english",max_features=10000)
    vectorizer.fit(source_data.data)

    Y_source_atom = source_data.target
    I2S, S2I = getFeatureMapping_sklearn(vectorizer)
    source_category_mapping = dict([(c, i) for (i, c) in enumerate(source_data.target_names)])


    Y_source =np.zeros(Y_source_atom.shape)
    for source in source_categories_list[0]: #Source Positive
        ci = source_category_mapping[source]
        Y_source[Y_source_atom==ci]=1


    Y_target_atom = target_data.target
    target_category_mapping = dict([(c, i) for (i, c) in enumerate(target_data.target_names)])


    Y_target =np.zeros(Y_target_atom.shape)
    for target in target_categories_list[0]: #Target Positive
        ct=target_category_mapping[target]
        Y_target[Y_target_atom==ct]=1

    Xsource=vectorizer.transform(source_data.data)
    Xtarget=vectorizer.transform(target_data.data)


    target_vectorizer = CountVectorizer(min_df=3, stop_words="english",max_features=10000)
    target_vectorizer.fit(target_data.data)
    Xtarget_f = target_vectorizer.fit_transform(target_data.data)

    return Xsource,Y_source,Xtarget,Y_target,Xtarget_f,vectorizer,target_vectorizer



def get_da_twenty_datasets_test_feature(datasetid):
    f = get_20NG_DomainAdaptation_TestFeatures
    if datasetid ==0:
        slist,tlist = get_comp_vs_sci_categories_list()
        return f(slist,tlist)

    elif datasetid==1:
        tlist,slist = get_comp_vs_sci_categories_list()
        return f(slist,tlist)


    elif datasetid ==2:
        slist,tlist = get_rec_vs_talk_categories_list()
        return f(slist,tlist)


    elif datasetid==3:
        tlist,slist = get_rec_vs_talk_categories_list()
        return f(slist,tlist)

    elif datasetid ==4:
        slist,tlist = get_rec_vs_sci_categories_list()
        return f(slist,tlist)


    elif datasetid==5:
        tlist,slist = get_rec_vs_sci_categories_list()
        return f(slist,tlist)

    elif datasetid ==6:
        slist,tlist = get_sci_vs_talk_categories_list()
        return f(slist,tlist)

    elif datasetid==7:
        tlist,slist = get_sci_vs_talk_categories_list()
        return f(slist,tlist)

         #Xtarget,Y_target,Xsource,Y_source,vectorizer=get_sci_vs_talk()
    elif datasetid ==8:
        slist,tlist = get_comp_vs_rec_categories_list()
        return f(slist,tlist)

         #Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_rec()
    elif datasetid==9:
        tlist,slist = get_comp_vs_rec_categories_list()
        return f(slist,tlist)

         #Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_rec()

    #elif datasetid ==10:
    #     Xsource,Y_source,Xtarget,Y_target,vectorizer=get_comp_vs_talk()

    #elif datasetid==11:
    #     Xtarget,Y_target,Xsource,Y_source,vectorizer=get_comp_vs_talk()

    else:
        raise Exception('Invalid Dataset ID')










