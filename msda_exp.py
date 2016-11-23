import numpy as np
from scipy import sparse as sp
import sklearn
from sklearn.linear_model import RidgeClassifier, LogisticRegression,RidgeClassifierCV,ridge_regression
from sklearn.metrics import classification_report
from sklearn.utils.extmath import  safe_sparse_dot
import denoising_autoencoders
from denoising_autoencoders import layer_function,mDA,get_most_frequent_features
import domain_adaptation_baseline
from termweight import term_weighting


__author__ = 'sclincha'



def msda_exp_testset(Xs,Ys,Xt,Xtest,Ytest,clf_class=LogisticRegression,noise=0.9,feat_type=0,layer_func=np.tanh,
            filter_W_option=0,topk=50,cross_valid=True,use_Xr=True,use_bias=True):
    #Stack Dataset Together
    ndocs_source = Xs.shape[0]
    ndocs_target = Xt.shape[0]

    X_all =sp.vstack([Xs,Xt])
    #TODO RLU
    word_selected = get_most_frequent_features(X_all,5000)

    Xdw_most_frequent=X_all[:,word_selected]



    #acc_bow = domain_adaptation_baseline.no_transfer(X_all[:ndocs_source,:],Ys,X_all[ndocs_source:,:],Yt)
    acc_bow=-1
    print "BOW Baseline",acc_bow
    if use_Xr:
        hw,W = denoising_autoencoders.mDA(X_all.T, noise, 1e-2, layer_func=layer_func, Xr=Xdw_most_frequent.T, filter_W_option=filter_W_option, topk=topk)
    else:
        if use_bias:
            hw,W = denoising_autoencoders.mDA(X_all.T, noise, 1e-2, layer_func=layer_func, filter_W_option=filter_W_option, topk=topk)
        else:
            print "Without Bias ...."
            hw,W = denoising_autoencoders.mDA_without_bias(X_all.T,noise,1e-2,layer_func=layer_func)


    X_all_dafeatures=hw.T


    Xs_mda = X_all_dafeatures[:ndocs_source,:]
    Xtest_msda = denoising_autoencoders.transform_test(Xtest.T,W,layer_func=layer_func,use_bias=use_bias).T

    #Train
    clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, Ys, clf_class)
    Y_pred = clf.predict(Xtest_msda)

    print classification_report(Ytest,Y_pred)

    accuracy = sklearn.metrics.accuracy_score(Ytest,Y_pred)

    return acc_bow,accuracy


class MsdaDae(domain_adaptation_baseline.DomAdapEstimator):
    """
    Marginalized (s) Denoising Autoencoders (Domain Adaptation Estimator)
    """
    def __init__(self,da_corpus,clf,noise=0.9,layer_func=np.tanh,use_Xr=True,use_bias=True):
        super(MsdaDae, self).__init__(da_corpus.Xs,da_corpus.Ys,da_corpus.Xt,da_corpus.Yt,clf)
        self.noise=noise
        self.layer_func=layer_func
        self.use_Xr=use_Xr
        self.use_bias=use_bias
        self.W=None
        #Question Store the Cs ..
        print "Warning Ignoring Classifier"

    def fit(self):
        ndocs_source = self.Xs.shape[0]

        if sp.issparse(self.Xs):
            X_all =sp.vstack([self.Xs,self.Xt])
        else:
             X_all    = np.vstack([self.Xs,self.Xt])

        if self.use_Xr:
            word_selected = get_most_frequent_features(X_all,5000)
            Xdw_most_frequent=X_all[:,word_selected]
            hw,W = denoising_autoencoders.mDA(X_all.T, self.noise, 1e-2, layer_func=self.layer_func, Xr=Xdw_most_frequent.T)
        else:
            if self.use_bias:
                hw,W = denoising_autoencoders.mDA(X_all.T, self.noise, 1e-2, layer_func=self.layer_func)
            else:
                print "Without Bias ...."
                hw,W = denoising_autoencoders.mDA_without_bias(X_all.T,self.noise,1e-2,layer_func=self.layer_func)


        X_all_dafeatures=hw.T
        Xs_mda = X_all_dafeatures[:ndocs_source,:]
        #Train
        self.clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, self.Ys, LogisticRegression)
        self.W=W


    def transform(self,Xtest):
        if self.W is not None:
            Xtest_msda = denoising_autoencoders.transform_test(Xtest.T,self.W,layer_func=self.layer_func,use_bias=self.use_bias).T
            return Xtest_msda
        else:
            raise Exception("Autoencoders not fitted")


class MsdaDomReg(domain_adaptation_baseline.DomAdapEstimator):
    def __init__(self,da_corpus,clf,noise=0.9,layer_func=np.tanh,alphas=[1e-3,0.1,1,100,500],etas=[1e-3,1e-2,0.1,1,10],
                 domain_classifier_feat='bow',method_cv='normal',orthogonal_reg=False,target_reg=False,source_reg=False):
        super(MsdaDomReg, self).__init__(da_corpus.Xs,da_corpus.Ys,da_corpus.Xt,da_corpus.Yt,clf)
        self.noise=noise
        self.layer_func=layer_func
        self.W=None
        self.alphas=alphas
        self.etas=etas
        self.C=None
        self.D_vector=None
        self.D_matrix=None

        self.source_cv_accuracy=[]
        self.domain_classifier_feat=domain_classifier_feat
        self.method_cv=method_cv
        self.orthogonal_reg=orthogonal_reg
        self.target_reg=target_reg
        self.source_reg=source_reg

        print "Warning Ignoring Classifier"
        self.init_domain_variable()
        self.cross_valid=True
        self.default_clf=LogisticRegression()


    def init_domain_variable(self):
        """
        Init the D_vector and D_matrix
        :return:
        """
        #This could be function in DA_corpus ....
        if sp.issparse(self.Xs):
            X_all   = sp.vstack([self.Xs,self.Xt])
        else:
            X_all    = np.vstack([self.Xs,self.Xt])

        self.D_vector = np.array([-1]*(self.Xs.shape[0])+[1]*(self.Xt.shape[0]))
        self.D_matrix=np.zeros((X_all.shape[0],2))
        self.D_matrix[:self.Xs.shape[0],0]=1
        self.D_matrix[self.Xs.shape[0]:,1]=1
        #if self.orthogonal_reg:
            #self.D_vector[:]=0.0
            #self.D_matrix[:,:]=0.0



    def fit_domain_classifier(self,alpha,X_all):
        #TODO Merge with the following method
        #TODO This work only for binary classifier
        C=ridge_regression(X_all,self.D_vector,alpha)
        #C=C.reshape((C.shape[0],1))
        self.C=C.reshape((C.shape[0],1))


    def fit_domain_classifier_cv(self,X_all):
        ridge_clf = RidgeClassifierCV(alphas=self.alphas)
        if self.domain_classifier_feat=='bow':
            ridge_clf.fit(X_all, self.D_vector) #Before it was X_all
        elif self.domain_classifier_feat=='msda':
            h0,_=denoising_autoencoders.mDA_without_bias(X_all.T,0.9,layer_func=self.layer_func,Xr=None)
            ridge_clf.fit(h0.T, self.D_vector) #Before it was X_all
        else:
            raise Exception('Invalid Feature type for Domain Classifier')
        #ridge_clf.fit(X_a ll, self.D_vector) #Before it was X_all
        self.C=ridge_clf.coef_.T

        print "Best alpha:", ridge_clf.alpha_
        #TODO Add Options for Sample Weight for element in target and source
        #TODO Works only for Binary Class now
        z=ridge_regression(Xlabelled_all,Ylabelled_all,alpha)
        #z=ridge_regression(self.Xs,self.Ys,alpha)
        self.ridge_class_coeff=z.reshape((z.shape[0],1))



    def fit_with_param(self,alpha,eta):
        ns      = self.Xs.shape[0]

        if sp.issparse(self.Xs):
            X_all   = sp.vstack([self.Xs,self.Xt])
        else:
             X_all    = np.vstack([self.Xs,self.Xt])

        self.fit_domain_classifier(alpha,X_all)
        Z=np.dot(self.C,self.C.T)


        Dvector = np.array(self.D_vector)
        if self.orthogonal_reg:
            eta =-eta
            #Changing Dvector only in msda else C become 0
            Dvector[:]=0.0

        if self.target_reg:
            eta =-eta
            #Changing Dvector only in msda else C become 0
            Dvector[:]=1.0

        if self.source_reg:
            Dvector[:]=-1.0

        Icc= np.linalg.inv(np.eye(X_all.shape[1])-eta*Z)

        print("Computing msda with Domain Regularizer")
        hx,W=denoising_autoencoders.mDA_domain_regularization(X_all.T,self.noise,eta,self.C,Dvector,Icc)
        #Check it is the same link function ....
        Xs_mda = hx.T[:ns,:]
        print("CrossValidating Source Classifier")


        da_corpus = domain_adaptation_baseline.DACorpus(self.Xs,self.Ys,self.Xt,self.Yt)
        Ytindex=da_corpus.get_labelled_target_index()

        if Ytindex is not []:
            Ytlabels= self.Yt[Ytindex]
            Yclf = np.hstack([self.Ys,Ytlabels])
            #sample_weight = np.hstack([np.ones(self.Ys.shape[0]),2*np.ones(Ytlabels.shape[0])])
            Xs_mda = hx.T[:ns,:]
            Xt_mda = hx.T[ns:,:]
            Xtrain =np.vstack([Xs_mda,Xt_mda[Ytindex]])
        else:
            Yclf = self.Ys
            Xtrain= hx.T[:ns,:]

        self.clf = domain_adaptation_baseline.cross_validate_classifier(Xtrain, Yclf, LogisticRegression, score='accuracy', ncv=3, n_jobs=3,verbose=1)
        #self.clf = exp_run.cross_validate_classifier(Xs_mda,self.Ys,LogisticRegression,score='accuracy',ncv=3,n_jobs=3)
        self.W=W



    def cross_val_source(self):
        ns      = self.Xs.shape[0]
        if sp.issparse(self.Xs):
            X_all   = sp.vstack([self.Xs,self.Xt])
        else:
            X_all   = np.vstack([self.Xs,self.Xt])

        ACC=[]



        for alpha in self.alphas:
            self.fit_domain_classifier(alpha,X_all)
            #print "Normalizing C" #
            #self.C = sklearn.preprocessing.normalize(self.C)
            Z=np.dot(self.C,self.C.T)
            for eta in self.etas:
                    Dvector = np.array(self.D_vector)
                    if self.orthogonal_reg:
                        eta =-eta
                        #Changing Dvector only in msda else C become 0
                        Dvector[:]=0.0

                    if self.target_reg:
                        eta =-eta
                        #Changing Dvector only in msda else C become 0
                        Dvector[:]=1.0

                    Icc= np.linalg.inv(np.eye(X_all.shape[1])-eta*Z)
                    print "Computing msda with Domain Regularizer"
                    hx,W=denoising_autoencoders.mDA_domain_regularization(X_all.T,self.noise,eta,self.C,Dvector,Icc)
                    #Check it is the same link function ....
                    Xs_mda = hx.T[:ns,:]

                    print "CV Source Classifier"
                    if self.cross_valid:
                        self.clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, self.Ys, LogisticRegression, score='accuracy', ncv=3, n_jobs=3,verbose=0)
                    else:
                        self.clf = self.default_clf.fit(Xs_mda,self.Ys)
                    #print clf.grid_scores_
                    #max_score= clf.best_score_
                    #clf.fit(Xs_mda,Ys)
                    source_score= sklearn.cross_validation.cross_val_score(self.clf,Xs_mda,y=self.Ys,cv=3,n_jobs=3).mean()
                    ACC.append((source_score,(alpha,eta)))

        #Then find max
        sorted_ACC=sorted(ACC)
        self.source_cv_accuracy=sorted_ACC
        print sorted_ACC

        best_alpha=sorted_ACC[-1][1][0]
        best_eta=sorted_ACC[-1][1][1]

        Dvector = np.array(self.D_vector)
        if self.target_reg:
                #Changing Dvector only in msda else C become 0
                Dvector[:]=1.0
        #The best eta should be a negative value ...
        # and Dvector already set


        #Print Refit the model
        self.fit_domain_classifier(best_alpha,X_all)
        Z=np.dot(self.C,self.C.T)
        Icc= np.linalg.inv(np.eye(X_all.shape[1])-best_eta*Z)
        print "Computing msda with Domain Regularizer"
        hx,W=denoising_autoencoders.mDA_domain_regularization(X_all.T,self.noise,best_eta,self.C,Dvector,Icc)
        Xs_mda = hx.T[:ns,:]
        self.W=W

        if self.cross_valid:
            self.clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, self.Ys, LogisticRegression, score='accuracy', ncv=3, n_jobs=3,verbose=0)
        else:
            self.clf = self.default_clf.fit(Xs_mda,self.Ys)




    def reverse_cross_val(self,validation_size=0.1):

        ndocs=self.Xs.shape[0]
        #Split Source Validation ...

        rs = sklearn.cross_validation.ShuffleSplit(ndocs, n_iter=1,test_size=validation_size)
        for train_idx,val_idx in rs:
            TRAIN_IDX=train_idx
            VAL_IDX=val_idx

        Xs_train =self.Xs[TRAIN_IDX,:]
        Ys_train =self.Ys[TRAIN_IDX]

        Xs_val   =self.Xs[VAL_IDX,:]
        Ys_val   =self.Ys[VAL_IDX]

        D_vector_source = self.D_vector[:ndocs]
        D_vector_target = self.D_vector[ndocs:]
        D_vector = np.hstack([D_vector_source[TRAIN_IDX],D_vector_target])

        if sp.issparse(self.Xs):
            X_all   = sp.vstack([Xs_train,self.Xt])
        else:
             X_all    = np.vstack([Xs_train,self.Xt])


        ns =Xs_train.shape[0]
        ACC=[]
        for alpha in self.alphas:
            #Fit Domain Classifier
            ridge_clf = RidgeClassifier(alpha=alpha)
            ridge_clf.fit(X_all, D_vector) #Before it was X_all
            self.C=ridge_clf.coef_.T
            #self.fit_domain_classifier(alpha,X_all)
            Z=np.dot(self.C,self.C.T)
            for eta in self.etas:


                DvectorReg = np.array(D_vector)
                if self.orthogonal_reg:
                    eta =-eta
                    #Changing Dvector only in msda else C become 0
                    DvectorReg[:]=0.0

                if self.target_reg:
                    eta =-eta
                    #Changing Dvector only in msda else C become 0
                    DvectorReg[:]=1.0

                Icc= np.linalg.inv(np.eye(X_all.shape[1])-eta*Z)
                print "Computing msda with Domain Regularizer"
                hx,W=denoising_autoencoders.mDA_domain_regularization(X_all.T,self.noise,eta,self.C,DvectorReg,Icc)
                #Check it is the same link function ....
                Xs_mda = hx.T[:ns,:]
                Xt_mda = hx.T[ns:,:]
                print "CV Source Classifier"
                self.clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, Ys_train, LogisticRegression, score='accuracy', ncv=3, n_jobs=3)
                #print self.clf.grid_scores_
                #max_score= clf.best_score_
                #clf.fit(Xs_mda,Ys)
                #Only the classifier needs to be relearned ... as the feature representation does not change ....
                #Do the reverse validation ... Here

                #I don't reuse Xs for learning ...
                Ypred = self.clf.predict(Xt_mda)
                Y_all = np.hstack([Ys_train,Ypred])
                #Retrain on Prediction and Target Features ...
                #It is not the proper reverse validation mechanism as here I cross validated ....
                #I should again cross val on the C here
                #self.clf =exp_run.cross_validate_classifier(Xt_mda,Ypred,LogisticRegression,score='accuracy',ncv=3,n_jobs=3)
                self.clf = domain_adaptation_baseline.cross_validate_classifier(hx.T, Y_all, LogisticRegression, score='accuracy', ncv=3, n_jobs=3)

                Xsval_msda = denoising_autoencoders.transform_test(Xs_val.T,W,layer_func=self.layer_func,use_bias=False).T
                validation_score = self.clf.score(Xsval_msda,Ys_val)
                #source_score= sklearn.cross_validation.cross_val_score(self.clf,Xs_mda,y=self.Ys,cv=3).mean()
                ACC.append((validation_score,(alpha,eta)))

        #Then find max
        sorted_ACC=sorted(ACC)
        self.source_cv_accuracy=sorted_ACC
        print sorted_ACC
        best_alpha=sorted_ACC[-1][1][0]
        best_eta=sorted_ACC[-1][1][1]
        #Print Refit the model
        #self.fit_domain_classifier(best_alpha,X_all)



        ridge_clf = RidgeClassifier(alpha=best_alpha)
        ridge_clf.fit(X_all, D_vector) #Before it was X_all
        self.C=ridge_clf.coef_.T

        DvectorReg = np.array(D_vector)
        if self.orthogonal_reg:
            eta =-eta
            #Changing Dvector only in msda else C become 0
            DvectorReg[:]=0.0

        if self.target_reg:
            eta =-eta
            #Changing Dvector only in msda else C become 0
            DvectorReg[:]=1.0

        Z=np.dot(self.C,self.C.T)
        Icc= np.linalg.inv(np.eye(X_all.shape[1])-best_eta*Z)
        print "Computing msda with Domain Regularizer"
        hx,W=denoising_autoencoders.mDA_domain_regularization(X_all.T,self.noise,best_eta,self.C,DvectorReg,Icc)
        Xs_mda = hx.T[:ns,:]
        self.W=W
        #Maybe Retrain on all Here ....
        #Here I remove some part of the training sets ...
        #Change here with feature labelling ...
        #It is cheating , is it doing msda with self-labelling ....
        self.clf = domain_adaptation_baseline.cross_validate_classifier(Xs_mda, Ys_train, LogisticRegression, score='accuracy', ncv=3, n_jobs=3)




    def fit(self):
        #Place Holder for Reverse Cross Validatation
        return self.cross_val_source()


    def transform(self,Xtest):
        if self.W is not None:
            Xtest_msda = denoising_autoencoders.transform_test(Xtest.T,self.W,layer_func=self.layer_func,use_bias=False).T
            return Xtest_msda
        else:
            raise Exception("Dom Reg Autoencoders not fitted")



