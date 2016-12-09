function W=targetreg_semisup_MS(method,srcX,srcY,splits,tgtrX,tgtrY,tgttX,params)
% Multisource (MS) target regularized MDA 
% the method precise which regularization losses are  to be used:
% MDA - the baseline MDA (this component is always present)
% TDr, - MDA +  domain loss 
% TD - MDA + domain loss (but without the regularizer  on W)
% TM -  with linear MMD domain 
% TCM with linear class specific MMD domain  
% TDM, TDCM - TM with MMD loss (M or CM)
% TCW,TFW -  MDA + classifier loss
% TDCW,TDFW -  MDA + domain loss +  classifier loss
% TMCW,TMFW,TCMCW,TCMFW - MDA + MMD loss (M or CM) +  classifier loss
% Except TD, TDCW and TDFW the other cases uses a  regularization  on the norm of W

% srcX contains the source features in form of (nbsrc, dimfeatures)
% srcY a list (nbsrc,1) containing the class labels (numeric);
% splits contains the split of sources splits=(nbsrc1,nbsrc2, ...) with sum(splits)=nbsrc 
% tgtrX contains the training (labeled) target  features in form of (nbtr, dimfeatures)
% tgtrY a list (nbtr,1) with the class labels for the training target set
% tgtrX  and tgtrY  can be empty sets in the Unsupervised Case
% tgttX contains the test (unlabeled) target  features in form of (nbtt, dimfeatures)

% note that in the case of PCA reduction the reduction should be done
% before this function and srcX, tgtrX and tgttX will correspond to the
% reduced features (the PCA reduction allows to decrease the computations especially when 
% the sylvester solver is used 


% params allows to define change the default parameters
% to set the domain (params.TD.reg) and classifier regularizer (params.TC.reg) 
% cross validation can be one option, or alternatively use  normest(mean(tgttX));

% Use W to project the data and use any classifier  (KNN, NCM< Ridge, NCM, LR) in the projected space
% or use the projected outputs (with or without nonlinear RELU or tanh) as a new input 
% to stack several layers together (outputs of several layers can be
% concatenated together with teh original features before feeding into any classifier) 


% If you use this code please cite:
% Csurka, Gabriela and Chidlovskii, Boris and Clinchant, St\'{e}phane and  Michel, Sophia, Unsupervised Domain Adaptation with Regularized Domain Instance Denoising
% ECCV workshop on Transferring and Adapting Source Knowledge in Computer
% Vision (TASK-CV), 2016

% If you have any questions related to or problem with the code
% Contact: Gabriela.Csurka@xrce.xerox.com
%


%% default parameters
if nargin <8
    
    params.MDA.ns=0.1;  % noise level in mda
    params.MDA.reg=0.001; % regularizer in mda
    params.TD.eta=1;   %weight for the domain loss
    params.TD.reg=1; % domain classifier regularizer
    params.TM.lambda=1; % weight for the MMD loss
    params.TC.gamma=1; % weight for the classifier loss
    params.TC.reg=1; % classifier regularizer 
    
    
end
tol = 1e-5;


W=[];


nbsrc=length(srcY);
nbttr=length(tgtrY);
nbtgt=length(tgtrY)+size(tgttX,1);
dim=size(tgttX,2);


% build binary labelmatrix
difflabels=unique(srcY);
nbcls=length(difflabels);

srcLabs=-1*ones(nbsrc,nbcls);
for l=1:nbcls
    srcLabs(srcY==difflabels(l),l)=1;
end

% in case we have labeled target set (semi-supervised case)
if ~isempty(tgtrY)
    tgtrLabs= -1*ones(nbttr,nbcls);
    for l=1:nbcls
     tgtrLabs(tgtrY==difflabels(l),l)=1;
    end
else
    tgtrLabs=[];
end


% buld the supervised training and the unsupervised set containing all data

trainX=[srcX;tgtrX];
allX=[srcX;tgtrX;tgttX];
trainLabs=[srcLabs; tgtrLabs];

% covariance matrixes (dim,dim)
S = trainX'*trainX;     
Sa = allX'*allX;

XY=trainX'*trainLabs;


% the baselne MDA related components
ns=params.MDA.ns;
q = ones(dim,1)*(1-ns);% corruption vector
Pa = Sa.*repmat(q',dim,1);% component P: (d x d)
Qa = Sa.*(q*q'); % component Q: (d x d), S .x 'noise' matrix
Qa(1:dim+1:end) = q.*diag(Sa); % put q*S instead of q^2 on the diagonal
Q = S.*(q*q'); % component Q: (d x d), S .x 'noise' matrix
Q(1:dim+1:end) = q.*diag(S); % put q*S instead of q^2 on the diagonal
Q=Q+tol^2*eye(dim);
Qa=Qa+tol^2*eye(dim);

% define domain labels and related components
switch method
    case {'TD','TDr','TDCW','TDWC','TDFW','TDM','TDCM'}
        YD=[];RT=[];
        % each row corresponding to one of the sources and each columns to
        % one data instance
        for s=1:length(splits)
            YD=[YD; -ones(1,sum(splits(1:s-1))), ones(1,splits(s)), -ones(1,sum(splits(s+1:end))), -ones(1,nbtgt)]; % groud truth source labels 
            RT=[RT; -ones(1,size(allX,1))];  % push all source data towards target
        end
        
        % the row corresponding to  the target
        YD=[YD; -ones(1,sum(splits)),ones(1,nbtgt)]';  % groud truth traget labels 
        RT=[RT;  ones(1,size(allX,1))]'; % push all data  towards target
        
        XYD=allX'*YD;
        Pd=allX'*RT;
end

% define components required when we have the linear MMD loss
switch method
    case {'TM','TMCW','TMWC','TMFW','TDM'}
        
        MMD=zeros(dim,dim);
         nball=size(allX,1);
        fft=nbsrc+[1:nbtgt];
        nst=nbtgt;
        for s1=1:length(splits)-1
            stepind1=sum(splits(1:s1-1));
            ffs=[1+stepind1:splits(s1)+stepind1];
            nsc=length(ffs);
            for s2=s1+1:length(splits)
                Mu=zeros(nball,nball);
                stepind2=sum(splits(1:s2-1));
                ffs2=[1+stepind2:splits(s2)+stepind2];
                nsc2=length(ffs2);
                Mu(ffs,ffs)=1/(nsc^2);
                Mu(ffs2,ffs2)=1/(nsc2^2);
                Mu(ffs,ffs2)=-1/(nsc*nsc2);
                Mu(ffs2,ffs)=-1/(nsc2*nsc);
                MMD= MMD+allX'*Mu*allX;
            end
            Mu=zeros(nball,nball);
            Mu(ffs,ffs)=1/(nsc^2);
            Mu(fft,fft)= 1/(nst^2);
            Mu(ffs,fft)= -1/(nsc*nst);
            Mu(fft,ffs)= -1/(nsc*nst);
            MMD= MMD+allX'*Mu*allX;
        end
             
            
        %Mu=[(1/nbsrc^2)*ones(nbsrc,nbsrc), -1/(nbsrc*nbtgt)*ones(nbsrc,nbtgt); -1/(nbsrc*nbtgt)*ones(nbtgt,nbsrc), (1/nbtgt^2)*ones(nbtgt,nbtgt)];
        %MMD=(allX'*Mu*allX);
        Qm=MMD.*(q*q');
        Qm(1:dim+1:end) = q.*diag(MMD);
        Qm=Qm+tol^2*eye(dim);
end

% define components required when we have the linear per class MMD loss
% we build the MC corresponding to all data
switch method
    case {'TCM','TDCM','TCMCW','TCMFW'}
        
        MMDC=zeros(dim,dim);
        nball=size(allX,1);
        nbcls=size(trainLabs,2);
        
        % use class means for the sources and labeled target samples 
        % and domain mean for the unlabeled target examples
        
        for s1=1:length(splits)-1
            stepind1=sum(splits(1:s1-1));
            srcLabs1=srcLabs(1+stepind1:splits(s1)+stepind1,:);
            
            for s2=s1+1:length(splits)
                stepind2=sum(splits(1:s2-1));
                srcLabs2=srcLabs(1+stepind2:splits(s2)+stepind2,:);
                
                for c=1: nbcls
                    Mc=zeros(nball,nball);
                    ffs=stepind1+find(srcLabs1(:,c)>0);
                    fft=stepind2+find(srcLabs2(:,c)>0);
                    nsc=length(ffs);
                    nst=length(fft);
                    Mc(ffs,ffs)=1/(nsc^2);
                    Mc(fft,fft)=1/(nst^2);
                    Mc(ffs,fft)=-1/(nsc*nst);
                    Mc(fft,ffs)=-1/(nsc*nst);
                    MMDC= MMDC+allX'*Mc*allX;
                end
            end
        end
        if nbttr~=0
            [nblbd, nbcls]=size(trainLabs);
            for s1=1:length(splits)
                stepind1=sum(splits(1:s1-1));
                srcLabs1=srcLabs(1+stepind1:splits(s1)+stepind1,:);
                for c=1: nbcls
                    Mc=zeros(nblbd,nblbd);
                    ffs=sum(splits(1:s1-1))+find(srcLabs1(:,c)>0);
                    fft=nbsrc+find(tgtrLabs(:,c)>0);
                    nsc=length(ffs);
                    nst=length(fft);
                    Mc(ffs,ffs)=1/(nsc^2);
                    Mc(fft,fft)=1/(nst^2);
                    Mc(ffs,fft)=-1/(nsc*nst);
                    Mc(fft,ffs)=-1/(nsc*nst);
                    MMDC= trainX'*Mc*trainX;;
                end
            end
        else
            fft=nbsrc+[1:nbtgt];
            nst=nbtgt;
            for s1=1:length(splits)
                stepind1=sum(splits(1:s1-1));
                srcLabs1=srcLabs(1+stepind1:splits(s1)+stepind1,:);
                
                for c=1: nbcls
                    Mc=zeros(nball,nball);
                    ffs=sum(splits(1:s1-1))+find(srcLabs1(:,c)>0);
                    nsc=length(ffs);
                    Mc(ffs,ffs)=1/(nsc^2);
                    Mc(fft,fft)= 1/(nst^2);
                    Mc(ffs,fft)= -1/(nsc*nst);
                    Mc(fft,ffs)= -1/(nsc*nst);
                    MMDC= MMDC+allX'*Mc*allX;
                end
                
            end
        end
        Qm=MMDC.*(q*q');
        Qm(1:dim+1:end) = q.*diag(MMDC);
        Qm=Qm+tol^2*eye(dim);
end



switch method
    case 'MDA'  % baseline regularized MDA
        W = (Qa+params.MDA.reg*eye(dim))\Pa;  % compute projection
    case {'TD'} %   MDA + domain loss  (without MDA reg) 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % compute domain classifier
        P_TD=(Pa+params.TD.eta*(1-ns)*Pd*ZD')/(eye(dim)+params.TD.eta*(ZD*ZD'));
        W=Qa\P_TD;
    case {'TDr'}   % MDA + domain loss 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % compute domain classifier
        B=(eye(dim)+params.TD.eta*(ZD*ZD'));
        C=Qa\(Pa+params.TD.eta*(1-ns)*Pd*ZD');
        A=params.MDA.reg*Qa^(-1);
        W =sylvester(A,B,C);
    case {'TM','TCM'}  
        W=(Qa+params.MDA.reg*eye(dim)+params.TM.lambda*Qm)\Pa;
    case {'TDM','TDCM'}  % MDA + domain loss 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % domain classifier
        B=(eye(dim)+params.TD.eta*(ZD*ZD'));
        C=Qa\(Pa+params.TD.eta*(1-ns)*Pd*ZD');
        A=Qa\(params.TM.lambda*Qm+params.MDA.reg*eye(dim));
        W =sylvester(A,B,C);

    case {'TCW','TFW','TDCW','TDFW','TMCW','TMFW','TCMCW','TCMFW'}  % combine several losses
        
        
        ZC=[]; mXY=[];
        
        % build and concatenate  terms for each source
        for s1=1:length(splits)
            stepind1=sum(splits(1:s1-1));
            trainXps=trainX(1+stepind1:splits(s1)+stepind1,:);
            Sps= trainXps'*trainXps;
            Qps = Sps.*(q*q'); % component Q: (d x d), S .x 'noise' matrix
            Qps(1:dim+1:end) = q.*diag(Sps); % put q*S instead of q^2 on the diagonal
            Qps=Qps+tol^2*eye(dim);
            
            
            XYps=trainXps'*trainLabs(1+stepind1:splits(s1)+stepind1,:);
            
            
            mXY=[mXY,XY];
            % nbttr=length(tgtrY);
            switch method
                case {'TCW','TDCW','TMCW','TCMCW'}  % use ridge to compute ZC
                    ZC=[ZC,(Sps+ params.TC.reg*eye(dim))\XYps];  % gt class labels
                case {'TFW','TDFW','TMFW','TCMFW'} % use MCF to compute ZC
                    ZC=[ZC,(Qps+ params.TC.reg*eye(dim))\XYps];  % gt class labels
            end
        end
        
        
        switch method
            case {'TCW','TFW'}  % MDA + classifier loss
                A=Q\(Qa+params.MDA.reg*eye(dim));
                B=params.TC.gamma*(ZC*ZC');
                C=Q\(Pa+params.TC.gamma*(1-ns)*mXY*ZC');
                W =sylvester(A,B,C);
                
            case {'TDCW','TDFW'}  % MDA + domain loss+  classifier loss
                ZD=(Sa+params.TD.reg*eye(dim))\XYD;
                ZZp=ZD*ZD';
                A=Q\Qa;
                B=params.TC.gamma*(ZC*ZC')/(eye(dim)+params.TD.eta*ZZp);
                C=(Q\(Pa+params.TC.gamma*(1-ns)*mXY*ZC'+params.TD.eta*(1-ns)*Pd*ZD'))/(eye(dim)+params.TD.eta*ZZp);
                W = sylvester(A,B,C);
                
                
            case {'TMCW','TMFW','TCMCW','TCMFW'}  % MDA + MMD loss + classifier loss
                A=Q\(Qa+params.MDA.reg*eye(dim)+params.TM.lambda*Qm);
                B=params.TC.gamma*(ZC*ZC');
                C=Q\(Pa+params.TC.gamma*(1-ns)*mXY*ZC');
                W = sylvester(A,B,C);
        end
end




