function [W,ZC]=targetreg_semisup_iterWZ(method,srcX,srcY,tgtrX,tgtrY,tgttX,params)
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

% in the case of classifier loss ( TCW, TDCW, TMCW, ...) the method can iterate between estimation
% of W and the estimation of an MCF (ridge with marginalized noise estimation) classifier ZC 

% srcX contains the source features in form of (nbsrc, dimfeatures)
% srcY a list (nbsrc,1) containing the class labels (numeric);
% splits contains the split of sources splits=(nbsrc1,nbsrc2, ...) with sum(splits)=nbsrc
% tgtrX contains the training (labeled) target  features in form of (nbtr, dimfeatures)
% tgtrY a list (nbtr,1) with the class labels for the training target set
% tgtrX  and tgtrY  can be empty sets in the unsupervised case
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
% concatenated together with the original features before feeding into any classifier)


% If you use this code please cite:
% Csurka, Gabriela and Chidlovskii, Boris and Clinchant, St\'{e}phane and  Michel, Sophia, Unsupervised Domain Adaptation with Regularized Domain Instance Denoising
% ECCV workshop on Transferring and Adapting Source Knowledge in Computer
% Vision (TASK-CV), 2016

% If you have any questions related to or problem with the code
% Contact: Gabriela.Csurka@xrce.xerox.com
%


%%
if nargin <7
    
    params.MDA.ns=0.1;  % noise level in mda
    params.MDA.reg=0.001; % regularizer in mda
    params.TD.eta=1;   %weight for the domain loss
    params.TD.reg=1; % domain classifier regularizer
    params.TM.lambda=1; % weight for the MMD loss
    params.TC.gamma=1; % weight for the classifier loss
    params.TC.reg=1; % classifier regularizer
    params.verbose=0; % show loss when iterate
    params.miter=5; % maximum number of iteration between W and ZC in joint learning
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

% in case we have only target (no source) 
if ~isempty(srcY)
    difflabels=unique(srcY);
    srcLabs=-1*ones(nbsrc,nbcls);
    for l=1:nbcls
        srcLabs(srcY==difflabels(l),l)=1;
    end
else
    srcLabs=[];
    difflabels=[];
end


% in case we have labeled target set (semi-supervised case)
if ~isempty(tgtrY)
    if isempty(difflabels)
        difflabels=unique(tgtrY);
    end
    tgtrLabs= -1*ones(nbttr,nbcls);
    for l=1:nbcls
        tgtrLabs(tgtrY==difflabels(l),l)=1;
    end
else
    tgtrLabs=[];
end


trainX=[srcX;tgtrX];
allX=[srcX;tgtrX;tgttX];
trainLabs=[srcLabs; tgtrLabs];


% buld the supervised training and the unsupervised set containing all data

S = trainX'*trainX;      % matrix S (dim,dim)
Sa=allX'*allX;
XY=trainX'*trainLabs;



ns=params.MDA.ns;

q = ones(dim,1)*(1-ns);% corruption vector
Pa = Sa.*repmat(q',dim,1);% component P: (d x d)size(
Qa = Sa.*(q*q'); % component Q: (d x d), S .x 'noise' matrix
Qa(1:dim+1:end) = q.*diag(Sa); % put q*S instead of q^2 on the diagonal
Q = S.*(q*q'); % component Q: (d x d), S .x 'noise' matrix
Q(1:dim+1:end) = q.*diag(S); % put q*S instead of q^2 on the diagonal
Q=Q+tol^2*eye(dim);
Qa=Qa+tol^2*eye(dim);

% define domain labels and related components
switch method
    case {'TD','TDr','TDCW','TDWC','TDFW'} % domain loss 
        YD=[ones(1,nbsrc),-ones(1,nbtgt)  ; -ones(1,nbsrc),ones(1,nbtgt)]';  % domain gt labels
        XYD=allX'*YD;
        RT=[-ones(1,size(allX,1));ones(1,size(allX,1))]';  % push all towards target
        Pd=allX'*RT;
end

 % define components required when we have the linear MMD loss    
 switch method
    case {'TM','TMCW','TMWC','TMFW','TDM'}

        Mu=[(1/nbsrc^2)*ones(nbsrc,nbsrc), -1/(nbsrc*nbtgt)*ones(nbsrc,nbtgt); -1/(nbsrc*nbtgt)*ones(nbtgt,nbsrc), (1/nbtgt^2)*ones(nbtgt,nbtgt)];
        MMD=(allX'*Mu*allX);
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
        
        
        if nbttr~=0
            [nblbd, nbcls]=size(trainLabs);
            for c=1: nbcls
                Mc=zeros(nblbd,nblbd);
                ffs=find(srcLabs(:,c)>0);
                fft=nbsrc+find(tgtrLabs(:,c)>0);
                nsc=length(ffs);
                nst=length(fft);
                Mc(ffs,ffs)=1/(nsc^2);
                Mc(fft,fft)=1/(nst^2);
                Mc(ffs,fft)=-1/(nsc*nst);
                Mc(fft,ffs)=-1/(nsc*nst);
                MMDC= MMDC+trainX'*Mc*trainX;
            end
        else
            nblbd=nbsrc+nbtgt;
            fft=nbsrc+[1:nbtgt];
            nbcls=size(trainLabs,2);
            nst=nbtgt;
            for c=1: nbcls
                Mc=zeros(nblbd,nblbd);
                ffs=find(srcLabs(:,c)>0);
                nsc=length(ffs);
                Mc(ffs,ffs)=1/(nsc^2);
                Mc(fft,fft)= 1/(nst^2);
                Mc(ffs,fft)= -1/(nsc*nst);
                Mc(fft,ffs)= -1/(nsc*nst);
                MMDC= MMDC+allX'*Mc*allX;
            end
            
        end
        Qm=MMDC.*(q*q');
        Qm(1:dim+1:end) = q.*diag(MMDC);
        Qm=Qm+tol^2*eye(dim);
   
        
end


ZC=[];
switch method
    case 'MDA'  % we have regularized MDA
        W = (Qa+params.MDA.reg*eye(dim))\Pa;  % compute projection
    case 'TD' %   MDA + domain loss  (without MDA reg) 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % domain classifier
        P_TD=(Pa+params.TD.eta*(1-ns)*Pd*ZD')/(eye(dim)+params.TD.eta*(ZD*ZD'));
        W=Qa\P_TD;
    case 'TDr'  % MDA + domain loss 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % domain classifier
        B=(eye(dim)+params.TD.eta*(ZD*ZD'));
        C=Qa\(Pa+params.TD.eta*(1-ns)*Pd*ZD');
        A=params.MDA.reg*Qa^(-1);
        W =sylvester(A,B,C);
    case {'TM','TCM'} % MDA + MMD loss
        W=(Qa+params.MDA.reg*eye(dim)+params.TM.lambda*Qm)\Pa;
    case {'TDM','SDM','TDCM','SDCM'} % MDA + domain loss 
        ZD=(Sa+params.TD.reg*eye(dim))\XYD;  % domain classifier
        B=(eye(dim)+params.TD.eta*(ZD*ZD'));
        C=Qa\(Pa+params.TD.eta*(1-ns)*Pd*ZD');
        A=Qa\(params.TM.lambda*Qm+params.MDA.reg*eye(dim));
        W =sylvester(A,B,C);
   
     case {'TCW','TFW','TDCW','TDFW','TMCW','TMFW','TCMCW','TCMFW'}  % combine several losses
        
        % different initialisation for ZC
        
        switch method
            case {'TCW','TDCW','TMCW','TCMCW'} %  use ridge to compute ZC
                ZC=(S+ params.TC.reg*eye(dim))\XY;  % gt class labels
            case {'TFW','TDFW','TMFW','TCMFW'}   % use MCF to compute ZC
                ZC =(Q+ params.TC.reg*eye(dim))\XY;  % gt class labels
            case {'TWC','TDWC','TMWC','TCMWC'}  % Initialize W with  MDA  then begin with ZC estimate
                W = (Qa+params.MDA.reg*eye(dim))\Pa;
                ZC=(params.TC.gamma*W'*Q*W+params.TC.reg*eye(dim))\(params.TC.gamma*(1-ns)*W'*XY);
        end
        
        
        switch method
            
            case {'TCW','TFW', 'TWC'}  % MDA + classifier loss
                
                A=Q\(Qa+params.MDA.reg*eye(dim));
                
                Loss=0;
                % iterate between ZC and W starting with W
                for it=1:params.miter
                    oldLoss=Loss;
                    
                    B=params.TC.gamma*(ZC*ZC');
                    C=Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC');
                    W =sylvester(A,B,C);
                    WQW=W'*Q*W;
                    
                    if params.verbose
                        %disp(sum(sum(A*W+W*B-C)))
                        Loss1=trace(W'*Qa*W)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+params.MDA.reg*trace(W'*W) ...
                            + params.TC.gamma*trace(ZC'*WQW*ZC)-2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC) +params.MDA.reg*trace(W'*W);
                        
                        showloss=sprintf('\n it:%d DZ=%f Loss=%f, dL=%f \t', it, sum(sum((params.TC.gamma*WQW+params.TC.reg*eye(dim))*ZC-params.TC.gamma*(1-ns)*W'*XY)), ...
                            Loss1, oldLoss-Loss1);
                        disp(showloss);
                    end
                    
                    
                    ZC=(params.TC.gamma*WQW+params.TC.reg*eye(dim))\(params.TC.gamma*(1-ns)*W'*XY);
                    
                    Loss=trace(W'*Qa*W)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+ +params.MDA.reg*trace(W'*W) ...
                        + params.TC.gamma*trace(ZC'*WQW*ZC)  - 2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC);
                    
                    if params.verbose
                        %disp(sum(sum(A*W+W*B-C)))
                        B=params.TC.gamma*(ZC*ZC');
                        C=Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC');
                        showloss=sprintf('DW= %f Loss=%f \t d1=%f d2=%f\t', sum(sum(A*W+W*B-C)), Loss,Loss1-Loss, oldLoss-Loss);
                        disp(showloss);
                    end
                    
                    if    abs(oldLoss-Loss)/abs(Loss)<tol;
                        break;
                    end
                end
                
            case {'TDCW','TDWC','TDFW' }  % MDA + domain loss+  classifier loss
                ZD=(Sa+params.TD.reg*eye(dim))\XYD;
                ZZp=ZD*ZD';
                A=Q\Qa;
                
                Loss=0;
                
                % iterate between ZC and W starting with W
                for it=1:params.miter
                    oldLoss=Loss;
                    
                    B=params.TC.gamma*(ZC*ZC')/(eye(dim)+params.TD.eta*ZZp);
                    C=(Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC'+params.TD.eta*(1-ns)*Pd*ZD'))/(eye(dim)+params.TD.eta*ZZp);
                    W = sylvester(A,B,C);
                    WQW=W'*Q*W;
                    WQaW=W'*Qa*W;
                    
                    if params.verbose
                        %disp(sum(sum(A*W+W*B-C)))
                        Loss1=trace(WQaW)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+ ...
                            params.TD.eta*trace(ZD'*WQaW*ZD)-2*params.TD.eta*(1-ns)*trace(Pd'*W*ZD)+ ...
                            params.TC.gamma*trace(ZC'*WQW*ZC)-2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC);
                        
                        showloss=sprintf('\n it:%d DZ=%f Loss=%f, dL=%f \t', it, sum(sum((params.TC.gamma*WQW+params.TC.reg*eye(dim))*ZC-params.TC.gamma*(1-ns)*W'*XY)), ...
                            Loss1, oldLoss-Loss1);
                        disp(showloss);
                    end
                    
                    
                    ZC=(params.TC.gamma*WQW+params.TC.reg*eye(dim))\(params.TC.gamma*(1-ns)*W'*XY);
                    
                    Loss=trace(WQaW)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+ ...
                        params.TD.eta*trace(ZD'*WQaW*ZD)-2*params.TD.eta*(1-ns)*trace(Pd'*W*ZD)+ ...
                        params.TC.gamma*trace(ZC'*WQW*ZC)-2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC);
                    
                    
                    if  params.verbose
                        B=params.TC.gamma*(ZC*ZC')/(eye(dim)+params.TD.eta*ZZp);
                        C=(Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC')+params.TD.eta*(1-ns)*Pd*ZD')/(eye(dim)+params.TD.eta*ZZp);
                        showloss=sprintf('DW= %f Loss=%f \t d1=%f d2=%f\t', sum(sum(A*W+W*B-C)), Loss,Loss1-Loss, oldLoss-Loss);
                        disp(showloss);
                        %disp(sum(sum(A*W+W*B-C)))
                        
                    end
                    
                    if     abs(oldLoss-Loss)/abs(Loss)<tol;
                        break;
                    end
            
                end
                
                
                
            case {'TMCW','TMWC','TCMCW','TCMWC','TMFW','TCMFW'} % MDA + MMD loss + classifier loss
                
                A=Q\(Qa+params.MDA.reg*eye(dim)+params.TM.lambda*Qm);
                
                Loss=0;
                for it=1:params.miter
                    oldLoss=Loss;
                    
                    B=params.TC.gamma*(ZC*ZC');
                    C=Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC');
                    
                    W = sylvester(A,B,C);
                    WQW=W'*Q*W;
                    WQaW=W'*Qa*W;
                    
                    if params.verbose
                        %disp(sum(sum(A*W+W*B-C)))
                        Loss1=trace(WQaW)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+ ...
                            params.TM.lambda*trace(W'*Qm*W) + params.MDA.reg*trace(W'*W)+ ...
                            params.TC.gamma*trace(ZC'*WQW*ZC)-2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC);
                        
                        
                        showloss=sprintf('\n it:%d DZ=%f, Loss=%f, dL=%f \t', it, sum(sum((params.TC.gamma*WQW+params.TC.reg*eye(dim))*ZC-params.TC.gamma*(1-ns)*W'*XY)), ...
                            Loss1, oldLoss-Loss1);
                        disp(showloss);
                    end
                    
                    ZC=(params.TC.gamma*WQW+params.TC.reg*eye(dim))\(params.TC.gamma*(1-ns)*W'*XY);
                    
                    Loss=trace(WQaW)-2*trace(Pa*W)+params.TC.reg*trace(ZC'*ZC)+ ...
                        params.TM.lambda*trace(W'*Qm*W) +  params.MDA.reg*trace(W'*W)+ ...
                        params.TC.gamma*trace(ZC'*WQW*ZC)-2*params.TC.gamma*(1-ns)*trace(XY'*W*ZC);
                    
                    if  params.verbose
                        B=params.TC.gamma*(ZC*ZC');
                        C=Q\(Pa+params.TC.gamma*(1-ns)*XY*ZC');
                        showloss=sprintf('DW= %f Loss=%f \t d1=%f d2=%f\t', sum(sum(A*W+W*B-C)), Loss,Loss1-Loss, oldLoss-Loss);
                        disp(showloss);
                        %disp(sum(sum(A*W+W*B-C)))
                        
                    end
                    
                    if    abs(oldLoss-Loss)/abs(Loss)<tol;
                        break;
                    end
                    
                end
 
        end
end



