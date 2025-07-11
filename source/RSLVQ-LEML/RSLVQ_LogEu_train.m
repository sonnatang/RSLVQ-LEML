function [model, varargout] = RSLVQ_LogEu_train(trainSet, trainLab, varargin)
%RSLVQ_LogEu_train.m - trains the Robust soft LVQ  with Log Euclidean
%metric learning algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet is n times n times m array, containing m  n times n SPD matrix;
%  trainLab = [1;1;2;...];
%  model=RSLVQ_LogEu_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = RSLVQ_LogEu_classify(trainSet, model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : traning set organized in matrix array of size n*n*m, 
%      containing m instances, each instance is an is n times n SPD matrix;
%  trainLab : vector with the labels of the training set
% optional parameters:
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could
%  be a number or a vector with the number for each class
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  initialMatrix     : the matrix omega to start with. If not given random
%  initialization for rectangular matrices and Unity for squared omega
%  dim               : (default=nb of features for training) the maximum rank or projection dimension
%  regularization    : (default=0) values usually between 0 and 1 treat with care. 
%  Regularizes the eigenvalue spectrum of omega'*omega to be more homogeneous
%  testSet           : (default=[]) an optional test set used to compute
%  the test error. The last column is expected to be a label vector
%  comparable        : (default=0) a flag which resets the random generator
%  to produce comparable results if set to 1
%  optimization      : (default=fminlbfgs) indicates which optimization is used: sgd or fminlbfgs
% parameter for the stochastic gradient descent sgd
%  nb_epochs             : (default=100) the number of epochs for sgd
%  learningRatePrototypes: (default=[]) the learning rate for the prototypes. 
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  learningRateMatrix    : (default=[]) the learning rate for the matrix.
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  MatrixStart           : (default=1) the epoch to start the matrix training

%
% output: the  model with prototypes w their labels c_w and the matrix
% omega or Q depends on the choice of learning the metric
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information: 
%   F. Tang, X. Zhang: Robust soft learning vector quantization with LEML for SPD matrices ...
%   IEEE Transactions on Emerging Topics in Computational Intellifence, 
% 
% Fengzhen Tang (adapted from the code written by Kerstin Bunte ...
% available at http://matlabserver.cs.rug.nl/gmlvqweb/web/)
%
% tangfengzhen@sia.cn
% Monday Dec 18 16:42 2023
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,3) & isnumeric(x));
p.addParamValue('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParamValue('initialPrototypes',[], @(x)(size(x,2)-1==size(trainSet,2) && isfloat(x)));
p.addParamValue('initialMatrix',[], @(x)(size(x,2)==size(trainSet,2) && isfloat(x)));
p.addParamValue('dim',size(trainSet,2), @(x)(~(x-floor(x)) && x<=size(trainSet,2) && x>0));
p.addParamValue('regularization',0, @(x)(isfloat(x) && x>=0));
p.addOptional('testSet', [], @(x) (size(x,1)-1)==size(trainSet,1)&...
    (size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));
%p.addOptional('optimization', 'fminlbfgs', @(x)any(strcmpi(x,{'sgd','fminlbfgs'})));
% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addParamValue('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle'))); % && (length(x)==2 || length(x)==p.Results.epochs)
p.addParamValue('learningRateMatrix', [], @(x)(isfloat(x)  || isa(x,'function_handle')));

p.addParamValue('sigma2', [], @(x)(isfloat(x)));

p.addOptional('MatrixStart', 1, @(x)(~(x-floor(x))));

p.addOptional('Qmetric', 'log-euclid', @(x)any(strcmpi(x,{'log-euclid','flat','polar'})));
p.CaseSensitive = true;
p.FunctionName = 'RSLVQ_LogEu';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable,
    rng('default');
end
%%% set useful variables



%convert trainset into its logithm domain
trainSet_log = spd2logm(trainSet);

[nb_features,nb_features1,nb_samples] = size(trainSet);

% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

classes = unique(trainLab);
nb_classes = length(classes);
dim = p.Results.dim;
Qmetric = p.Results.Qmetric;
%reducedDim = p.Results.reducedDim;

MatrixStart = p.Results.MatrixStart;
testSet = p.Results.testSet;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of omega''*omega with ',num2str(regularization)]);end

initialization = rmfield(p.Results, 'trainSet');
initialization.trainSet = [num2str(nb_features), 'x', num2str(nb_features),...
    'x',num2str(nb_samples),' matrix'];
initialization = rmfield(initialization, 'trainLab');
initialization.trainLab = ['vector of length ',num2str(length(trainLab))];
if ~isempty(testSet)
    initialization = rmfield(initialization, 'testSet');
    initialization.testSet = [num2str(size(testSet,1)-1),'x',...
        num2str(size(testSet,2)-1),'x',num2str(size(testSet,3)), ' matrix'];
end

% Display all arguments.
disp 'Settings for RSLVQ_LEML:'
disp(initialization);

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
nb_ppc = p.Results.PrototypesPerClass;
if length(nb_ppc)~=nb_classes,
    nb_ppc = ones(1,nb_classes)*nb_ppc;
end


%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    % initialize near the class centers
    nb_prot = sum(nb_ppc);
    w = zeros(nb_features,nb_features1,nb_prot);
    c_w = zeros(nb_prot,1);
    actPos = 1;
    for actClass=1:nb_classes
        nb_prot_c = nb_ppc(actClass);
        classMean = mean(trainSet_log(:,:,trainLab==classes(actClass)),3);
        % set the prototypes to the class mean and add a random variation between -0.1 and 0.1
        c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
        for i = 1:nb_prot_c       
            d = (2*rand(nb_features,nb_features1) - ones(nb_features,nb_features1))/1000;
            tp = triu(d);
            t = tp+tp'- diag(diag(tp));
            init_w  = classMean+t;
            w(:,:,actPos) = init_w ;
            actPos = actPos+1;
        end
    end
else
    % initialize with given w
    w = p.Results.initialPrototypes(:,1:end-1);
    c_w = p.Results.initialPrototypes(:,end);
end

%%% initialize the matrix
if strcmp(Qmetric,'log-euclid') % no dimension reduction, we work with Q directly using log-Euclidean metric. 
    if(p.Results.dim==nb_features)
       if isempty(p.Results.initialMatrix)
%             Q = eye(nb_features)/nb_features; 
            Q = eye(nb_features); 
       else
            Q = p.Results.initialMatrix;
       end
        model = struct('w',w,'c_w',c_w,'Q',Q);
        clear w c_w Q;     
    end
    
elseif strcmp(Qmetric,'flat') %Q is semi-positive definite, let Q = omega^T * omega and we work with omega on quotient manifold 
    if isempty(p.Results.initialMatrix)
        if(p.Results.dim==nb_features)
            omega = eye(nb_features);
        else % initialize with random numbers between -1 and 1
            omega = rand(dim,nb_features)*2-ones(dim,nb_features);
        end
    else
        omega = p.Results.initialMatrix;
    end
    % normalize the matrix
    %omega = omega / sqrt(sum(diag(omega'*omega)));
    model = struct('w',w,'c_w',c_w,'omega',omega);
    clear w c_w omega;
else
    if isempty(p.Results.initialMatrix)
        if(p.Results.dim==nb_features)
            polar_G = eye(nb_features,nb_features-1);
        else % initialize with random numbers between -1 and 1
            polar_G = rand(dim,nb_features)*2-ones(dim,nb_features);
        end
    else
        polar_G = p.Results.initialMatrix;
    end
    [polar_Z,polar_S,polar_V] = svd(polar_G,'econ');
    polar_U = polar_Z*polar_V';
    polar_R = polar_V*polar_S*polar_V';
    % normalize the matrix
    model = struct('w',w,'c_w',c_w,'polar_U',polar_U,'polar_R',polar_R);
    clear w c_w polar_U polar_R;
end

%%% gradient descent variables
nb_epochs = p.Results.nb_epochs;
% compute the vector of nb_epochs learning rates alpha for the prototype learning
if isa(p.Results.learningRatePrototypes,'function_handle')
    % with a given function specified from the user
    alphas = arrayfun(p.Results.learningRatePrototypes, 1:nb_epochs);
elseif length(p.Results.learningRatePrototypes)>2
    if length(p.Results.learningRatePrototypes)==nb_epochs
        alphas = p.Results.learningRatePrototypes;
    else
        disp('The learning rate vector for the prototypes does not fit the nb of epochs');
        return;
    end
else
    % or use an decay with a start and a decay value
    if isempty(p.Results.learningRatePrototypes)
        initialization.learningRatePrototypes =p.Results.PrototypesPerClass*[nb_features/100, nb_features/10000];
    end
    alpha_start = initialization.learningRatePrototypes(1);
    alpha_end = initialization.learningRatePrototypes(2);
    alphas = arrayfun(@(x) alpha_start * (alpha_end/alpha_start)^(x/nb_epochs), 1:nb_epochs);
%     Eta = alphas/10/p.Results.PrototypesPerClass;
%     alphas = arrayfun(@(x) alpha_start / (1+(x-1)*alpha_end), 1:nb_epochs);
end
% compute the vector of nb_epochs learning rates epsilon for the Matrix learning
epsilons = zeros(1,nb_epochs);
if isa(p.Results.learningRateMatrix,'function_handle')
    % with a given function specified from the user
% 	epsilons = arrayfun(p.Results.learningRateMatrix, 1:nb_epochs);
    epsilons(MatrixStart:nb_epochs) = arrayfun(p.Results.learningRateMatrix, MatrixStart:nb_epochs);
elseif length(p.Results.learningRateMatrix)>2
    if length(p.Results.learningRateMatrix)==nb_epochs
        epsilons = p.Results.learningRateMatrix;
    else
        disp('The learning rate vector for the Matrix does not fit the nb of epochs');
        return;
    end
else
    % or use an decay with a start and a decay value
    if isempty(p.Results.learningRateMatrix)
        initialization.learningRateMatrix = [nb_features/1000, nb_features/100000];
    end
    eps_start = initialization.learningRateMatrix(1);
    eps_end = initialization.learningRateMatrix(2);
%     epsilons = arrayfun(@(x) eps_start * (eps_end/eps_start)^(x/nb_epochs), 1:nb_epochs);
    epsilons(MatrixStart:nb_epochs) = arrayfun(@(x) eps_start * (eps_end/eps_start)^((x-MatrixStart)/(nb_epochs-MatrixStart)), MatrixStart:nb_epochs);
    epsilons = epsilons/10;
end

%%% the variance sigma2
if isempty(p.Results.sigma2)
    dists = computeDistanceLogEu(trainSet_log, model.w);
    sigma2_opt = 0.5*median(dists(:))+0.001*rand();    
    beta = 0.99;
    sigma2 = sigma2_opt + 0.2;
    sigma2s = zeros(nb_epochs,1);
    for t=1:nb_epochs
       if sigma2>=max(sigma2_opt - 0.2,0.01)
           beta = beta^1.1;
           sigma2 = sigma2*beta;
       end
       sigma2s(t) = sigma2;
    end
else
    sigma2s = p.Results.sigma2;
end  

if length(sigma2s)~=nb_epochs
    sigma2s = ones(nb_epochs,1)*sigma2s;
end

%%% initialize requested outputs
sigma2 = sigma2s(1);
model.sigma2 = sigma2;
trainError = [];
costs = [];
testError = [];
if nout>=2,
    % costs requested
    disp('The computation of the costs is an expensive operation, do it only if you really need it!');
    costs = ones(1,nb_epochs+1);
    costs(1) = RiemanPLVQ_costfun(trainSet_log,trainLab,model);

    if nout>=3,
            % train error requested
        trainError = ones(1,nb_epochs+1);
        estimatedLabels = PLVQ_LogEu_classify(trainSet, model); % error after initialization
        trainError(1) = sum( trainLab ~= estimatedLabels )/nb_samples;
        
       if nout>=4,
               % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                testError = ones(1,nb_epochs+1);
                estimatedLabels = PLVQ_LogEu_classify(testSet(1:end-1,1:end-1,:), model); % error after initialization
                testError(1) = sum( squeeze(testSet(end,end,:)) ~= estimatedLabels )/length(estimatedLabels);
            end        
            
        end
    end
end

%     figure;
%     plotData(trainSet,trainLab);
%     plotPrototypes(model.w,model.c_w);
%     hold off;
%%% optimize with stochastic gradient descent
for epoch=1:nb_epochs
    if mod(epoch,100)==0, disp(epoch); end
    % generate order to sweep through the trainingset
    order = randperm(nb_samples);	
    %sigma2 = sigma2s(epoch);
    %model.sigma2 = sigma2;
    
    % perform one sweep through trainingset
    for i=1:nb_samples
        sigma2 = model.sigma2;
        % select one training sample randomly
        xi = trainSet_log(:,:,order(i));
        c_xi = trainLab(order(i));

        dist = computeDistanceLogEu(xi, model.w, model);
        % determine the two winning prototypes
        % nearest prototype with the same class
        
%         prob = exp(-beta*dist); %changes by Fengzhen 2020-1-15 10:28   
%         proby = prob;
%         proby(~(model.c_w == c_xi))=0;
%         proby = proby/sum(proby);       
%         probx = prob/sum(prob);
        
        beta = 0.5/model.sigma2;
         %%%compute class probability
        fs = -beta*dist;
        fsmax = max(fs);
        prob = exp(fs-fsmax);
        probx = prob/sum(prob);
        fsy = fs;
        fsy(~(model.c_w == c_xi)) = [];
        fsymax = max(fsy);
        probyy = exp(fsy - fsymax); 
        probyy = probyy/sum(probyy);
        proby = zeros(size(prob));
	    proby(model.c_w == c_xi) = probyy; 
        
        if isnan(sum(proby)) | isnan(sum(probx))
            warning('nan probability, no update here');
            continue;  % no updates
        end
        alpha = alphas(epoch);
        
        if strcmp(Qmetric,'log-euclid')
		    Q = model.Q; 
        elseif strcmp(Qmetric,'flat')
		    Q = model.omega'*model.omega;
            G = model.omega;
        else
            Q = model.polar_U*(model.polar_R)^2*(model.polar_U)';
            U_w = model.polar_U;
            R_w = model.polar_R;
        end
        % prototype update
        probt = zeros(1,nb_prot);
        for J = 1:nb_prot
            wJ = model.w(:,:,J);
            c_wJ = model.c_w(J);
           
            if c_wJ == c_xi
              probt(J) = (proby(J)-probx(J));
            else
              probt(J) = - probx(J);
            end
             
             % gradient of prototypes
             DJ  = (xi-wJ);
             XJ = alpha/sigma2*probt(J)*Q*DJ;
             %update prototypes
             wJ = wJ + XJ;
             model.w(:,:,J) = wJ;
        end
        % update matrices
        if epsilons(epoch) > 0 % epoch >= MatrixStart epsilons(epoch) > 0
            switch Qmetric
                case 'log-euclid'
                    sum_Qy=0;sum_Qall=0;
                    for J = 1:nb_prot
                        wJ = model.w(:,:,J);
                        c_wJ = model.c_w(J);
                        sum_Qall= sum_Qall+probx(J)*((xi-wJ)^2)*Q;
                        if c_wJ == c_xi
                            sum_Qy = sum_Qy+proby(J)*((xi-wJ)^2)*Q;
                        end
                    end
                    model.Q=expm(logm(Q)-epsilons(epoch)/(2*sigma2)*sum_Qy+epsilons(epoch)/(2*sigma2)*sum_Qall);
%                     model.Q=expm(logm(Q)-Eta(epoch)/(2*sigma2)*sum_Qy+Eta(epoch)/(2*sigma2)*sum_Qall);
                case 'flat'
                    sum_Gy=0;sum_Gall=0;
                    for J = 1:nb_prot
                        wJ = model.w(:,:,J);
                        c_wJ = model.c_w(J);
                        sum_Gall= sum_Gall+probx(J)*((xi-wJ)^2)*G;
                        if c_wJ == c_xi
                            sum_Gy = sum_Gy+proby(J)*((xi-wJ)^2)*G;
                        end
                    end
                    model.omega=G - epsilons(epoch)/sigma2*sum_Gy+epsilons(epoch)/sigma2*sum_Gall;
                    %model.omega = model.omega / sqrt(sum(diag(model.omega'*model.omega)));                    
                    %model.omega=G - Eta(epoch)/sigma2*sum_Qy+Eta(epoch)/sigma2*sum_Qall;
                case 'polar'
                    lambda = 0.5;
                    sum_Uy=0;sum_Uall=0;sum_Ry=0;sum_Rall=0;
                    for J = 1:nb_prot
                        wJ = model.w(:,:,J);
                        c_wJ = model.c_w(J);
                        sum_Uall = sum_Uall+probx(J)*(eye(size(trainSet_log,1))-U_w*U_w')*((xi-wJ)^2)*U_w*(R_w)^2;
                        sum_Rall = sum_Rall+probx(J)*(R_w*U_w'*((xi-wJ)^2)*U_w*R_w);
                        if c_wJ == c_xi
                            sum_Uy = sum_Uy+proby(J)*(eye(size(trainSet_log,1))-U_w*U_w')*((xi-wJ)^2)*U_w*(R_w)^2;
                            sum_Ry = sum_Ry+proby(J)*(R_w*U_w'*((xi-wJ)^2)*U_w*R_w);
                        end
                    end
                    [qr_Q,~] = qr(U_w-lambda*epsilons(epoch)/sigma2*sum_Uy+lambda*epsilons(epoch)/sigma2*sum_Uall,0);
                    model.polar_U = qr_Q;
                    model.polar_R = R_w*expm(-(1-lambda)*epsilons(epoch)/(4*sigma2)*sum_Ry+(1-lambda)*epsilons(epoch)/(4*sigma2)*sum_Rall);
            end
        end
        
       % learning sigma2
       % adapt
       sigma = sqrt(model.sigma2);
       dSigma = 1/model.sigma2*sigma*probt*dist';
       sigma = sigma + 0.01*alphas(epoch)*dSigma;
       model.sigma2 = sigma^2; 
    end
    sigma2s(epoch+1) = model.sigma2;
    if nout>=2,
         % costs requested
         costs(epoch+1) = RiemanPLVQ_costfun(trainSet_log,trainLab,model);                      
        if nout>=3,
             % train error requested
             estimatedLabels = PLVQ_LogEu_classify(trainSet, model); % error after epoch
            trainError(epoch+1) = sum( trainLab ~= estimatedLabels )/nb_samples;
          
            if nout>=4,
            %   test error requested
                if ~isempty(testSet)
                    estimatedLabels = PLVQ_LogEu_classify(testSet(1:end-1,1:end-1,:), model); % error after initialization
                    testError(epoch+1) = sum( squeeze(testSet(end,end,:)) ~= estimatedLabels )/length(estimatedLabels);
                end 
               
           end
        end
    end
end

%%% output of the training
initialization.sigma2s = sigma2s; 
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {initialization};
		case(2)
            varargout(k) = {costs};
        case(3)
			varargout(k) = {trainError};
		case(4)
			varargout(k) = {testError};
	end
end

% figure;
% plotData(trainSet,trainLab);
% plotPrototypes(model.w,model.c_w);
% hold off;
