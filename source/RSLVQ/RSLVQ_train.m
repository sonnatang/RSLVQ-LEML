function [model, varargout] = RSLVQ_train(trainSet, trainLab, varargin)
%GMLVQ_trai.m - trains the Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=GMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = GMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
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
% output: the GMLVQ model with prototypes w their labels c_w and the matrix omega 
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information:
% Petra Schneider, Michael Biehl, Barbara Hammer: 
% Adaptive Relevance Matrices in Learning Vector Quantization. Neural Computation 21(12): 3532-3561 (2009)
% 
% K. Bunte, P. Schneider, B. Hammer, F.-M. Schleif, T. Villmann and M. Biehl, 
% Limited Rank Matrix Learning - Discriminative Dimension Reduction and Visualization, 
% Neural Networks, vol. 26, nb. 4, pp. 159-173, 2012.
% 
% P. Schneider, K. Bunte, B. Hammer and M. Biehl, Regularization in Matrix Relevance Learning, 
% IEEE Transactions on Neural Networks, vol. 21, nb. 5, pp. 831-840, 2010.
% 
% Kerstin Bunte (modified based on the code of Marc Strickert http://www.mloss.org/software/view/323/ and Petra Schneider)
% uses the Fast Limited Memory Optimizer fminlbfgs.m written by Dirk-Jan Kroon available at the MATLAB central
% kerstin.bunte@googlemail.com
% Fri Nov 09 14:13:52 CEST 2012
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
% changed by Fengzhen 2020-4-13 10:50
nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,1) & isnumeric(x));
p.addParamValue('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParamValue('initialPrototypes',[], @(x)(size(x,2)-1==size(trainSet,2) && isfloat(x)));
p.addParamValue('regularization',0, @(x)(isfloat(x) && x>=0));
p.addOptional('testSet', [], @(x)(size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));

%p.addOptional('optimization', 'fminlbfgs', @(x)any(strcmpi(x,{'sgd','fminlbfgs'})));
% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addParamValue('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle'))); % && (length(x)==2 || length(x)==p.Results.epochs)
p.addParamValue('sigma2', 5, @(x)(isfloat(x)));

%automatically adapted sigma2, added by Fengzhen Tang at Fri Sep 04 08:13 2020
p.addOptional('sigmaadapt', 0, @(x)(~(x-floor(x))));
p.addParamValue('learningRateSigma', [], @(x)(isfloat(x) || isa(x,'function_handle'))); % && (length(x)==2 || length(x)==p.Results.epochs)

%p.addParamValue('prior',[],@(x)(isfloat(x)));

p.CaseSensitive = true;
p.FunctionName = 'RSLVQ';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable,
    rng('default');
end

epsilon = 10^-3;

%%% set useful variables
[nb_samples,nb_features] = size(trainSet);

% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

classes = unique(trainLab);
nb_classes = length(classes);

testSet = p.Results.testSet;

% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the relevances with ',num2str(regularization)]);end

initialization = rmfield(p.Results, 'trainSet');
initialization.trainSet = [num2str(nb_samples),'x',num2str(nb_features),' matrix'];
initialization = rmfield(initialization, 'trainLab');
initialization.trainLab = ['vector of length ',num2str(length(trainLab))];
if ~isempty(testSet)
    initialization = rmfield(initialization, 'testSet');
    initialization.testSet = [num2str(size(testSet,1)),'x',num2str(size(testSet,2)),' matrix'];
end

%automatically adapted sigma2, added by Fengzhen Tang at Fri Sep
        % 04 08:13 2020
sigmaadapt = p.Results.sigmaadapt;

% Display all arguments.
disp 'Settings for RSLVQ:'
disp(initialization);

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
nb_ppc = p.Results.PrototypesPerClass;
if length(nb_ppc)~=nb_classes,
    nb_ppc = ones(1,nb_classes)*nb_ppc;
end

%prior = p.Results.prior; 
sigma2 = p.Results.sigma2; 
%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    % initialize near the class centers
    w = zeros(sum(nb_ppc),nb_features);
    c_w = zeros(sum(nb_ppc),1);
    actPos = 1;
    for actClass=1:nb_classes
        nb_prot_c = nb_ppc(actClass);
        classMean = mean(trainSet(trainLab==classes(actClass),:));
        % set the prototypes to the class mean and add a random variation
        % between -0.1 and 0.1
        w(actPos:actPos+nb_prot_c-1,:) = classMean(ones(nb_prot_c,1),:)+(rand(nb_prot_c,nb_features)*2-ones(nb_prot_c,nb_features))/10;
        c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
        actPos = actPos+nb_prot_c;
    end
else
    % initialize with given w
    w = p.Results.initialPrototypes(:,1:end-1);
    c_w = p.Results.initialPrototypes(:,end);
end

model = struct('w',w,'c_w',c_w,'sigma2',sigma2);
clear w c_w ;


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
%     alphas = arrayfun(@(x) alpha_start / (1+(x-1)*alpha_end), 1:nb_epochs);
end

%%automatically adapted sigma2, added by Fengzhen Tang at Fri Sep 04 08:13 2020
if sigmaadapt
    if isa(p.Results.learningRateSigma,'function_handle')
    % with a given function specified from the user
        alphas = arrayfun(p.Results.learningRateSigma, 1:nb_epochs);
    elseif length(p.Results.learningRateSigma)>2
        if length(p.Results.learningRateSigma)==nb_epochs
            betas = p.Results.learningRateSigma;
        else
            disp('The learning rate vector for the sigma does not fit the nb of epochs');
            return;
        end
    else
        % or use an decay with a start and a decay value
        if isempty(p.Results.learningRateSigma)
            initialization.learningRateSigma =[nb_features/100, nb_features/10000];
        end
        beta_start = initialization.learningRateSigma(1);
        beta_end = initialization.learningRateSigma(2);
        betas = arrayfun(@(x) beta_start * (beta_end/beta_start)^(x/nb_epochs), 1:nb_epochs);
    end

end
%%% initialize requested outputs
trainError = [];
costs = [];
testError = [];
if nout>=2,
    % costs requested
    disp('The computation of the costs is an expensive operation, do it only if you really need it!');
    costs = ones(1,nb_epochs+1);
    costs(1) = RSLVQ_costfun(trainSet, trainLab, model, regularization);

    if nout>=3,
            % train error requested
        trainError = ones(1,nb_epochs+1);
        estimatedLabels = RSLVQ_classify(trainSet, model); % error after initialization
        trainError(1) = sum( trainLab ~= estimatedLabels )/nb_samples;
        
       if nout>=4,
               % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                testError = ones(1,nb_epochs+1);
                estimatedLabels = RSLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError(1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
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
    % perform one sweep through trainingset
    for i=1:nb_samples
        % select one training sample randomly
        xi = trainSet(order(i),:);
        c_xi = trainLab(order(i));

        dist = computeDistance(xi, model.w,model);
        
        % compute class probability
        fs = -dist/(2*sigma2);
        fsmax = max(fs);
        prob = exp(fs-fsmax);
        probx = prob/sum(prob);
        fsy = fs;
        fsy(~(model.c_w == c_xi)) = [];
        fsymax = max(fsy);
        p = exp(fsy - fsymax);
        p = p/sum(p);
        proby = zeros(size(prob));
        proby(model.c_w == c_xi) = p;      
    
        
        %% the following codes are equavelent way to compute class probability
        % but as sigma gets closer to 0, the prob becomes Nan, but the
        % above code wont.
%         prob = exp(-dist/(2*sigma2)); 
%         proby1 = prob;
%         proby1(~(model.c_w == c_xi))=0;
%         proby1 = proby1/sum(proby1);       
%         probx1 = prob/sum(prob);
%         if isnan(sum(proby1)) | isnan(sum(probx1))
%             warning('nan probability, no update here');
%             continue;  % no updates
%         end
        %*************************
        %% update prototypes: 
        
        nb_prot = sum(nb_ppc);
        probt = zeros(1,nb_prot);
        for J = 1:nb_prot
            wJ = model.w(J,:);
            c_wJ = model.c_w(J);
            if c_wJ == c_xi
              probt(J) = (proby(J)-probx(J));
            else
               probt(J) = - probx(J);
            end
             
             % gradient of prototypes
             DJ  = (xi-wJ);
             XJ = alphas(epoch)/sigma2*probt(J)*DJ;
             %update prototypes
             model.w(J,:) = wJ + XJ;
        end
        
        % automatically adapted sigma2, added by Fengzhen Tang at Fri Sep
        % 04 08:13 2020
        if sigmaadapt 
            
           dSigma2 = 0.5/sigma2^2*probt*dist';

           sigma2 = sigma2 + betas(epoch)*dSigma2;
           if sigma2<epsilon
               sigma2 = epsilon;
           end

           model.sigma2 = sigma2;
        end
         
    end
    if nout>=2,
         % costs requested
         costs(epoch+1) = RSLVQ_costfun(trainSet, trainLab, model, regularization);                      
        if nout>=3,
             % train error requested
             estimatedLabels = RSLVQ_classify(trainSet, model); % error after epoch
              trainError(epoch+1) = sum( trainLab ~= estimatedLabels )/nb_samples;
          
            if nout>=4,
            %   test error requested
                if ~isempty(testSet)
                    estimatedLabels = RSLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                    testError(epoch+1) = sum( testSet(:,end) ~= estimatedLabels )/length(estimatedLabels);
                end 
               
           end
        end
    end
end

%%% output of the training
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