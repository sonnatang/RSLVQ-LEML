
%%% demo file for running RSLVQ
%
% Fengzhen Tang (modified based on the code of Kerstin Bunte at...
%  available at http://matlabserver.cs.rug.nl/gmlvqweb/web/)
%tangfengzhen@sia.cn
% Fri Sep 04 07:39 2020
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
addpath(genpath('.'));
% load the data
% actData = 'Iris';
% load data/Iris.csv;
% data = Iris(:,1:end-1);
% label = Iris(:,end);
% nb_samples_per_class = 45;
actData = 'UCIsegmentation';
load data/segment.dat;
data = segment(:,1:end-1);
data(:,std(data)==0) = []; % feature 3 is constant -> exclude it
label= segment(:,end);
nb_samples_per_class = 300;

fprintf('Load the %s data set containing %i samples with %i features.\n',actData,size(data,1),size(data,2));
% draw randomly 100 samples from every class
nb_folds = 1;
indices = nFoldCrossValidation(data,'labels',label,'splits','random','nb_samples',nb_samples_per_class,'nb_folds',nb_folds,'comparable',1);
actSet = 1;
% extract the training set
trainSet = data(indices{actSet},:);
trainLab = label(indices{actSet});
% extract the test set
testIdx = 1:length(label);
testIdx(indices{actSet}) = [];
testSet = data(testIdx,:);
testLab = label(testIdx);

disp('preprocess the data using zscore');
[trainSet, zscore_model] = zscoreTransformation(trainSet);%nomalize the data to have zero mean and unit variance 
testSet = zscoreTransformation(testSet, 'parameter', zscore_model);


%% run RSLVQ

RSLVQparams = struct('PrototypesPerClass',2,'regularization',0);

[RSLVQ_model,settting,Cost,trainErr, testErr] = RSLVQ_train(trainSet, trainLab,'PrototypesPerClass',RSLVQparams.PrototypesPerClass,...
    'testSet',[testSet,testLab],'sigma2', 5,'nb_epochs',50,'sigmaadapt',1);
estimatedTrainLabels = RSLVQ_classify(trainSet, RSLVQ_model);
trainError = mean( trainLab ~= estimatedTrainLabels );
fprintf('RSLVQ: error on the train set: %f\n',trainError);
estimatedTestLabels = RSLVQ_classify(testSet, RSLVQ_model);
testError = mean( testLab ~= estimatedTestLabels );
fprintf('RSLVQ: error on the test set: %f\n',testError);



