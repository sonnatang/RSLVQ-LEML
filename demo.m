%%% demo of RSLVQ with log-Euclidean metric
rs = RandStream.create('mt19937ar','seed',333);
RandStream.setGlobalStream(rs);
addpath(genpath('./source'))
addpath(genpath('./data'))

fname = 'CV_normF10_30CA01';


load([fname '.mat']);

trainIdx = ~testIdx;
% testIdx(2051:3280,:) = 0;
% trainIdx(2051:3280,:) = 0;
trainP = P(:,:,trainIdx);
trainLab = Label(trainIdx);

testP = P(:,:,testIdx);
testLab = Label(testIdx);

testSetLab = zeros(size(testP,1)+1,size(testP,2)+1,size(testP,3));
testSetLab(1:end-1,1:end-1,:) = testP;
testSetLab(end,end,:) = testLab;

classes = unique(trainLab);


%% train RSLVQ-LEML-FM
nPrototype = 1;%needs to specify
nb_epochs = 10;%needs to specify
sigma2 = 0.5;% needs to specify
% time_train = cputime;
[model,parametersetting,cost] = RSLVQ_LogEu_train(trainP, trainLab,...
 'PrototypesPerClass',nPrototype,...
'nb_epochs',nb_epochs,'Qmetric','flat',...
'testSet',testSetLab,'sigma2',sigma2);
% trtime_train = cputime-time_train
%%%training accuracy
predtrainLab  = RSLVQ_LogEu_classify(trainP,model);
trainacc = evaluation_measures(trainLab,predtrainLab,classes, 'RA' );
trainkappa= evaluation_measures(trainLab,predtrainLab,classes, 'KAPPA' );
fprintf('RSLVQ-LEML-FM: accuracy on the training set: %f\n',trainacc);
fprintf('RSLVQ-LEML-FM: kappa on the training set: %f\n',trainkappa);

%%%test accuracy
% time_test = cputime;
[predLab] = RSLVQ_LogEu_classify(testP, model);
testacc = evaluation_measures(testLab, predLab,classes, 'RA' );
testkappa = evaluation_measures(testLab, predLab,classes, 'KAPPA' );
% trtime_test = cputime-time_test
fprintf('RSLVQ-LEML-FM: accuracy on the test set: %f\n',testacc);
fprintf('RSLVQ-LEML-FM: kappa on the test set: %f\n',testkappa);

%% train PLVQ-LEML-polar
nPrototype = 1;%needs to specify
nb_epochs = 10;%needs to specify
sigma2 = 0.5;% needs to specify
% time_train = cputime;
[model,parametersetting,cost] =RSLVQ_LogEu_train(trainP, trainLab,...
 'PrototypesPerClass',nPrototype,...
'nb_epochs',nb_epochs,'Qmetric','polar',...
'testSet',testSetLab,'sigma2',sigma2);
% trtime_train = cputime-time_train
%%%training accuracy
predtrainLab  = RSLVQ_LogEu_classify(trainP,model);
trainacc = evaluation_measures(trainLab,predtrainLab,classes, 'RA' );
trainkappa= evaluation_measures(trainLab,predtrainLab,classes, 'KAPPA' );
fprintf('RSLVQ-LEML-LEM: accuracy on the training set: %f\n',trainacc);
fprintf('RSLVQ-LEML-LEM: kappa on the training set: %f\n',trainkappa);

%%%test accuracy
% time_test = cputime;
[predLab] = RSLVQ_LogEu_classify(testP, model);
testacc = evaluation_measures(testLab, predLab,classes, 'RA' );
testkappa = evaluation_measures(testLab, predLab,classes, 'KAPPA' );
% trtime_test = cputime-time_test
fprintf('RSLVQ-LEML-LEM: accuracy on the test set: %f\n',testacc);
fprintf('RSLVQ-LEML-LEM: kappa on the test set: %f\n',testkappa);

%% train PLVQ-LEML-LEM
nPrototype = 1;%needs to specify
nb_epochs = 10;%needs to specify
sigma2 = 0.5;% needs to specify
% time_train = cputime;
[model,parametersetting,cost] =RSLVQ_LogEu_train(trainP, trainLab,...
 'PrototypesPerClass',nPrototype,...
'nb_epochs',nb_epochs,'Qmetric','log-euclid',...
'testSet',testSetLab,'sigma2',sigma2);
% trtime_train = cputime-time_train
%%%training accuracy
predtrainLab  = RSLVQ_LogEu_classify(trainP,model);
trainacc = evaluation_measures(trainLab,predtrainLab,classes, 'RA' );
trainkappa= evaluation_measures(trainLab,predtrainLab,classes, 'KAPPA' );
fprintf('RSLVQ-LEML-LEM: accuracy on the training set: %f\n',trainacc);
fprintf('RSLVQ-LEML-LEM: kappa on the training set: %f\n',trainkappa);

%%%test accuracy
% time_test = cputime;
[predLab] = RSLVQ_LogEu_classify(testP, model);
testacc = evaluation_measures(testLab, predLab,classes, 'RA' );
testkappa = evaluation_measures(testLab, predLab,classes, 'KAPPA' );
% trtime_test = cputime-time_test
fprintf('RSLVQ-LEML-LEM: accuracy on the test set: %f\n',testacc);
fprintf('RSLVQ-LEML-LEM: kappa on the test set: %f\n',testkappa);



%% train RSLVQ-LEM

nPrototype = 1;%needs to specify
nb_epochs = 10;%needs to specify
initsigma2 = 5;%needs to specify
%implementing RSLVQ with log Euclidean metric
%   converting SPD matrices into their logorithm domain and then
%   apply standard RSLVQ learning rule

trainSet = spd2vec(trainP);

RSLVQparams = struct('PrototypesPerClass',nPrototype,'regularization',0);

[RSLVQ_model,settting,Cost] = RSLVQ_train(trainSet, trainLab,'PrototypesPerClass',RSLVQparams.PrototypesPerClass,...
    'nb_epochs',nb_epochs,'sigma2',initsigma2);

predtrainLab = RSLVQ_classify(trainSet, RSLVQ_model);
trainacc = evaluation_measures(trainLab,predtrainLab,classes, 'RA' );
trainkappa= evaluation_measures(trainLab,predtrainLab,classes, 'KAPPA' );
fprintf('RSLVQ-LEM: accuracy on the training set: %f\n',trainacc);
fprintf('RSLVQ-LEM: kappa on the training set: %f\n',trainkappa);

testSet = spd2vec(testP);
predLab = RSLVQ_classify(testSet, RSLVQ_model);
testacc = evaluation_measures(testLab, predLab,classes, 'RA' );
testkappa = evaluation_measures(testLab, predLab,classes, 'KAPPA' );
% trtime_test = cputime-time_test
fprintf('RSLVQ-LEML-LEM: accuracy on the test set: %f\n',testacc);
fprintf('RSLVQ-LEML-LEM: kappa on the test set: %f\n',testkappa);



