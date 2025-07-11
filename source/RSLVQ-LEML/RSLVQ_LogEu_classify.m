function [predictLabel] = RSLVQ_LogEu_classify(testP, model)
%%%GLVQ_LogEu_classify.m - classifies the given data with the given model
%  example for usage:
%  trainSet is n times n times m array, containing m  n times n SPD matrix
%  trainLab = [1;1;2;...];
%  model=GLVQ_LogEu_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = GLVQ_LogEu_classify(trainSet, model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  testP :  matrix array with training samples in its 3rd dimension
%  model    : GLVQ_LogEu model with prototypes w their labels c_w, and
%  omega under quotient manifold with flat metric,or Q under SPD manifold
%  with log-Euclidean metric
% 
% output    : the estimated labels
%  
% Fengzhen Tang (adapted from the code written by Kerstin Bunte available ...
% at http://matlabserver.cs.rug.nl/gmlvqweb/web/)
% tangfengzhen@sia.cn
% Thursday Sep 03 12:00 2020
testSet = spd2logm(testP);

d = computeDistanceLogEu(testSet,model.w,model);
[min_v,min_id] = min(d,[],2);
predictLabel = model.c_w(min_id);

