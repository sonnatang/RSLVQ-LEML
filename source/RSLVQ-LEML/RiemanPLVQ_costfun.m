function cost = RiemanPLVQ_costfun(trainSet,trainLab,model)
%%RiemanPLVQ_costfun.m - computes the costs for a given training set and
% RPLVQ model
%
% input: 
%  testSet :  matrix array with training samples in its 3rd dimension
%  model    : RiemanPLVQ model with prototypes w their labels c_w 
%  sigma2: the teperature parameter
%  output    : the estimated labels
%  
% @Fengzhen Tang
% tangfengzhen@sia.cn
% Monday Dec 7 08:27 2020
nb_samples = length(trainLab);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end
LabelEqPrototype = bsxfun(@eq,trainLab,model.c_w');
dists = computeDistanceLogEu(trainSet, model.w,model);
% prob = exp(-0.5*dists/model.sigma2);
% Dcorrect = prob;
% Dcorrect(~LabelEqPrototype) = 0;
% probture = sum(Dcorrect,2);
% norm = sum(prob,2);
% logR = log(probture) - log(norm);
% cost = -sum(logR);

%% The above code is equivalent. The following process is only to change the scale. It is useful when sigma2 becomes very
% small
fs = -dists/(2*model.sigma2);
 
fmax = max(fs,[],2);
nf = bsxfun(@minus,fs,fmax);

p = exp(nf);
probx = sum(p,2);

fy = fs;
fy(~LabelEqPrototype) = -10^6;

fymax = max(fy,[],2);
nfy = bsxfun(@minus,fy,fymax);

py = exp(nfy);
py(~LabelEqPrototype) = 0;
proby = sum(py,2);

logr = log(proby) - log(probx);
cost = -sum(logr);

end


