function cost = RSLVQ_costfun(trainSet,trainLab,model,regularization)
%RIEMANGLVQ_COSTFUN 此处显示有关此函数的摘要
%   此处显示详细说明
nb_samples = length(trainLab);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end
LabelEqPrototype = bsxfun(@eq,trainLab,model.c_w');
dists = computeDistance(trainSet, model.w,model);

fs = -dists/(2*model.sigma2);

% this is only to change the scale. It is useful when sigma2 becomes very
% small 
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
% c = zeros(nb_samples,1);
% for i=1:nb_samples
%     % select one training sample randomly
%         xi = trainSet(i,:);
%         c_xi = trainLab(i);
% 
%         dist = computeDistance(xi, model.w,model);
%         
%         % compute class probability
%         fs = -dist/(2*model.sigma2);
%         fsmax = max(fs);
%         prob = exp(fs-fsmax);
%         fsy = fs;
%         fsy(~(model.c_w == c_xi)) = [];
%         fsymax = max(fsy);
%         p = exp(fsy - fsymax);
%         c(i) = log(sum(p)) - log(sum(prob));  
% end
% co = -sum(c);
% [cost co]
%%

% prob = exp(-dists/(2*model.sigma2));
% Dcorrect = prob;
% Dcorrect(~LabelEqPrototype) = 0;
% probture = sum(Dcorrect,2);
% norm = sum(prob,2);
% 
% logR = log(probture) - log(norm);
% cost = -sum(logR);

end

