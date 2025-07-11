function [predictLabel] = RSLVQ_classify(testP, model)
%PREDICT_RIEMANGLVQ 此处显示有关此函数的摘要
%   此处显示详细说明

d = computeDistance(testP,model.w,model);
[min_v,min_id] = min(d,[],2);
predictLabel = model.c_w(min_id);

