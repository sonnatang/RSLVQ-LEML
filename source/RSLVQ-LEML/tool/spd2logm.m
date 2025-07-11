function [ X] = spd2logm(P)
 %%transform spd matrix P to its logrithm domain
[n,n1,nb_samples] = size(P);
if n~=n1
    fprintf('wrong');
end
X = zeros(n,n1,nb_samples);
for i=1:nb_samples
    X(:,:,i) = logm(P(:,:,i));
end


