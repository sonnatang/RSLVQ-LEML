function [ vec] = spd2vec(P)
 %%vectorize data
[n,n1,nb_samples] = size(P);
if n~=n1
    fprintf('wrong');
end
vec = zeros(nb_samples, 0.5*n*(n+1));
index = reshape(triu(ones(n)),n*n,1)==0;
for i=1:nb_samples
    X = logm(P(:,:,i));
    vec(i,:) = X(not(index));
end





