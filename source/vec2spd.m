function [ psd ] = vec2spd(vec)
%UNTITLED2 
n = length(vec);
N = 0.5*(sqrt(1+8*n) - 1);
if 0.5*N*(N+1)~=n
    fprintf('wrong in vec to spd');
end
psd = zeros(N,N);
index = reshape(triu(ones(N)),N*N,1)==0;
psd(not(index))= vec;
psd = psd + psd' - diag(diag(psd));

