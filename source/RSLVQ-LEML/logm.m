function Out = logm(X)
%matrix logrithm
[V D] = eig(X);
Out = V*diag(log(diag(D)))*V';
