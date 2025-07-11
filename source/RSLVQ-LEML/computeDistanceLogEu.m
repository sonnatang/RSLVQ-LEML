function distances = computeDistanceLogEu(X, W, model)
%computeDistanceLogEu.m - Compute Log Eucliean metric learning induced geodesic distance between 
%samples X, and prototpyes W
% input: 
%  X : instances organized in matrix array of size n*n*m, 
%      containing m instances, each instance is an is n times n SPD matrix;
%  W: prototypes organized in matrix array of size n*n*M, 
%     containing M prototypes, each prototype is an  n * n SPD matrix;
%model    : GLVQ_LogEu model with prototypes w their labels c_w, and
%  omega under quotient manifold with flat metric,or Q under SPD manifold
%  with log-Euclidean metric
% Fengzhen Tang
% tangfengzhen@sia.cn
% Thursday Sep 03 12:00 2020
nb_samples = size(X,3);
nb_w = size(W,3);
distances = zeros(nb_samples,nb_w);

if isfield(model,'Q')
    Q = model.Q;
elseif isfield(model,'omega')
    Q = model.omega'*model.omega;
elseif isfield(model,'polar_U')
    Q = model.polar_U*(model.polar_R)^2*(model.polar_U)';
else
    printf('wrong in computeDistanceLogEu')
end

for i = 1:nb_samples
    for j = 1:nb_w
        logdiff = (X(:,:,i)) - (W(:,:,j));
        distances(i,j) = trace(Q*logdiff*logdiff);
    end
end

