function distance = computeDistance(X, W, model)
nb_samples = size(X,1);
distance = zeros(nb_samples,length(model.c_w));

if isfield(model,'psis')
    if length(model.psis)~=size(W,1)
        classes = unique(model.c_w);
        for i = 1:size(W,1)
            matrixIdx = classes==model.c_w(i);
            distance(1:nb_samples,i) = sum((bsxfun(@minus, X, W(i,:))*model.psis{matrixIdx}').^2, 2);
        end
    else
        for i = 1:size(W,1)             
            distance(1:nb_samples,i) = sum((bsxfun(@minus, X, W(i,:))*model.psis{i}').^2, 2);
        end
    end
elseif isfield(model,'lambda')
        for i = 1:size(W,1)
%             delta = X-ones(nb_samples,1)*W(i,:);
%             distance(1:nb_samples,i) = sum( (( delta *model.lambda).*delta) ,2 );
            delta = bsxfun(@minus, X, W(i,:));
            distance(1:nb_samples,i) = sum(bsxfun(@times,delta.^2,model.lambda), 2);
%             distance(1:nb_samples,i) = sum( (( delta *model.lambda).*delta) ,2 );
        end
elseif isfield(model,'omega')
        % tic;
        for i = 1:size(W,1)
%             delta = X-ones(nb_samples,1)*W(i,:);
%             distance(1:nb_samples,i) = sum( (( delta *model.omega'*model.omega).*delta) ,2 );
            distance(1:nb_samples,i) = sum((bsxfun(@minus, X, W(i,:))*model.omega').^2, 2);
        end
        % disp(toc);   
else % no metric learning
    for i = 1:size(W,1)
         distance(1:nb_samples,i) = sum((bsxfun(@minus, X, W(i,:))).^2, 2);
    end
    
end