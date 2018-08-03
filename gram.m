function G = gram(X1, X2, kernel, param1, param2)

% Computes the Gram-matrix of data points X1 and X2 using the specified kernel
% function.

    % Check inputs
    if size(X1, 2) ~= size(X2, 2)
        error('Dimensionality of both datasets should be equal');
    end

    % If no kernel function is specified
    if nargin == 2 || strcmp(kernel, 'none')
        kernel = 'linear';
    end
    
    switch kernel
        
        % Linear kernel
        case 'linear'
            G = X1 * X2';
        
        % Gaussian kernel
        case 'gauss'
            if ~exist('param1', 'var'), param1 = 1; end
            G = L2_distance(X1', X2');
            G = exp(-(G.^2 / (2 * param1.^2)));
                        
        % Polynomial kernel
        case 'poly'
            if ~exist('param1', 'var'), param1 = 1; param2 = 3; end
%             c = 1/(param1 + 1)^param2;
            G = ((X1 * X2') + param1) .^ param2;
            

        case 'hint' % histogram intersection
            G = hist_isect0(X1, X2) ; 
            
        case 'tanh'
            G = tanh(param1 + param2*(X1 * X2') );
        
        case 'exp'
            G = L2_distance(X1', X2');
            G = exp(  -( G/ ( param1) )  );

        case 'wave'
            G = L2_distance(X1', X2');
            G = (param1./G).*sin(G/param1);
        otherwise
            error('Unknown kernel function.');
    end
    

end


function K = hist_isect0(x1, x2)

% Evaluate a histogram intersection kernel, for example
%
    K = hist_isect(x1, x2);
%
% where x1 and x2 are matrices containing input vectors, where 
% each row represents a single vector.
% If x1 is a matrix of size m x o and x2 is of size n x o,
% the output K is a matrix of size m x n.
% 
% n = size(x2,1);
% m = size(x1,1);
% K = zeros(m,n);
% 
% if (m <= n)
%    for p = 1:m
%        nonzero_ind = find(x1(p,:)>0);
%        tmp_x1 = repmat(x1(p,nonzero_ind), [n 1]); 
%        K(p,:) = sum(min(tmp_x1,x2(:,nonzero_ind)),2)';
%    end
% else
%    for p = 1:n
%        nonzero_ind = find(x2(p,:)>0);
%        tmp_x2 = repmat(x2(p,nonzero_ind), [m 1]);
%        K(:,p) = sum(min(x1(:,nonzero_ind),tmp_x2),2);
%    end
% end
end