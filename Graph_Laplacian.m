function [L,D,W]=Graph_Laplacian(X, k, sigma)    %%%%%(K½üÁÚ)
%  disp('Constructing neighborhood graph...');
    if size(X, 1) < 4000
        W = L2_distance(X', X');
        % Compute neighbourhood graph
        [tmp, ind] = sort(W); 
%          G(G == 0) = 1;%%%%test
        for i=1:size(W, 1)
            W(i, ind((2 + k):end, i)) = 0; 
        end
         W=W+diag(-diag(W));
         W(W~=0)=1;
         W = max(W, W');             % Make sure distance matrix is symmetric
%          W=(W+W')/2;
         D=sum(W);
         D=diag(D);
         L=D-W;
   end      
end
        
        
        
%        G = sparse(double(G));
%     else
%         G = find_nn(X, k);
%     end
%     G = G .^ 2;
% 	G = G ./ max(max(G)); 
%     
%     % Compute weights (W = G)
% %     disp('Computing weight matrices...');
%     
%     % Compute Gaussian kernel (heat kernel-based weights)
%     G(G ~= 0) = exp(-G(G ~= 0) / (2 * sigma ^ 2));
%         
%     % Construct diagonal weight matrix
%     D = diag(sum(G, 2));
%     
%     % Compute Laplacian
%     L = D - G;
%     L(isnan(L)) = 0; D(isnan(D)) = 0;
% 	L(isinf(L)) = 0; D(isinf(D)) = 0;
%     
%     G=full(G);  
%     D=full(D);
%     L=full(L);
%     
% end