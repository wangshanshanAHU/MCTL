function  [Z,P,Ks, Kt,KY_test]=HSIC_MC(X,Xs_train,Xt_train,Xt_test,Graph_Xt_train_num,Graph_Xt_test_num,target_label_train,target_label_test,Kernel)
theta=1;        %%MC系数
tau=1;         %%MMD系数
alpha=0.3;      %%Z梯度下降系数
mu=0.1;         %%R更新系数
lambdaJ=1;      %%辅助变量J系数
max_mu = 10^6;
rho =1.01;

 Ns=size(Xs_train,2);
 Nt=size(Xt_train,2);
 Nt_test=size(Xt_test,2);
I=eye(size(X,2));     
 H=eye(Nt)-1/Nt*ones(Nt,1)*ones(1,Nt);
 Z =ones(Ns,Nt); 
 l=ones(Nt,Nt);
 lns=ones(Ns,1);
 lnt=ones(Nt,1);
 R1 = zeros(Ns,Nt);
 J=zeros(Ns,Nt);
Y = Construct_Y(target_label_train,length(target_label_train)); 
Y2 = Construct_Y(target_label_test,length(target_label_test)); 

   switch Kernel 
       case'linear'    
       kervar1=0.5;% free parameter
       kervar2=10;% no use 
       case  'gauss'
       kervar1=1.2;% free parameter
       kervar2=10;% no use
   end    
       X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]); 
       K    = gram(X',X',Kernel,kervar1,kervar2);
       K =max(K,K');
       K = K./repmat(sqrt(sum(K.^2)),[size(K,1) 1]);  
       Kxt  = gram(Xt_train',Xt_train',Kernel,kervar1,kervar2);
       Kxt = Kxt./repmat(sqrt(sum(Kxt.^2)),[size(Kxt,1) 1]);  
       Kxs  = gram(Xs_train',Xs_train',Kernel,kervar1,kervar2);
       Kxs = Kxs./repmat(sqrt(sum(Kxs.^2)),[size(Kxs,1) 1]); 
       Kt   = gram(X',Xt_train',Kernel,kervar1,kervar2);
       Kt = Kt./repmat(sqrt(sum(Kt.^2)),[size(Kt,1) 1]);  
       Ks   = gram(X',Xs_train',Kernel,kervar1,kervar2);
       Ks = Ks./repmat(sqrt(sum(Ks.^2)),[size(Ks,1) 1]);  
       KY_test  = gram(X',Xt_test',Kernel,kervar1,kervar2);
       KY_test = KY_test./repmat(sqrt(sum(KY_test.^2)),[size(KY_test,1) 1]); 
            
       k=Graph_Xt_train_num;
       k2=Graph_Xt_test_num;
       s_pa=1;
          [L,D,W]=Graph_Laplacian(Xt_train', k, s_pa);       
  iter=0;     
 while iter<30    
       iter=iter+1;
       
     %%% Update P     
       Ph1= theta/(Nt^2)*Ks*Z*D*Z'*Ks';
       Ph2= theta/(Nt^2)*Kt*D*Kt';
       Ph3= theta/(Nt^2)*Ks*Z*W*Kt';
       Ph4=theta/(Nt^2)*Kt*W*Z'*Ks'; 
       Ph5= tau/(Nt^2)*Ks*Z*l*Z'*Ks';
       Ph6= tau/(Nt^2)* Ks*Z*l*Kt';
       Ph7= tau/(Nt^2)*Kt*l*Z'*Ks';
       Ph8= tau/(Nt^2)*Kt*l*Kt';
       Ph=(Ph1+Ph2-Ph3-Ph4+Ph5-Ph6-Ph7+Ph8);
       Ph=(Ph+Ph')/2;
       P = UpdateP(K+0.001*I,Ph,size(Ph,2)); 
        A=P'*(K+0.001*I)*P;%%test
      %%% Update Z    
       Zh1=2*theta/(Nt^2)*Ks'*P*P'*Ks*Z*D;
       Zh2=2*theta/(Nt^2)*Ks'*P*P'*Kt*W;
       Zh3=R1+mu*(Z-J);
       Zh4=2*tau/(Nt^2)*Ks'*P*P'*Ks*Z*l;
       Zh5=2*tau/(Nt^2)*Ks'*P*P'*Kt*l;
       Derta_Zold=Zh1-Zh2+Zh3+Zh4-Zh5; 
       Derta= Derta_Zold;
       Z_iter=alpha*Derta; 
       Z=Z-Z_iter;   
      
   %%% Update J
      ta = lambdaJ/mu;
    temp_J = Z + R1/mu;
    [U,sigma,V] = svd(temp_J,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>ta));
    if svp>=1
        sigma = sigma(1:svp)-ta;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';

    %%% Update R1 
    R1 = R1+mu*(Z-J);         
    %%% updating mu
    mu = min(rho*mu,max_mu);
    
 end  
end    

function [P] = UpdateP(K,Ph,C)
  ReducedDim = C;  
   Ph_eig=(Ph+Ph')/2;
   K_eig=(K+K')/2;
[eigV, eigD] = eig(Ph_eig,K_eig);
[eigD, d_site] = sort(diag(eigD),'ascend');
V = eigV(:,d_site(1:ReducedDim));
P = V;
end

function Y = Construct_Y(gnd,num_l)
%%
% gnd:标签向量；
% num_l:表示有标签样本的数目；
% Y:生成的标签矩阵；
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end  
    end
end
end
