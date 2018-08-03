function [Rate,T,Y]=decision_class(X,T)
% X: NxC
% Y: Nx1
if size(T,2)~=1
    for i=1:size(T,1)
        [v,p]=max(T(i,:));
        Tn(i)=p;
    end
    T=Tn';
end

   for i=1:size(X,1)
       [v,p]=max(X(i,:));
       Y(i)=p;
   end
   Rate=length(find(Y'==T))/length(T)*100;

    