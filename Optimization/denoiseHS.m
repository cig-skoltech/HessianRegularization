function [x,P,iter,L]=denoiseHS(y,lambda,varargin)

%Image denoising with a Hessian Schatten-Norm regularizer.

% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values description
% =========================================================================
% y             Noisy blured image.
% lambda        Regularization penalty parameter.
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values' description
%
% img           Original Image. (For the compution of the ISNR improvement)
% maxiter       Number of iterations (Default: 100)
% tol           Stopping threshold for denoising (Default:1e-4)
% optim         The type of gradient-based method used {'fgp'|'gp'}. 
%               (Fast Gradient Projection or Gradient Projection) 
%               (Default: 'fgp')
% P             Initialization of the dual variables. (Default: zeros([size(y) 3])).
% verbose       If verbose is set on then info for each iteration are
%               printed on screen. (Default: false)
% bounds        Minimize the Objective with a box constraint on the
%               solution (Default: [-inf +inf])
% bc            Boundary conditions for the differential operators.
%               {'reflexive'|'circular'|'zero'} (Default: 'reflexive')
% snorm         Specifies the type of the Hessian Schatten norm.
%               {'spectral'|'nuclear'|'frobenius'|'Sp'}. (Default:
%               'frobenius'). If snorm is set to Sp then the order of
%               the norm has also to be specified.
% order         The order of the Sp-norm. 1<order<inf. For order=1 set
%               snorm='nuclear', for order=2 set snorm='frobenius' and 
%               for order=inf set snorm='spectral'.
% =========================================================================
% ========================== OUTPUT PARAMETERS ============================
% x             Denoised image.
% P             The solution of the dual problem.
% iter          Number of iterations until convergence of the algorithm.
% L             Lipschitz constant of the dual objective.
% =========================================================================
%
% Author: stamatis.lefkimmiatis@epfl.ch
%
% =========================================================================

[maxiter,L,tol,optim,verbose,img,bounds,P,bc,snorm,order]=process_options(...
  varargin,'maxiter',100,'L',64,'tol',1e-4,'optim','fgp','verbose',...
  false,'img',[],'bounds',[-inf +inf],'P',zeros([size(y) 3]),'bc',...
  'reflexive','snorm','frobenius','order',[]);

if isempty(L)
  L=64/1.25;%Lipschitz constant
end


if isequal(snorm,'Sp') && isempty(order)
  error('denoiseHS: The order of the Sp-norm must be specified.');
end

if isequal(snorm,'Sp') && isinf(order)
  error('denoiseHS: Try spectral norm for the type-norm instead.');
end

if isequal(snorm,'Sp') && order==1
  error('denoiseHS: Try nuclear norm for the type-norm instead.');
end

if isequal(snorm,'Sp') && order < 1
  error('denoiseHS: The order of the Sp-norm should be greater or equal to 1.');
end

count=0;
flag=false;

if verbose
  fprintf('******************************************\n');
  fprintf('**  Denoising with Hessian Regularizer  **\n');
  fprintf('******************************************\n');
  fprintf('#iter     relative-dif   \t fun_val         Duality Gap        ISNR\n')
  fprintf('====================================================================\n');
end
switch optim
  case 'fgp'
    t=1;
    F=P;
    for i=1:maxiter
      K=y-lambda*AdjHessianOp2D(F,bc);
      %Pnew=F-step*lambda*HessianOp(lambda*AdjHessianOp(F)-y);
      %step=1/(L*lambda^2)==>
      %Pnew=F-(1/(L*lambda))*HessianOp(lambda*AdjHessianOp(F)-y);
      Pnew=F+(1/(L*lambda))*HessianOp2D(project(K,bounds),bc);
      Pnew=projectLB(Pnew,snorm,order);
      
      re=norm(Pnew(:)-P(:))/norm(Pnew(:));%relative error
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      tnew=(1+sqrt(1+4*t^2))/2;
      F=Pnew+(t-1)/tnew*(Pnew-P);
      P=Pnew;
      t=tnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjHessianOp2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda*AdjHessianOp2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc,snorm);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
        end
      end
      
      if count >=5
        flag=true;
        iter=i;
        break;
      end
    end
    
  case 'gp'
    
    for i=1:maxiter
      K=y-lambda*AdjHessianOp2D(P,bc);
      %Pnew=P-step*lambda*HessianOp(lambda*AdjHessianOp(P)-y);
      %step=1/(L*lambda^2)==>
      %Pnew=P-(1/(L*lambda))*HessianOp(lambda*AdjHessianOp(P)-y);
      Pnew=P+(1/(L*lambda))*HessianOp2D(project(K,bounds),bc);
      Pnew=projectLB(Pnew,snorm,order);
      
      re=norm(Pnew(:)-P(:))/norm(Pnew(:));%relative error
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      P=Pnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjHessianOp2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          ISNR=20*log10(norm(y-img,'fro')/norm(x-img,'fro'));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
        else
          k=y-lambda*AdjHessianOp2D(P,bc);
          x=project(k,bounds);
          fun_val=cost(y,x,lambda,bc,snorm);
          dual_fun_val=dualcost(y,k,bounds);
          dual_gap=(fun_val-dual_fun_val);
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
        end
      end
      
      if count >=5
        flag=true;
        iter=i;
        break;
      end
    end
end

if ~flag
  iter=maxiter;
end

x=project(y-lambda*AdjHessianOp2D(P,bc),bounds);

function Hf=HessianOp2D(f,bc)

[r,c]=size(f);

fxx=(f-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc));
fyy=(f-2*shift(f,[0,-1],bc)+shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));

%Compute Hf (Apply to image f the Hessian Operator)
%Hf will be a cube (3D image)
Hf=zeros(r,c,3);
Hf(:,:,1)=fxx;
Hf(:,:,2)=fxy;
Hf(:,:,3)=fyy;


function HaA=AdjHessianOp2D(A,bc)
%A is symmetric in the 3rd dimension, that is A(:,:,2)=A(:,:,3)
%Therefore we dont have to store the redundant dimension for A, and we
%provide only a cropped version of A without the A(:,:3) info.

Axx=A(:,:,1);
Axx=(Axx-2*shiftAdj(Axx,[-1,0],bc)+shiftAdj(Axx,[-2,0],bc));
A2xy=2*A(:,:,2);
A2xy=(A2xy-shiftAdj(A2xy,[0,-1],bc)-shiftAdj(A2xy,[-1,0],bc)+...
  shiftAdj(A2xy,[-1,-1],bc));
Ayy=A(:,:,3);
Ayy=(Ayy-2*shiftAdj(Ayy,[0,-1],bc)+shiftAdj(Ayy,[0,-2],bc));


%Compute H*A (Apply to cube A the adjoint of the Hessian Operator)
%H*A will be an image
HaA=Axx+A2xy+Ayy;

function Ap=projectLB(A,snorm,order)

if nargin < 3
  order=[];
end

switch snorm
  case 'spectral'
    Ap=projectSpMat2x2(A,1,1);
        
  case 'frobenius'
    Ap=projectSpMat2x2(A,2,1);
        
  case 'nuclear'
    Ap=projectSpMat2x2(A,inf,1);
        
  case 'Sp'
    Ap=projectSpMat2x2(A,order/(order-1),1);
        
  otherwise
    error('denoiseHS::Unknown type of norm.');
end


function Px=project(x,bounds)
lb=bounds(1);%lower box bound
ub=bounds(2);%upper box bound

if isequal(lb,-Inf) && isequal(ub,Inf)
  Px=x;
elseif isequal(lb,-Inf) && isfinite(ub)
  x(x>ub)=ub;
  Px=x;
elseif isequal(ub,Inf) && isfinite(lb)
  x(x<lb)=lb;
  Px=x;
else
  x(x<lb)=lb;
  x(x>ub)=ub;
  Px=x;
end


function [Q,Hnorm]=cost(y,f,lambda,bc,snorm,order)

if nargin < 6
  order=[];
end

fxx=(f-2*shift(f,[-1,0],bc)+shift(f,[-2,0],bc));
fyy=(f-2*shift(f,[0,-1],bc)+shift(f,[0,-2],bc));
fxy=(f-shift(f,[0,-1],bc)-shift(f,[-1,0],bc)+shift(f,[-1,-1],bc));

switch snorm
  case 'spectral'
    %Sum of the Hessian spectral radius
    Lf=fxx+fyy;%Laplacian of the image f
    Of=sqrt((fxx-fyy).^2+4*fxy.^2);% Amplitude of the Orientation vector
    Hnorm=sum(sum(0.5*(abs(Lf)+Of)));%Sum of the Hessian spectral radius
  case 'frobenius'
    Hnorm=sum(sqrt(fxx(:).^2+fyy(:).^2+2*fxy(:).^2));
  case 'nuclear'
    Lf=fxx+fyy;%Laplacian of the image f
    Of=sqrt((fxx-fyy).^2+4*fxy.^2);% Amplitude of the Orientation vector
    Hnorm=0.5*sum(abs(Lf(:)+Of(:))+abs(Lf(:)-Of(:)));
  case 'Sp'
    Lf=fxx+fyy;%Laplacian of the image f
    Of=sqrt((fxx-fyy).^2+4*fxy.^2);% Amplitude of the Orientation vector
    %e1=0.5*(Lf(:)+Of(:));
    %e2=0.5*(Lf(:)-Of(:));
    Hnorm=sum((abs(0.5*(Lf(:)+Of(:))).^(order)+abs(0.5*(Lf(:)-Of(:))).^(order)).^(1/order));
  otherwise
    error('denoiseHS::Unknown type of norm.');
end

Q=0.5*norm(y-f,'fro')^2+lambda*Hnorm;

function Q=dualcost(y,f,bounds)
r=f-project(f,bounds);
Q=0.5*(sum(r(:).^2)+sum(y(:).^2)-sum(f(:).^2));



