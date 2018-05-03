function run_demo()

%Demonstration code for image deblurring using a Hessian Schatten-norm
%regularization approach. 

path=which('run_demo.m');
idx=strfind(path,filesep);
path=path(1:idx(end));
load([path 'demo_data']);

%Create blurred and noisy version of the image
y=imfilter(f,h,'conv','circular')+noise;


%Set of parameters for the deblurring algorithm (see deconvHS.m and
% denoiseHS.m for a description.)

options={'x_init',[],'iter',100,'den_iter',10,'verbose',true,...
  'showfig',false,'optim','mfista','den_thr',1e-3,'deconv_thr',1e-5,...
  'den_optim','fgp','bounds',[0 1],'img',f,'bc','reflexive'};


lambda=2.29e-4; % Regularization parameter
%Deblurring using the nuclear norm (Schatten norm of order 1).
%Note that deconvolution is performed by not using the original blur kernel
%but rather a perturbed version, which corresponds to adding Gaussian noise 
%of std=1e-3 to the original blur kernel. The reason is that this is a more
%realistic scenario since in practice we cannot have access to the real
%blur kernel but only an estimate. 
[x,~,fun,ISNR]=deconvHS(y,h_perturb,lambda,options{:},'snorm','nuclear');

figure(100);
imshow(f,[]);title('Ground-truth','fontsize',16);
figure(101);
imshow(y,[]);title('Blurred and noisy image','fontsize',16);
figure(102);imshow(x,[]);title('Restored image','fontsize',16);


figure(103);plot(fun);ylabel('Objective function','fontsize',16);
xlabel('Number of iterations','fontsize',16);
title('Evolution of the objective cost funtion','fontsize',16);

figure(104);plot(ISNR);ylabel('ISNR','fontsize',16);xlabel('Number of iterations','fontsize',16);
title('Evolution of the SNR improvement','fontsize',16);set(gca,'fontsize',16)