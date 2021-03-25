function [param,history,ll] = em_mix(data,number_of_components, eps)
%  runs EM to estimate a Gaussian mixture model.
%
%  another form: [param] = em_mix(data,number_of_components)
%
%  data: input data, N * dims double.
%  number_of_components: assumed number of components, int.
%  eps(optional): stopping criterion, float.
%
%  param: params of different gaussians, list.
%  history(optional): params of different gaussians during iteration, dict.
%  ll(optional): log-likelihood of the data during iteration, list.

% set stopping  criterion
if (nargin < 3), eps = min(1e-3,1/(size(data,1)*100));  end

% initial
param = initialize_mixture(data,number_of_components);

% plot
plot_all(data,param); 

history = {}; ll = [];

cont = 1; it = 1; log_likel = 0; 
while (cont)
    % one step
    [param,new_log_likel] = one_EM_iteration(data,param);
    
    history{length(history)+1}=param;
    ll(length(ll)+1)=new_log_likel;
    
    % plot
    plot_all(data,param); 
    
    % when to stop
    cont = (new_log_likel - log_likel)>eps*abs(log_likel);
    cont = cont | it<10; it = it + 1;

    log_likel = new_log_likel; 

    % uncomment if you wish to monitor the likelihood convergence
    %fprintf('%4d %f\n',it,log_likel);

    pause(0.1);
end

% --------------------------------------------------------
function [] = plot_all(data,param,dim1,dim2)
%  Plot the visualize results.
%
%  data: input data, N * dims double.
%  param: params of different gaussians, list.
%  dim1(optional): the first plot dim of data, int.
%  dim2(optional): the second plot dim of data, int.

if (nargin<3), dim1 = 1; dim2 = 2; end

set(0,'DefaultLineLineWidth',2)    
set(0,'DefaultAxesFontSize',14)        

% plot data
[n,d] = size(data);
log_prob = zeros(n,length(param)); 
for i=1:length(param)
 log_prob(:,i) = gaussian_log_prob(data,param(i))+log(param(i).p);
end
[value, index] = max(log_prob, [], 2);
scatter(data(:,dim1),data(:,dim2), [], index); hold on;
colormap jet;
myaxis = axis;

% plot gaussian
for i=1:length(param)
  plot_gaussian(param(i),dim1,dim2,'k');
  axis(myaxis);
end

hold off;

% --------------------------------------------------------
function [] = plot_gaussian(param,dim1,dim2,st)
%  Plot the gaussian distribution.
%
%  param: params of different gaussians, list.
%  dim1: the first plot dim of data, int.
%  dim2: the second plot dim of data, int.
%  st: the plot color, str.

[V,E] = eig(param.cov);
V = V';

s = diag(sqrt(E)); % standard deviations

t=(0:0.05:2*pi)'; 
X = s(dim1)*cos(t)*V(dim1,:)+s(dim2)*sin(t)*V(dim2,:); 
X = X + repmat(param.mean,length(t),1);

plot(X(:,1),X(:,2),st);

% --------------------------------------------------------
function [param,log_likel] = one_EM_iteration(data,param)
%  One iteration of EM.
%
%  data: input data, N * dims double.
%  param: params of different gaussians, list.
%
%  param: params of different gaussians, list.
%  log_likel: log-likelihood of the data, double.

[n,d] = size(data);

% E-step
log_prob = zeros(n,length(param)); 
for i=1:length(param)
 log_prob(:,i) = gaussian_log_prob(data,param(i))+log(param(i).p);
end

log_likel = sum(log(sum(exp(log_prob),2)));

post_prob = exp(log_prob);
post_prob = post_prob./repmat(sum(post_prob,2),1,length(param));

% M-step
for i=1:length(param)
  post_n = sum(post_prob(:,i));
    
  param(i).p = post_n/n; 
  param(i).mean = post_prob(:,i)'*data/post_n; 

  Z = data-repmat(param(i).mean,n,1);
  weighted_cov = (repmat(post_prob(:,i),1,d).*Z)'*Z;
  param(i).cov = weighted_cov/post_n;

end

% --------------------------------------------------------
function [log_prob] = gaussian_log_prob(data,param)
%  Log probability of gaussian distribution.
%
%  data: input data, N * dims double.
%  param: params of different gaussians, list.
%
%  log_prob: log probability of gaussian distribution, double.

[n,d] = size(data);
Ci = inv(param.cov);

Z = data-repmat(param.mean,n,1); 
log_prob = (-sum( (Z*Ci).*Z, 2 ) + log(det(Ci))-d*log(2*pi))/2;

% --------------------------------------------------------
function [param] = initialize_mixture(data,number_of_components)
%  Initialize params of different gaussians.
%
%  data: input data, N * dims double.
%  number_of_components: assumed number of components, int.
%
%  param: params of different gaussians, list.

[n,d] = size(data);

e = eig(cov(data)); 
C = median(e)*eye(d); % initial spherical covariance matrix

[rt,I] = sort(rand(n,1)); % random ordering of the examples

param=[];
for i=1:number_of_components

  prm.mean = data(I(i),:); % random point as the init mean
  prm.cov  = C; % spherical covariance as the init cov
  prm.p    = 1/number_of_components; % uniform freq as the init mix
  
  param = [param;prm];
end
