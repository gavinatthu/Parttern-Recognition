clear;
clc;
data = [[0.42, -0.087, 0.58];...
        %[-0.2, -3.3, -3.4];...
        [1.3, -0.32, 1.7];...
        %[0.39, 0.71, 0.23];...
        [-1.6, -5.3, -0.15];...
        %[-0.029, 0.89, -4.7];...
        [-0.23, 1.9, 2.2];...
        %[0.27, -0.3, -0.87];...
        [-1.9, 0.76, -2.1]];
        %[0.87, -1.0, -2.6]];
m = 1;
eps = 1e-4;
[param, history, ll] = em_mix(data,m,eps);
%[param2, history2, ll2] = em_mix(data(:,2:3),m,eps);
% BIC(Bayesian Information Criterion) Step
n = size(data,1);
k = 3 * m + 2;
BIC = -2 * ll(end) + log(n) * k;