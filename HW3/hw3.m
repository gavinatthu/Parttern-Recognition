% E-M Algorithm
clear;
clc;
data = importdata('emdata3.mat');
m = 4;
eps = 1e-6;
[param, history, ll] = em_mix2(data,m,eps);

% BIC(Bayesian Information Criterion) Step
n = size(data,1);
k = 3 * m + 2;
BIC = -2 * ll(end) + log(n) * k;

