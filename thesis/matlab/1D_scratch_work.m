%%

clc; clear; close all; restoredefaultpath; % start clean
manoptpath  = 'CertifiablyRobustPerception/manopt';
utilspath   = 'CertifiablyRobustPerception/utils';
addpath(genpath(utilspath))
addpath(genpath('CertifiablyRobustPerception/STRIDE'))
addpath(genpath(manoptpath))
addpath(genpath('CertifiablyRobustPerception/solvers'))
addpath(genpath('CertifiablyRobustPerception/spotless')) % Use spotless for defining polynomials
addpath('CertifiablyRobustPerception/SDPRelaxations') % implementations for SDP relaxation

%%

addpath '/home/agrobenj/mosek/10.0/toolbox/r2017a';
load("1D_problem_4_landmarks_redundant.mat")
d       = length(Q);
x       = msspoly('x',d); % symbolic decision variables using SPOTLESS
c       = randn(d,1);
f       = x'*Q*x 
h       = []; % equality constraints of the BQP (binary variables)
bs = cast(bs, "double")
s = size(As);
for i = 1:s(1)
    h = [h (x'*squeeze(As(i, :, :))*x - bs(i))];
end

addpath /home/agrobenj/mosek/10.0/toolbox/r2017a
problem.vars            = x;
problem.objective       = f;
problem.equality        = h; 
problem.inequality      = [];
kappa                   = 4; % relaxation order
[SDP,info]              = dense_sdp_relax(problem,kappa);

prob       = convert_sedumi2mosek(SDP.sedumi.At,...
                                  SDP.sedumi.b,...
                                  SDP.sedumi.c,...
                                  SDP.sedumi.K);
                              
[r, res] = mosekopt('write(dump.ptf)', prob);

%At  = SDP.At{1};
%m   = size(SDP.b,1);
%n   = SDP.blk{1,2};
%A = zeros(m,n,n);
%for i = 1:m
%    A(i,:,:) = smat(SDP.blk,At(:,i));
%end



[r,res]    = mosekopt('minimize info',prob);
[Xopt,yopt,Sopt,obj] = recover_mosek_sol_blk(res,SDP.blk);

Xopt = Xopt{1};
Xopt(2:1+d,2:1+d);
depth = Xopt(2,1+d)



eigs = eig(Xopt(2:1+d, 2:1+d));
bar(eigs);
set(gca,'YScale','log')
eig_max = eigs(end);
eig_second= eigs(end-1);
log10_diff = log10(eig_max / eig_second)
primal = obj(1)