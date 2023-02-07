%% Generate random binary quadratic program
d       = 10; % BQP with d variables
x       = msspoly('x',d); % symbolic decision variables using SPOTLESS
Q       = randn(d,d); Q = Q + Q'; % a random symmetric matrix
c       = randn(d,1);
f       = x'*Q*x + c'*x; % objective function of the BQP
h       = [x.^2 - 1]; % equality constraints of the BQP (binary variables)
g       = [x(1)]; % ask the first variable to be positive

%% Relax BQP into an SDP

addpath /home/agrobenj/mosek/10.0/toolbox/r2017a
problem.vars            = x;
problem.objective       = f;
problem.equality        = h; 
problem.inequality      = g;
kappa                   = 2; % relaxation order
[SDP,info]              = dense_sdp_relax(problem,kappa);

prob       = convert_sedumi2mosek(SDP.sedumi.At,...
                                  SDP.sedumi.b,...
                                  SDP.sedumi.c,...
                                  SDP.sedumi.K);
                              
[r, res] = mosekopt('write(dump.ptf)', prob);

%% loading SDP problem

load("sdp_test.mat")
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
kappa                   = 1; % relaxation order
[SDP,info]              = dense_sdp_relax(problem,kappa);

prob       = convert_sedumi2mosek(SDP.sedumi.At,...
                                  SDP.sedumi.b,...
                                  SDP.sedumi.c,...
                                  SDP.sedumi.K);
                              
[r, res] = mosekopt('write(dump.ptf)', prob);

%%

[r,res]    = mosekopt('minimize info',prob);
[Xopt,yopt,Sopt,obj] = recover_mosek_sol_blk(res,SDP.blk);

%%