function [F_hat, tmin] = patch_lmi(As, Bs, Ak, Bk, eta, beta, kappa)
%%%%%%%%%%%%%%%%%%%%%%  DOC HELP  %%%%%%%%%%%%%%%%%%%%%%
%% Inputs
%
%      Ac :  A(s) in continuous form      -- 4x4
%      Bc :  B(s) in continuous form      -- 4x1
%      Ak :  A(s) in discrete form        -- 4x4
%      Bk :  B(s) in discrete form        -- 4x1
%
%% Outputs
%   F_hat :  Feedback control gain        -- 1x4
%    tmin :  Feasibility of LMI solution  -- 1x4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SP = [13.3425, 6.73778, 16.2166, 3.47318;
      6.73778, 3.94828, 9.69035, 2.09032;
      16.2166, 9.69035, 25.9442, 5.31439;
      3.47318, 2.09032, 5.31439, 1.16344];

beta1 = 0.0004;
% beta1 = 0.0015;

% eta = 1.2;
% beta = 0.95;
% beta = 0.9;
% kappa = 0.012;
% kappa = 0.02;

setlmis([]) 
Q = lmivar(1,[4 1]); 
R = lmivar(2,[1 4]);
 
lmiterm([1 1 1 Q], eye(4), SP);
lmiterm([1 1 1 0], -eye(4));

lmiterm([-2 1 1 Q], eta*eye(4), SP);
lmiterm([-2 1 1 0], -eye(4));

lmiterm([-3 1 1 Q],1,1);

lmiterm([-4 1 1 Q],1,(beta - 2*eta*kappa)*eye(4));
lmiterm([-4 2 1 Q],Ak,1);
lmiterm([-4 2 1 R],Bk,1);
lmiterm([-4 2 2 Q],1,1);

lmiterm([-5 1 1 Q],1,1);
lmiterm([-5 2 1 R],1,1);
lmiterm([-5 2 2 0],1/beta1);

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);
% assert(tmin < 0)

Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);
F_hat = R*pinv(Q);

M = As + Bs*F_hat;
assert(all(eig(M)<0))

end