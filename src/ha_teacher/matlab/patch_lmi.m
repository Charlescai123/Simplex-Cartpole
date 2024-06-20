function [F_hat, tmin] = patch_lmi(Ac, Bc, Ak, Bk)
%%%%%%%%%%%%%%%%%%%%%%  DOC HELP  %%%%%%%%%%%%%%%%%%%%%%
%% Inputs
%
%   Ac  : A(s) in continuous form  -- 4x4
%   Bc  : B(s) in continuous form  -- 4x1
%   Ak  : A(s) in discrete form    -- 4x4
%   Bk  : B(s) in discrete form    -- 4x1
%
%% Outputs
%   F   : Feedback control gain    -- 1x4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SP = [11.7610, 6.1422, 14.7641, 3.1142;
%       6.1422, 3.8248, 9.3641, 1.9903;
%       14.7641, 9.3641, 25.9514, 5.0703;
%       3.1142, 1.9903, 5.0703, 1.0949];

SP = [13.3425, 6.73778, 16.2166, 3.47318;
      6.73778, 3.94828, 9.69035, 2.09032;
      16.2166, 9.69035, 25.9442, 5.31439;
      3.47318, 2.09032, 5.31439, 1.16344];

beta1 = 0.0004;
% beta1 = 0.0015;

eta = 1.1;
% beta = 0.95;
beta = 0.95;
% kappa = 0.012;
kappa = 0.02;

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
F_hat = R*inv(Q);

M = Ac + Bc*F_hat;
eig(M)
assert(all(eig(M)<0))

end