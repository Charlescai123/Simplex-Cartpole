function [F_hat] = patch_cvx(Ac, Bc, Ak, Bk)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % e_val = As*s_star + Bs*u_star
    % v = -pinv(Bk)*Ak*s_star
    n = 4;
    alpha = 0.96;

    D = [1/0.4,     0        0,       0;
             0,    1/4.5,       0,      0;
             0,      0,   1/0.4,      0;
             0,      0,       0,   1/4.5];

    cvx_begin sdp
    
        variable Q(n,n) symmetric;
        variable R(1,4);
    
        % minimize -log_det(Q)
    %     D * Q * D' - eye(4) < 0
        [alpha*Q,    Q*Ak' + R'*Bk';
          Ak*Q + Bk*R,    Q] >= 0;
    %     [Q2,      R2';
    %      R2,      1/beta] > 0
        % [1   sd';
        %  sd   Q] >= 0

        % [1   e';
        %  e   Q] >= 0
    
        D * Q * D' - eye(4) <= 0;
    
    cvx_end
    
    P = pinv(Q);
    F_hat = R*P;
    M = Ac + Bc*F_hat;
    eig(M)
    assert(all(eig(M)<0))


end