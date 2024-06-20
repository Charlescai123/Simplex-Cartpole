% Matlab Code for Inverted Pendulum Simulation
clear, clc

syms mp mc l theta theta_dot F g
syms x x_dot theta theta_dot

% Equations of Motion
theta_ddot = (g*sin(theta)+cos(theta)*((-F-mp*l*theta_dot^2*sin(theta))/(mc+mp)))/(l*(4/3-(mp*cos(theta)^2)/(mc+mp)));
x_ddot = (F+mp*l*(sin(theta)*theta_dot^2-cos(theta)*theta_ddot))/(mc+mp);

% dd_theta_theta_dot = jacobian(dd_theta, theta);
% dd_x_theta = jacobian(dd_x, theta);

% jacobian(x_ddot, F)

% (F + l*mp*theta_dot^2*sin(theta) - (cos(theta)*(g*sin(theta) - (cos(theta)*(l*mp*sin(theta)*theta_dot^2 + F))/(mc + mp))*(3*mc + 3*mp))/(l*(4*mc + 4*mp - 3*mp*cos(theta)^2)))/(mc + mp)

% A(s) and B(s)
A = sym(zeros(4,4));
B = sym(zeros(4,1));

long_term = 4/3*(mc+mp) - mp*cos(theta)^2; 

A(1,2) = 1;
A(3,4) = 1;
A(2,3) = -mp*g*sin(theta)*cos(theta)/(theta*long_term);
A(2,4) = 4/3*mp*l*sin(theta)*theta_dot/long_term;
A(4,3) = g*sin(theta)*(mc+mp)/(l*theta*long_term);
A(4,4) = -mp*sin(theta)*cos(theta)*theta_dot/long_term;

B(2) = 4/3/long_term;
B(4) = -cos(theta)/(l*long_term);

% Safety Set
x_set = [-0.9, 0.9];
x_dot_set = [-3, 3];
theta_set = [-0.8, 0.8];
theta_dot_set = [-4.5, 4.5];

% D = [1/0.9,   0,        0,        0;
%        0,     0,       1/0.8,     0;
%        0,     1/100,    0,         0;
%        0,     0,       0,     1/100];

D = [1/0.9,      0        0,       0;
         0,    1/3,       0,       0;
         0,      0,   1/0.8,       0;
         0,      0,       0,   1/4.5];

% Value Assignment
mc_v = 0.94;
mp_v = 0.23;
l_v = 0.64/2;
g_v = 9.8;
T = 1/100;

% long_term_v = subs(long_term, [mc, mp, theta], [mc_v, mp_v, theta_v]);

% Linearized around Eq
[theta_v, theta_dot_v] = deal(-0.45499458, -5.82901209);

% [theta_v, theta_dot_v] = deal(-0.41187448, -2.6532297);
% [theta_v, theta_dot_v] = deal(-0.25915376, -0.73479655);

% Continuous Form
As = subs(A, [mc, mp, l, g, theta, theta_dot], [mc_v, mp_v, l_v, g_v, theta_v, theta_dot_v]);
Bs = subs(B, [mc, mp, l, g, theta, theta_dot], [mc_v, mp_v, l_v, g_v, theta_v, theta_dot_v]);
As = double(As);
Bs = double(Bs);

% Discretized Form
Ak = As*T + eye(4);
Bk = Bs*T;

%%%%%%%%%%%%%%%%%%%%%%%%%% Validatioin %%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [-0.35045414  1.2595911  -0.06073488 -0.34843698]';
u0 = -0.9753906194720958;
x1 = Ak*x0 + Bk*u0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 4;
alpha = 0.985;
beta = 0.002;
invV = 1/beta;
c = 1/15;

A_eq_c = double(subs(A, [mc, mp, l, g, theta, theta_dot], [mc_v, mp_v, l_v, g_v, 0.001, 0.001]));
B_eq_c = double(subs(B, [mc, mp, l, g, theta, theta_dot], [mc_v, mp_v, l_v, g_v, 0.001, 0.001]));

A_eq_k = A_eq_c*T + eye(4);
B_eq_k = B_eq_c*T;

cvx_begin sdp

    variable Q(n,n) symmetric;
    variable R(1,4);

    minimize -log_det(Q)
    D * Q * D' - eye(4) < 0
    [alpha*Q,      Q*A_eq_k' + R'*B_eq_k';
      A_eq_k*Q + B_eq_k*R,  Q] > 0
    [Q,      R';
     R,      1/beta] > 0

cvx_end

P = inv(Q);
F = R*P;
A_bar = A_eq_k + B_eq_k*F;

C = eig((A_eq_k+B_eq_k*F)'*P*(A_eq_k+B_eq_k*F) - alpha*P);

% Sub Constraint
pP = zeros(2, 2);
vP = zeros(2, 2);

% For Position
pP(1, 1) = P(1, 1);
pP(2, 2) = P(3, 3);
pP(1, 2) = P(1, 3);
pP(2, 1) = P(1, 3);

% For Velocity
vP(1, 1) = P(2, 2);
vP(2, 2) = P(4, 4);
vP(1, 2) = P(2, 4);
vP(2, 1) = P(2, 4);

% Plot Safety Envelope
plot_envelope(x_set, theta_set, pP)

%%%%%%%%%%%%%%%%%%%%%%% Calc K %%%%%%%%%%%%%%%%%%%%%%%
syms s
poles = [-1 -2 -1-1i -1+1i];
ctl_system = poly(poles);
phi_bar = polyvalm(ctl_system,As);
CMs = [Bs As*Bs As^2*Bs As^3*Bs];
K = -[0,0,0,1]*inv(CMs)*phi_bar;
eig(As+Bs*K)

AA = (As+Bs*K)*T + eye(4)

% s_star =  [0.14998542  4.84414696 -0.25146283 -5.87739881]';
s_star = [0.23434349, 0, -0.22644896, 0]';
u_star = -pinv(Bs)*As*s_star
% u_star = -Bk'*Ak*s_star/(Bk'*Bk);

e_val = As*s_star + Bs*u_star
v = -pinv(Bk)*Ak*s_star

alpha2 = 0.96

D2 = [1/0.4,      0        0,       0;
         0,    1/5,       0,       0;
         0,      0,   1/0.4,       0;
         0,      0,       0,   1/5];

cvx_begin sdp

    variable Q2(n,n) symmetric;
    variable R2(1,4);

    minimize -log_det(Q2)
%     D * Q * D' - eye(4) < 0
    [alpha2*Q2,      Q2*Ak' + R2'*Bk';
      Ak*Q2 + Bk*R2,  Q2] > 0
%     [Q2,      R2';
%      R2,      1/beta] > 0
    [1   s_star';
     s_star Q2] > 0

    D2 * Q2 * D2' - eye(4) < 0

cvx_end

P2 = pinv(Q2);
K = R2*P2
M = As + Bs*K;
eig(M)