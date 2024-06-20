clear, clc

x_set = [-0.9, 0.9];
theta_set = [-0.8, 0.8];

A = [1,    0.03333333,    0,            0;
     0,    1,             -0.05649123,  0;
     0,    0,             1,            0.03333333;
     0,    0,             0.89802632,   1        ];

B = [0; 0.03341688; 0; -0.0783208];

D = [1/0.9,   0,        0,        0;
       0,     0,       1/0.8,     0;
       0,     1/100,    0,         0;
       0,     0,       0,     1/100];


n = 4;
alpha = 0.849;
beta = 0.002;
invV = 1/beta;
c = 1/15;

cvx_begin sdp

    variable Q(n,n) symmetric;
    variable R(1,4);

    minimize -log_det(Q)
    D * Q * D' - eye(4) < 0
    [alpha*Q,      Q*A' + R'*B';
      A*Q + B*R,  Q] > 0
    [Q,      R';
     R,      1/beta] > 0

cvx_end

P = inv(Q)
F = R*P
A + B*F

C = eig((A+B*F)'*P*(A+B*F) - alpha*P)


% syms mp mc l theta theta_dot F g
% syms x x_dot theta theta_dot
% 
% theta_ddot = (g*sin(theta)+cos(theta)*((-F-mp*l*theta_dot^2*sin(theta))/(mc+mp)))/(l*(4/3-(mp*cos(theta)^2)/(mc+mp)));
% 
% x_ddot = (F+mp*l*(sin(theta)*theta_dot^2)-cos(theta)*theta_ddot)/(mc+mp);
% 
% % dd_theta_theta_dot = jacobian(dd_theta, theta);
% % dd_x_theta = jacobian(dd_x, theta);
% 
% jacobian(x_ddot, F)


pP = zeros(2, 2);
vP = zeros(2, 2);

% For position
pP(1, 1) = P(1, 1);
pP(2, 2) = P(3, 3);
pP(1, 2) = P(1, 3);
pP(2, 1) = P(1, 3);

% For velocity
vP(1, 1) = P(2, 2);
vP(2, 2) = P(4, 4);
vP(1, 2) = P(2, 4);
vP(2, 1) = P(2, 4);

[eig_vector, ~] = eig(pP);
eig_value = eig(pP);

% Define theta vector
theta = linspace(-pi, pi, 1000);
ty1 = cos(theta) / sqrt(eig_value(1));
ty2 = sin(theta) / sqrt(eig_value(2));
ty = [ty1; ty2];
tQ = inv(eig_vector');
tx = tQ * ty;
tx1 = tx(1, :);
tx2 = tx(2, :);

% Plot safety envelope
figure;
plot(tx1, tx2, 'k', 'LineWidth', 2);
hold on;
line([x_set(1), x_set(2)], [theta_set(1), theta_set(1)],'color','k', 'LineWidth', 2);
line([x_set(1), x_set(2)], [theta_set(2), theta_set(2)],'color','k', 'LineWidth', 2);
line([x_set(1), x_set(1)], [theta_set(1), theta_set(2)],'color','k', 'LineWidth', 2);
line([x_set(2), x_set(2)], [theta_set(1), theta_set(2)],'color','k', 'LineWidth', 2);

xlabel('x');
ylabel('theta');
title('Safety Envelope');
grid on;
