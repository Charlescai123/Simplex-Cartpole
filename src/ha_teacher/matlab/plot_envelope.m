function plot_envelope(x_set, theta_set, p_mat)
    assert(all(size(p_mat) == [2, 2]))

    [eig_vector, ~] = eig(p_mat);
    eig_value = eig(p_mat);
    
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
end