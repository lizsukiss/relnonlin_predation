% checking if the simulation results for lin_lin_P are the same as the
% analytical ones

% example parameters
a1 = 1;
a2 = 4;
h2 = 8;
aP = 1;
dP = 0.25;

% analytical

% % grids
d1 = linspace(0,a1,32);
d1 = d1(2:end-1);
d2 = linspace(0,a2/(1+a2*h2),32);
d2 = d2(2:end-1);

% % preallocate
% coexistence_lin_lin_predator = zeros(30, 30);
% stability_lin_lin_predator   = zeros(30, 30);

P_star  = zeros(30, 30);
R_star  = zeros(30, 30);
C1_star = zeros(30, 30);
C2_star = zeros(30, 30);

% symbolic setup
syms R C1 C2 P real positive

for i = 1:30
    for j = 1:30

        % linearized attack rate
        aLin = (1 - d2(j) * h2) * a2;

        % equilibrium
        R_star(i,j)  = (d1(i) - d2(j)) / (a1 - aLin);
        C1_star(i,j) = (1 - R_star(i,j) - dP/aP * aLin) / (a1 - aLin);
        C2_star(i,j) = dP/aP - C1_star(i,j);
        P_star(i,j)  = (d1(i) * aLin - d2(j) * a1) / (a1 - aLin);

        % % system
        % Rdot  = ((1 - R) - a1*C1 - aLin*C2) * R;
        % C1dot = (a1*R - d1(i) - aP*P/(1 + aP*hP*(C1+C2))) * C1;
        % C2dot = (aLin*R - d2(j) - aP*P/(1 + aP*hP*(C1+C2))) * C2;
        % Pdot  = (aP*(C1+C2)/(1 + aP*hP*(C1+C2)) - dP) * P;

        % eqns = [Rdot; C1dot; C2dot; Pdot];
        % vars = [R, C1, C2, P];

        % % Jacobian
        % J = jacobian(eqns, vars);

        % % evaluate Jacobian at equilibrium
        % Jnum = double(subs(J, vars, ...
        % [R_star(i,j), C1_star(i,j), C2_star(i,j), P_star(i,j)]));

        % % eigenvalues
        % ev = eig(Jnum);

        % if all(real(ev) < 0)
        %     stability_lin_lin_predator(i,j) = 1; % stable fixed point
        % end
    end
end

    % positive equilibrium
    matrix_analytical = ...
        (C1_star > 0) & (C2_star > 0) & (P_star > 0) & (R_star > 0);

    matrix_analytical = double(matrix_analytical);

    % % coexistence = positive AND stable
    % coexistence_lin_lin_predator = ...
    %     coexistence_positive & (stability_lin_lin_predator == 1);
    % 
    % coexistence_lin_lin_predator = double(coexistence_lin_lin_predator);

% numerical

[file, location] = uigetfile('*.mat', 'Select matrix');
fullpath = fullfile(location,file)
loaded_file = load(fullpath);        % load .mat file
matrix_numerical = loaded_file.coexistence;      % loading the matrix

% plot both cases next to each other

% % analytical
subplot(1, 2, 1)
plot_matrix(matrix_analytical)

% % numerical
subplot(1, 2, 2)
plot_matrix(matrix_numerical)

% % title
title(sprintf('$a_1=%d$, $a_2 = %d$, $h_2 = %d$', a1, a2, h2),Interpreter="latex");


% % one shared colorbar
cb = colorbar('Position',[0.92 0.15 0.02 0.7]);
cb.Ticks = [1/3 1 5/3];
cb.Limits = [0 2];
cb.TickLabels = {
    'no coexistence'
    'static eq.'
    'dynamic eq.'
};
cb.Ruler.TickLabelRotation=90;


function plot_matrix(M)
    
    imagesc(M')
    axis image
    set(gca,'YDir','normal')
    
    % discrete colormap for 0/1/2
    cmap = [ ...
        1 1 1;    % 0 → white
        0.3 0.7 0.3;  % 1 → green
        0.8 0.2 0.2   % 2 → red
    ];
    
    colormap(cmap)
    caxis([0 2])
    
    % make NaNs gray
    set(gca,'Color',[0.7 0.7 0.7])   % background color
    set(findobj(gca,'Type','Image'),'AlphaData',~isnan(M'))

end