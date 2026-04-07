%%% translate all simulation results to individual invasion matrices

% quite slow. speed could be much improved by: only opening the R-C1/C2
% files once and checking both for the linear and saturating functional
% response of the potential predator whether it can invade. the other parts
% still need to be handled separately
% + when P is absent, the computations for it are not necessary

% Define varying parameters
a2_values = 8 %2.^linspace(-2,3,21);
h2_values = .5 %2.^linspace(-2,3,21);

% Parameters for all sets
C1_params.a1 = 1;
C1_params.h1 = 0;
resolution = 30;

Pabs_params.aP = 0;
Pabs_params.hP = 0;
Pabs_params.dP = 0.1;

Plin_params.aP = 2.25;
Plin_params.hP = 0;
Plin_params.dP = 0.1;

Psat_params.aP = 2.5;
Psat_params.hP = 1;
Psat_params.dP = 0.1;

parameter_sets = {};
set_idx = 1;
for a2_value = a2_values
    for h2_value = h2_values
        C2_params = struct();
        
        C2_params.a2 = a2_value;
        C2_params.h2 = h2_value;
        
        parameter_sets{set_idx} = C2_params;
        set_idx = set_idx + 1;
    end
end


for C2_params = [parameter_sets{:}]
    
    % matrix translations
    
    filename1 = translate2matrix(C1_params, C2_params, Pabs_params, resolution, 'rnl', 'reversed');
    filename3a = translate2matrix(C1_params, C2_params, Plin_params, resolution, 'rnl', 'reversed'); 
    filename2a = translate2matrix(C1_params, C2_params, Plin_params, resolution, 'linearized', 'reversed'); 
    filename3b = translate2matrix(C1_params, C2_params, Psat_params, resolution, 'rnl', 'reversed');
    filename2b = translate2matrix(C1_params, C2_params, Psat_params, resolution, 'linearized', 'reversed'); 
        
end

%{
%% 
%data = load(ui.) FIX THIS AT SOME POINT TO PLOT AN EXAMPLE OF THE P
%INVASION
figure()
% discrete colormap
cmap = [ ...
    0 0 0;                 % 0
    0 0.4470 0.7410;       % 0.5
    0.8500 0.3250 0.0980;  % 0.6
    0.4940 0.1840 0.5560   % 0.7
];

matrices = {data.P_invasion_in_C1, data.P_invasion_in_C2, data.P_invasion_in_C1C2, data.P_invasion_in_main};
d1_values = get_grid(a1,h1,resolution);
d2_values = get_grid(a2,h2,resolution);
[D1, D2] = meshgrid(d1_values, d2_values);
titles = {'$R-C_1$', '$R-C_2$', '$R-C_1/C_2$',"'dominant' system"};
t = tiledlayout(2,2);

for ii = 1:4
    ax = nexttile;
    hold on
    M = matrices{ii};
    color_ind = 1;
    for k = [0,0.5,0.6,0.7]
        mask = M' == k;
        scatter(D1(mask), D2(mask), 48, cmap(color_ind,:), 'filled');
        color_ind = color_ind + 1;
    end
    
    colormap(cmap)
    clim([0 3])
    xlim([0,a1])
    ylim([0,a2/(1+a2*h2)])
    title(titles{ii},'Interpreter','latex')
    ax.TickLabelInterpreter='latex';
    axis square
end

% Create legend once, attached to layout
lgd = legend({'$P$ cannot invade', ...
              '$P$ can invade $C_1$', ...
              '$P$ can invade $C_2$', ...
              '$P$ can invade $C_1$ and $C_2$'}, ...
              'Interpreter','latex');

lgd.Layout.Tile = 'south';
%{
for ii = 1:4
    ax = subplot(3,2,ii);
    hold on
    M = matrices{ii};
    color_ind = 1;
    for k = [0,0.5,0.6,0.7]
        mask = M' == k;
        scatter(D1(mask), D2(mask), 48, cmap(color_ind,:), 'filled', 'o');
        color_ind = color_ind + 1;
    end
    
    colormap(cmap)
    clim([0 3])
    xlim([0,a1])
    ylim([0,a2/(1+a2*h2)])
    title(titles{ii},'Interpreter','latex')
    ax.TickLabelInterpreter='latex';
    
    axis square
        
end
subplot(3,2,5)
axis off
% annoying legend part
% Create proper legend handles (using plot, not scatter)
h(1) = plot(nan, nan, 'o', ...
    'MarkerFaceColor', cmap(1,:), ...
    'MarkerEdgeColor', cmap(1,:), ...
    'MarkerSize', 12);

h(2) = plot(nan, nan, 'o', ...
    'MarkerFaceColor', cmap(2,:), ...
    'MarkerEdgeColor', cmap(2,:), ...
    'MarkerSize', 12);

h(3) = plot(nan, nan,  'o', ...
    'MarkerFaceColor', cmap(3,:), ...
    'MarkerEdgeColor', cmap(3,:), ...
    'MarkerSize', 12);

h(4) = plot(nan, nan,  'o', ...
    'MarkerFaceColor', cmap(4,:), ...
    'MarkerEdgeColor', cmap(4,:), ...
    'MarkerSize', 12);

lgd = legend(h, ...
    {'$P$ cannot invade', ...
     '$P$ can invade $C_1$', ...
     '$P$ can invade $C_2$', ...
     '$P$ can invade $C_1$ and $C_2$'}, ...
    'Location','southoutside', ...
    'Orientation','horizontal', ...
    'box','off',...
    'FontSize',16, 'Interpreter','latex');
%}
%}