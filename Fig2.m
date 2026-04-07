
png_idx = 1;

% for a2 = 2.^linspace(-2,4,7)
%     for h2 = 2.^linspace(-2,4,7)


% load matrices
a2 = 2;
h2 = 8;

aPabs = 0;
hPabs = 0;

aPlin = 1;
hPlin = 0;

aPsat = 2;
hPsat = 2;

dP = 0.1;
resolution = 50;

d1_values = get_grid(1,0,resolution);
d2_values = get_grid(a2,h2,resolution);

filename1 = sprintf('./matrices/model_linsatlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                num2str(a2), num2str(h2), num2str(aPabs), num2str(hPabs), num2str(dP)); % lin sat
params_model{1} = [1, a2, h2, aPabs, hPabs, dP];
model{1} = 'linsat';

filename2 = sprintf('./matrices/model_linlinlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                num2str(a2), num2str(h2), num2str(aPlin), num2str(hPlin), num2str(dP)); % lin lin lin
params_model{2} = [1, a2, h2, aPlin, hPlin, dP];
model{2} = 'linlinlin';

filename3 = sprintf('./matrices/model_linsatlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',... 
                num2str(a2), num2str(h2), num2str(aPlin), num2str(hPlin), num2str(dP)); % lin sat lin
params_model{3} = [1, a2, h2, aPlin, hPlin, dP];
model{3} = 'linsatlin';

filename4 = sprintf('./matrices/model_linlinsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                num2str(a2), num2str(h2), num2str(aPsat), num2str(hPsat), num2str(dP)); % lin lin sat
params_model{4} = [1, a2, h2, aPsat, hPsat, dP];
model{4} = 'linlinsat';

filename5 = sprintf('./matrices/model_linsatsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',... 
                num2str(a2), num2str(h2), num2str(aPsat), num2str(hPsat), num2str(dP)); % lin sat sat
params_model{5} = [1, a2, h2, aPsat, hPsat, dP];
model{5} = 'linsatsat';

M1 = load(filename1).coexistence;
M2 = load(filename2).coexistence;
M3 = load(filename3).coexistence;
M4 = load(filename4).coexistence;
M5 = load(filename5).coexistence;

figure()

t = tiledlayout(1,5,'Padding','compact','TileSpacing','compact')

% discrete colormap
cmap = [ ...
    0 0 0;                 % 0
    0 0.4470 0.7410;       % 1
    0.8500 0.3250 0.0980;  % 2
    0.4940 0.1840 0.5560   % 3
];

[D1, D2] = meshgrid(d1_values, d2_values);

matrices = {M1, M2, M3, M4, M5};   % <-- 5 invasion matrices
titlestr = {'Model 1','Model 2a','Model 3a','Model 2b','Model 3b'};

for i = 1:5

    ax(i) = nexttile;
    hold on

    M = matrices{i};

    for k = 0:3
        mask = M' == k;
        scatter(D1(mask), D2(mask), 48, cmap(k+1,:), 'filled', 'o');
    end

    colormap(cmap)
    clim([0 3])

    axis square
    BoundaryConditions(gcf, params_model{i}, model{i})
    title(titlestr{i},'Interpreter','latex','FontSize',16)

    if i == 1
        ylabel('$d_2$','Interpreter','latex', 'FontSize', 16)
    end
    xlabel('$d_1$','Interpreter','latex', 'FontSize', 16)
end

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

lgd = legend(ax(3), h, ...
    {'mutually protected', ...
     '$C_1$ can invade', ...
     '$C_2$ can invade', ...
     'mutually invasible'}, ...
    'Location','southoutside', ...
    'Orientation','horizontal', ...
    'box','off',...
    'FontSize',16, 'Interpreter','latex');


title(t,sprintf('$a_2 = %s$, $h_2 = %s$',num2str(a2),num2str(h2)),'Interpreter','latex','FontSize',30)
exportgraphics(gcf,sprintf('./figs/pngs/coex_areas_%d.png',png_idx))

png_idx = png_idx + 1;


%close(gcf)

%     end
% end
