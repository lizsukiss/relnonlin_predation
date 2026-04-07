% Large figure with different parameter combinations

% Parameters for a2 and h2
a2_values = [2, 4, 4, 8];
h2_values = [8, 0.25, 1, 0.5];

% Other parameters

aPabs = 0; % when P absent
hPabs = 0;

aPlin = 2.25; % when P linear
hPlin = 0;

aPsat = 2.5; % when P saturating
hPsat = 1;

dP = 0.1; % mortality rate of P for all cases

resolution = 30;

% set up the whole figure
figure()
t = tiledlayout(length(a2_values),5,'Padding','compact','TileSpacing','compact');
    
% discrete colormap
cmap = [ ...
    0 0 0;                 % 0
    0 0.4470 0.7410;       % 1
    0.8500 0.3250 0.0980;  % 2
    0.4940 0.1840 0.5560   % 3
];


for idx = 1:length(a2_values)
    % load matrices
    a2 = a2_values(idx);
    h2 = h2_values(idx);

    % load the data
    model1 = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPabs,hPabs,dP,'rnl','reversed'));
    model2a = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPlin,hPlin,dP,'linearized','reversed'));
    model2b = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPsat,hPsat,dP,'linearized','reversed'));
    model3a = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPlin,hPlin,dP,'rnl','reversed'));
    model3b = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPsat,hPsat,dP,'rnl','reversed'));
    
    d1_values = get_grid(1,0,resolution);
    d2_values = get_grid(a2,h2,resolution);
    
    %filename1 = sprintf('./matrices/model_linsatlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
    %                num2str(a2), num2str(h2), num2str(aPabs), num2str(hPabs), num2str(dP)); % lin sat
    params_model{1} = [1, a2, h2, aPabs, hPabs, dP];
    model{1} = 'linsat';
    
    %filename2 = sprintf('./matrices/model_linlinlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
    %                num2str(a2), num2str(h2), num2str(aPlin), num2str(hPlin), num2str(dP)); % lin lin lin
    params_model{2} = [1, a2, h2, aPlin, hPlin, dP];
    model{2} = 'linlinlin';
    
    %filename3 = sprintf('./matrices/model_linsatlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',... 
    %                num2str(a2), num2str(h2), num2str(aPlin), num2str(hPlin), num2str(dP)); % lin sat lin
    params_model{3} = [1, a2, h2, aPlin, hPlin, dP];
    model{3} = 'linsatlin';
    
    %filename4 = sprintf('./matrices/model_linlinsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
    %                num2str(a2), num2str(h2), num2str(aPsat), num2str(hPsat), num2str(dP)); % lin lin sat
    params_model{4} = [1, a2, h2, aPsat, hPsat, dP];
    model{4} = 'linlinsat';
    
    %filename5 = sprintf('./matrices/model_linsatsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',... 
    %                num2str(a2), num2str(h2), num2str(aPsat), num2str(hPsat), num2str(dP)); % lin sat sat
    params_model{5} = [1, a2, h2, aPsat, hPsat, dP];
    model{5} = 'linsatsat';
    
    M1 = model1.C1_invasion + model1.C2_invasion;
    M2a = model2a.C1_invasion + model2a.C2_invasion + model2a.P_invasion_in_main;
    M2a_woP = model2a.C1_invasion + model2a.C2_invasion;
    M2b = model2b.C1_invasion + model2b.C2_invasion + model2b.P_invasion_in_main;
    M2b_woP = model2b.C1_invasion + model2b.C2_invasion;
    M3a = model3a.C1_invasion + model3a.C2_invasion + model3a.P_invasion_in_main;
    M3a_woP = model3a.C1_invasion + model3a.C2_invasion;
    M3b = model3b.C1_invasion + model3b.C2_invasion + model3b.P_invasion_in_main;
    M3b_woP = model3b.C1_invasion + model3b.C2_invasion;
    
    [D1, D2] = meshgrid(d1_values, d2_values);
    
    matrices = {M1, M2a, M3a, M2b, M3b};   % <-- 5 invasion matrices
    titlestr = {'Model 1','Model 2a','Model 3a','Model 2b','Model 3b'};
    
    for i = 1:5
    
        ax(i) = nexttile;
        hold on
    
        M = matrices{i};
        
        for k = [0,1,2,3,3.5]
            if k==3.5
                mask = M' >= k; % P can invade the 'dominant' subsystem (C1/C2/C1+C2)
            else
                mask = M' == k;
            end
            scatter(D1(mask), D2(mask), 48, cmap(ceil(k+1),:), 'filled', 'o');
        end
    
        colormap(cmap)
        clim([0 3])
    
        axis square
        BoundaryConditions(gcf, params_model{i}, model{i})
        
        if idx == 1
            title(titlestr{i},'Interpreter','latex','FontSize',16)
        end

        if i == 1
            ylabel('$d_2$','Interpreter','latex', 'FontSize', 16)
        end
        xlabel('$d_1$','Interpreter','latex', 'FontSize', 16)


        % add the numerical boundaries of P persisting in the system
        % RC2P
        mod = params_model{i};
        base_folder_C2 = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(2), mod(3), mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C2, 'normal/d2_idx_%d_of_%d.mat', resolution);
        P_persistence_C2 = nan(length(d2_values),1);
        P_persistence_C2(dens_stats.Pmin > eps(0)) = 0;
        %P_persistence_C2(dens_stats.Pmean > 10e-30) = 0;
        

        % RC1P
        mod = params_model{i};
        base_folder_C1 = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(1), 0, mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C1, 'normal/d1_idx_%d_of_%d.mat', resolution);
        P_persistence_C1 = nan(length(d2_values),1);
        P_persistence_C1(dens_stats.Pmin > eps(0)) = 0;
        %P_persistence_C1(dens_stats.Pmean > 10e-30) = 0;

        
        % d2 axes
        plot(P_persistence_C2, d2_values,"Linewidth",4,'Color','y')

        % d1 axes
        plot(d1_values, P_persistence_C1,"Linewidth",4,'Color','y')

    end
    
    if idx == length(a2_values)

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
    end 

    %title(t,sprintf('$a_2 = %s$, $h_2 = %s$',num2str(a2),num2str(h2)),'Interpreter','latex','FontSize',30)
    %exportgraphics(gcf,sprintf('./figs/pngs/coex_areas_%d.png',png_idx))
    
    %png_idx = png_idx + 1;
    
end

% same without considering the invasion propensity of P
figure()
t = tiledlayout(length(a2_values),5,'Padding','compact','TileSpacing','compact');
    
for idx = 1:length(a2_values)
    % load matrices
    a2 = a2_values(idx);
    h2 = h2_values(idx);

    % load the data
    model1 = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPabs,hPabs,dP,'rnl','reversed'));
    model2a = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPlin,hPlin,dP,'linearized','reversed'));
    model2b = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPsat,hPsat,dP,'linearized','reversed'));
    model3a = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPlin,hPlin,dP,'rnl','reversed'));
    model3b = load(sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        1,a2,h2,aPsat,hPsat,dP,'rnl','reversed'));
    
    d1_values = get_grid(1,0,resolution);
    d2_values = get_grid(a2,h2,resolution);
    
    params_model{1} = [1, a2, h2, aPabs, hPabs, dP];
    model{1} = 'linsat';
    
    params_model{2} = [1, a2, h2, aPlin, hPlin, dP];
    model{2} = 'linlinlin';
    
    params_model{3} = [1, a2, h2, aPlin, hPlin, dP];
    model{3} = 'linsatlin';
    
    params_model{4} = [1, a2, h2, aPsat, hPsat, dP];
    model{4} = 'linlinsat';
    
    params_model{5} = [1, a2, h2, aPsat, hPsat, dP];
    model{5} = 'linsatsat';
    
    M1 = model1.C1_invasion + model1.C2_invasion;
    M2a = model2a.C1_invasion + model2a.C2_invasion + model2a.P_invasion_in_main;
    M2a_woP = model2a.C1_invasion + model2a.C2_invasion;
    M2b = model2b.C1_invasion + model2b.C2_invasion + model2b.P_invasion_in_main;
    M2b_woP = model2b.C1_invasion + model2b.C2_invasion;
    M3a = model3a.C1_invasion + model3a.C2_invasion + model3a.P_invasion_in_main;
    M3a_woP = model3a.C1_invasion + model3a.C2_invasion;
    M3b = model3b.C1_invasion + model3b.C2_invasion + model3b.P_invasion_in_main;
    M3b_woP = model3b.C1_invasion + model3b.C2_invasion;
            
    % same without considering the invasion propensity of P
    matrices = {M1, M2a_woP, M3a_woP, M2b_woP, M3b_woP};   % <-- 5 invasion matrices
    titlestr = {'Model 1','Model 2a','Model 3a','Model 2b','Model 3b'};
    
    for i = 1:5
    
        ax(i) = nexttile;
        hold on
    
        M = matrices{i};
        
        for k = [0,1,2,3,3.5]
            if k==3.5
                mask = M' >= k; % P can invade the 'dominant' subsystem (C1/C2/C1+C2)
            else
                mask = M' == k;
            end
            scatter(D1(mask), D2(mask), 48, cmap(ceil(k+1),:), 'filled', 'o');
        end
    
        colormap(cmap)
        clim([0 3])
    
        axis square
        BoundaryConditions(gcf, params_model{i}, model{i})
        
        if idx == 1
            title(titlestr{i},'Interpreter','latex','FontSize',16)
        end

        if i == 1
            ylabel('$d_2$','Interpreter','latex', 'FontSize', 16)
        end
        xlabel('$d_1$','Interpreter','latex', 'FontSize', 16)


        % add the numerical boundaries of P persisting in the system
        % RC2P
        mod = params_model{i};
        base_folder_C2 = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(2), mod(3), mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C2, 'normal/d2_idx_%d_of_%d.mat', resolution);
        P_persistence_C2 = nan(length(d2_values),1);
        P_persistence_C2(dens_stats.Pmin > eps(0)) = 0;
        %P_persistence_C2(dens_stats.Pmean > 10e-30) = 0;
        

        % RC1P
        mod = params_model{i};
        base_folder_C1 = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(1), 0, mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C1, 'normal/d1_idx_%d_of_%d.mat', resolution);
        P_persistence_C1 = nan(length(d2_values),1);
        P_persistence_C1(dens_stats.Pmin > eps(0)) = 0;
        %P_persistence_C1(dens_stats.Pmean > 10e-30) = 0;

        
        % d2 axes
        plot(P_persistence_C2, d2_values,"Linewidth",4,'Color','y')

        % d1 axes
        plot(d1_values, P_persistence_C1,"Linewidth",4,'Color','y')

    end

    if idx == length(a2_values)

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
    end    
        

    %title(t,sprintf('$a_2 = %s$, $h_2 = %s$',num2str(a2),num2str(h2)),'Interpreter','latex','FontSize',30)
    %exportgraphics(gcf,sprintf('./figs/pngs/coex_areas_%d.png',png_idx))
    
    %png_idx = png_idx + 1;
    
end


%{

for idx = 1:length(a2_values)
    % load matrices
    a2 = a2_values(idx);
    h2 = h2_values(idx);

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
    
    [D1, D2] = meshgrid(d1_values, d2_values);
    
    matrices = {M1, M2, M3, M4, M5};   % <-- 5 invasion matrices
    titlestr = {'Model 1','Model 2a','Model 3a','Model 2b','Model 3b'};
    
    for i = 1:5
    
        ax(i) = nexttile;
        hold on
    
        M = matrices{i};
        max(max(M))
        for k = [0,1,2,3,3.5]
            mask = M' == k;
            scatter(D1(mask), D2(mask), 48, cmap(ceil(k+1),:), 'filled', 'o');
        end
    
        colormap(cmap)
        clim([0 3])
    
        axis square
        BoundaryConditions(gcf, params_model{i}, model{i})
        
        if idx == 1
            title(titlestr{i},'Interpreter','latex','FontSize',16)
        end

        if i == 1
            ylabel('$d_2$','Interpreter','latex', 'FontSize', 16)
        end
        xlabel('$d_1$','Interpreter','latex', 'FontSize', 16)


        % add the numerical boundaries of P persisting in the system
        % RC2P
        mod = params_model{i};
        base_folder_C2 = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(2), mod(3), mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C2, 'normal/d2_idx_%d_of_%d.mat', resolution);
        P_persistence_C2 = nan(length(d2_values),1);
        P_persistence_C2(dens_stats.Pmax > 10e-30) = 0;
        %P_persistence_C2(dens_stats.Pmean > 10e-30) = 0;
        

        % RC1P
        mod = params_model{i};
        base_folder_C1 = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            mod(1), 0, mod(4), mod(5), mod(6));
        dens_stats = load_bifurcation(base_folder_C1, 'normal/d1_idx_%d_of_%d.mat', resolution);
        P_persistence_C1 = nan(length(d2_values),1);
        P_persistence_C1(dens_stats.Pmax > 10e-30) = 0;
        %P_persistence_C1(dens_stats.Pmean > 10e-30) = 0;

        
        % d2 axes
        plot(P_persistence_C2, d2_values,"Linewidth",4,'Color','y')

        % d1 axes
        plot(d1_values, P_persistence_C1,"Linewidth",4,'Color','y')

    end
    
    if idx == length(a2_values)

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
    end    
        
    %title(t,sprintf('$a_2 = %s$, $h_2 = %s$',num2str(a2),num2str(h2)),'Interpreter','latex','FontSize',30)
    %exportgraphics(gcf,sprintf('./figs/pngs/coex_areas_%d.png',png_idx))
    
    %png_idx = png_idx + 1;
    
end

%}