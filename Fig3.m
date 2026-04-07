clear all

png_idx = 1;

a2_values = [0.25 0.5 1 2 4 8]; %2.^linspace(-2,4,7);
h2_values = [0.25 0.5 1 2 4 8]; %2.^linspace(-2,4,7);

coex_linsat = zeros(length(a2_values),length(h2_values));
coex_linlinlin = zeros(length(a2_values),length(h2_values));
coex_linsatlin = zeros(length(a2_values),length(h2_values));
coex_linlinsat = zeros(length(a2_values),length(h2_values));
coex_linsatsat = zeros(length(a2_values),length(h2_values));


for a2_idx = 1:length(a2_values)
    for h2_idx = 1:length(h2_values)
        
        a2 = a2_values(a2_idx);
        h2 = h2_values(h2_idx);

        % load matrices
        aPabs = 0;
        hPabs = 0;
        
        aPlin = 1;
        hPlin = 0;
        
        aPsat = 8;
        hPsat = 3;
        
        dP = 0.25;
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
        
        try
            M1 = load(filename1).coexistence;
            M2 = load(filename2).coexistence;
            M3 = load(filename3).coexistence;
            M4 = load(filename4).coexistence;
            M5 = load(filename5).coexistence;
            
            coex_linsat(a2_idx,h2_idx) = sum(M1(:)==3)/sum(M1(:)>-1);
            coex_linlinlin(a2_idx,h2_idx) = sum(M2(:)==3)/sum(M2(:)>-1);
            coex_linsatlin(a2_idx,h2_idx) = sum(M3(:)==3)/sum(M3(:)>-1);
            coex_linlinsat(a2_idx,h2_idx) = sum(M4(:)==3)/sum(M4(:)>-1);
            coex_linsatsat(a2_idx,h2_idx) = sum(M5(:)==3)/sum(M5(:)>-1);
        catch
            coex_linsat(a2_idx,h2_idx) = nan;
            coex_linlinlin(a2_idx,h2_idx) = nan;
            coex_linsatlin(a2_idx,h2_idx) = nan;
            coex_linlinsat(a2_idx,h2_idx) = nan;
            coex_linsatsat(a2_idx,h2_idx) = nan;
        end
    end

end
coex_linsat = coex_linsat';
coex_linlinlin = coex_linlinlin';
coex_linsatlin = coex_linsatlin';
coex_linlinsat = coex_linlinsat';
coex_linsatsat = coex_linsatsat';



%% Plot coex matrices

set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

figure()

t = tiledlayout(1,5,'Padding','tight','TileSpacing','tight');

% data and titles
matrices = {coex_linsat, coex_linlinlin, coex_linsatlin, coex_linlinsat, coex_linsatsat};
titles   = {'Model 1','Model 2a','Model 3a','Model 2b','Model 3b'};


% for the maximum in the colorbar
% compute absolute maximum across all matrices
all_values = cell2mat(cellfun(@(M) M(:), matrices, 'UniformOutput', false));
cmax = max(max(all_values));
cmin = 0;  % or min(all_values) if you want dynamic lower bound


%xticks_vals = [0.25 0.5 1 2 4 8 16];
xticks_vals = a2_values; %[0.25 0.5 1 2 4];
yticks_vals = h2_values; %[0.25 0.5 1 2 4 8];


colormap(gcf, turbo)   % set colormap for the figure


for i = 1:5
    ax(i) = nexttile;
    % 
    % % Use surf for all so we can log-scale axes
    % surf(a2_values, h2_values, matrices{i}', 'EdgeColor','none')
    % view(2)
    [X,Y] = meshgrid(a2_values, h2_values);
    s = scatter(X(:), Y(:), 1600, matrices{i}(:), 's', 'filled');
    % force shared color limits
    clim(ax(i), [0 cmax])

    set(gca, 'XScale','log','YScale','log')
    xticks(xticks_vals)
    yticks(yticks_vals)
    xticklabels(arrayfun(@num2str, xticks_vals, 'UniformOutput', false))
    yticklabels(arrayfun(@num2str, yticks_vals, 'UniformOutput', false))
    
    title(titles{i}, 'Interpreter','latex', 'FontSize',16)
    xlabel('$a_2$', 'Interpreter','latex', 'FontSize', 16)
    if i == 1
        ylabel('$h_2$', 'Interpreter','latex', 'FontSize', 16)
    end

    xlim([min(a2_values)/2,max(a2_values)*2]);
    ylim([min(h2_values)/2,max(h2_values)*2]);
    axis square

end

a=colorbar(ax(3),'southoutside','TickLabelInterpreter','latex');
a.Label.String = 'Relative coexistence area';
a.Label.Interpreter = 'latex'; 
a.Label.FontSize = 16; 

for i = 1:5
    ax(i).Title.Units = 'normalized';
    ax(i).Title.Position(2) = 1.1;  % move up a bit
end