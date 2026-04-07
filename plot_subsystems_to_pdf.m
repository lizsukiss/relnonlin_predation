%%% Plot bifurcation diagrams for both subsystems R-C1-P and R-C2-P in
%%% PDF files

% slow. problems: R-C1-P is opened again and again, instead of plotting
% them multiple times by opening it once


%%% Parameters

% Define varying parameters
a2_values = 8;%2.^linspace(-2,3,21);
h2_values = 0.5;%2.^linspace(-2,3,21);

% Parameters for all sets
a1 = 1;
h1 = 0;
resolution = 30;

parameter_sets = {};
set_idx = 1;
for a2_value = a2_values
    for h2_value = h2_values
      

            % model 1 (the others will be copying this inside the loop)
            mod1 = struct();
            mod1.a1 = 1;
            mod1.h1 = 0;
            mod1.a2 = a2_value;
            mod1.h2 = h2_value;
            mod1.resolution = resolution;
            mod1.aP = 0;
            mod1.hP = 0;
            mod1.dP = 0.10;
            
            parameter_sets{set_idx} = mod1;
            set_idx = set_idx + 1;
    
      
    end
end


% colors
rcolor = [0.4660, 0.6740, 0.1880];
ccolor = [0.8500, 0.3250, 0.0980];
pcolor = [0.4940, 0.1840, 0.5560];

% for each parameter set
for model_index = 1:length(parameter_sets) % mod1 has the model 1 parameters

    mod1 = parameter_sets{model_index}
    pdf_file = sprintf('./pdfs/bifurcations_a2=%s_h2=%s_resolution=%s.pdf',num2str(mod1.a2), num2str(mod1.h2),num2str(mod1.resolution)); % include the parameters
    
    % delete if it already exists (important!)
    if isfile(pdf_file)
        delete(pdf_file)
    end
       
    % additional parameters
    
    % model 2a & 3a (give the saturating response as input for C2 and linear for P)
    moda = mod1;
    moda.aP = 2.25;
    moda.hP = 0;
    moda.dP = 0.1;
    
    % model 2b & 3b (give the saturating response as input for C2 and P)
    modb = mod1;
    modb.aP = 2.5;
    modb.hP = 1;
    modb.dP = 0.1;
    
    d1_values = get_grid(mod1.a1,0,mod1.resolution); % does not change between the models
    d2_values = get_grid(mod1.a2,mod1.h2,mod1.resolution);

    % preface/parameters in the pdf
    fig = figure('Visible','off');  % or 'on' if you want to preview
    axis off
    
    txt = {
    'Parameter set'
    sprintf('$a_1 = %.2f$', mod1.a1)
    sprintf('$h_1 = %.2f$', mod1.h1)
    ''
    sprintf('$a_2 = %.2f$', mod1.a2)
    sprintf('$h_2 = %.2f$', mod1.h2)
    ''
    sprintf('$a_P = %.2f$', modb.aP)
    sprintf('$h_P = %.2f$', modb.hP)
    sprintf('$d_P = %.2f$', modb.dP)
    ''
    sprintf('$\\mathrm{resolution} = %d$', mod1.resolution)
    };

    text(0.1, 0.9, txt, ...
    'Interpreter','latex', ...
    'FontSize',14, ...
    'VerticalAlignment','top')
    
    exportgraphics(gcf, pdf_file)   % first page (no 'Append')
    close(fig)
    
    % Model 1
    mod = mod1;
    
    base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);
    
    rc2 = load_bifurcation(base, 'normal/d2_idx_%d_of_%d.mat', mod.resolution);
    rc2_reversed = load_bifurcation(base, 'reversed/d2_idx_%d_of_%d.mat', mod.resolution);
    
    base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);
    
    rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);
    rc1_reversed = load_bifurcation(base, 'reversed/d1_idx_%d_of_%d.mat', mod.resolution);

    plot_bifurcation(rc2, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow d_2^{\mathrm{max}}$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
    sgtitle("Model 1 - normal",'Interpreter','latex')
    
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)

    plot_bifurcation(rc2_reversed, rc1_reversed, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
    sgtitle("Model 1 - reversed",'Interpreter','latex')

    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    
    % Model 2a - C2 normal
    mod = moda;
    
    % R–C2–P (vary d2)
    base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);
    
    rc2 = load_bifurcation(base, 'linearized/d2_idx_%d_of_%d.mat', mod.resolution);
    rc2_reversed = load_bifurcation(base, 'linearized_reversed/d2_idx_%d_of_%d.mat', mod.resolution);
    
    
    % R–C1–P (vary d1)
    base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);
    
    rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);
    rc1_reversed = load_bifurcation(base, 'reversed/d1_idx_%d_of_%d.mat', mod.resolution);
    
    
    plot_bifurcation(rc2, rc1, d2_values, d1_values, ...
        rcolor, ccolor, pcolor, ...
        "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
        "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
    )
    sgtitle("Model 2a - normal",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)

    plot_bifurcation(rc2_reversed, rc1_reversed, d2_values, d1_values, ...
        rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow 0$")
    sgtitle("Model 2a - reversed",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    % Model 2b - C2 linearized
    mod = modb;
    
    % R–C2–P (vary d2)
    base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);
    
    rc2 = load_bifurcation(base, 'linearized/d2_idx_%d_of_%d.mat', mod.resolution);
    rc2_reversed = load_bifurcation(base, 'linearized_reversed/d2_idx_%d_of_%d.mat', mod.resolution);
    
    
    % R–C1–P (vary d1)
    base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);
    
    rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);
    rc1_reversed = load_bifurcation(base, 'reversed/d1_idx_%d_of_%d.mat', mod.resolution);
    
    
    plot_bifurcation( ...
        rc2, rc1, ...
        d2_values, d1_values, ...
        rcolor, ccolor, pcolor, ...
        "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
        "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
    )
    sgtitle("Model 2b - normal",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)

    plot_bifurcation(rc2_reversed, rc1_reversed, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
    sgtitle("Model 2b - reversed",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    % Model 3a - C2 normal 
    mod = moda;
    
    % R–C2–P (vary d2)
    base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);
    
    rc2 = load_bifurcation(base, 'normal/d2_idx_%d_of_%d.mat', mod.resolution);
    %rc2_reversed = load_bifurcation(base, 'reversed/d2_idx_%d_of_%d.mat', mod.resolution);
    
    
    % R–C1–P (vary d1)
    base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);
    
    rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);
    rc1_reversed = load_bifurcation(base, 'reversed/d1_idx_%d_of_%d.mat', mod.resolution);
    
    plot_bifurcation( ...
        rc2, rc1, ...
        d2_values, d1_values, ...
        rcolor, ccolor, pcolor, ...
        "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
        "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
    )
    sgtitle("Model 3a - normal",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    plot_bifurcation(rc2_reversed, rc1_reversed, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
    sgtitle("Model 3a - reversed",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    
    % Model 3b - C2 normal
    mod = modb;
    
    % R–C2–P (vary d2)
    base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);
    
    rc2 = load_bifurcation(base, 'normal/d2_idx_%d_of_%d.mat', mod.resolution);
    rc2_reversed = load_bifurcation(base, 'reversed/d2_idx_%d_of_%d.mat', mod.resolution);
    
    % R–C1–P (vary d1)
    base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);
    
    rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);
    rc1_reversed = load_bifurcation(base, 'reversed/d1_idx_%d_of_%d.mat', mod.resolution);
    
    
    plot_bifurcation( ...
        rc2, rc1, ...
        d2_values, d1_values, ...
        rcolor, ccolor, pcolor, ...
        "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
        "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
    )
    sgtitle("Model 3b - normal",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
    sgtitle("Model 3b - reversed",'Interpreter','latex')
    exportgraphics(gcf, pdf_file, 'Append', true)
    close(gcf)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp("PDF created")
    disp(pdf_file)
end

function plot_bifurcation(rc2, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, d2title, d1title)
    
    figure('Visible','off');  % or 'on' if you want to preview
    
    % R–C2–P branch (vary d2)
    subplot(2,1,1)
    hold on
    
    plot(d2_values, rc2.Rmin,'Color',rcolor,'LineWidth',2)
    plot(d2_values, rc2.Rmax,'Color',rcolor,'LineWidth',2)
    r_line = plot(d2_values, rc2.Rmean,'Color',rcolor,'LineStyle',':','LineWidth',2);
    
    plot(d2_values, rc2.Cmin,'Color',ccolor,'LineWidth',2)
    plot(d2_values, rc2.Cmax,'Color',ccolor,'LineWidth',2)
    c_line = plot(d2_values, rc2.Cmean,'Color',ccolor,'LineStyle',':','LineWidth',2);
    
    legend_handles = [r_line c_line];
    legend_labels = {"$R$","$C_2$"};
    
    if any(rc2.Pmax > 1e-200)   % predator exists
        plot(d2_values, rc2.Pmin,'Color',pcolor,'LineWidth',2)
        plot(d2_values, rc2.Pmax,'Color',pcolor,'LineWidth',2)
        p_line = plot(d2_values, rc2.Pmean,'Color',pcolor,'LineStyle',':','LineWidth',2);
    
        legend_handles = [legend_handles p_line];
        legend_labels{end+1} = "$P$";
    end
    
    legend(legend_handles,legend_labels,'interpreter','latex')
    
    xlabel("$d_2$",'Interpreter','latex')
    ylabel("density",'Interpreter','latex')
    title(d2title,'Interpreter','latex')
    
    
    % R–C1–P branch (vary d1)
    subplot(2,1,2)
    hold on
    
    plot(d1_values, rc1.Rmin,'Color',rcolor,'LineWidth',2)
    plot(d1_values, rc1.Rmax,'Color',rcolor,'LineWidth',2)
    r_line = plot(d1_values, rc1.Rmean,'Color',rcolor,'LineStyle',':','LineWidth',2);
    
    plot(d1_values, rc1.Cmin,'Color',ccolor,'LineWidth',2)
    plot(d1_values, rc1.Cmax,'Color',ccolor,'LineWidth',2)
    c_line = plot(d1_values, rc1.Cmean,'Color',ccolor,'LineStyle',':','LineWidth',2);
    
    legend_handles = [r_line c_line];
    legend_labels = {"$R$","$C_1$"};
    
    if any(rc1.Pmax > 1e-290) 
        plot(d1_values, rc1.Pmin,'Color',pcolor,'LineWidth',2)
        plot(d1_values, rc1.Pmax,'Color',pcolor,'LineWidth',2)
        p_line = plot(d1_values, rc1.Pmean,'Color',pcolor,'LineStyle',':','LineWidth',2);
    
        legend_handles = [legend_handles p_line];
        legend_labels{end+1} = "$P$";
    end
    
    legend(legend_handles,legend_labels,'interpreter','latex')
    
    xlabel("$d_1$",'Interpreter','latex')
    ylabel("density",'Interpreter','latex')
    title(d1title,'Interpreter','latex')
    
    
end


%%% loads the statistics (Pmin (mean of local minima), Pmax (mean of local maxima), Pmean etc.) 
function stats = load_bifurcation(base_folder, filename_pattern, resolution)

stats.Rmin = zeros(resolution,1);
stats.Rmax = zeros(resolution,1);
stats.Rmean = zeros(resolution,1);

stats.Cmin = zeros(resolution,1);
stats.Cmax = zeros(resolution,1);
stats.Cmean = zeros(resolution,1);

stats.Pmin = zeros(resolution,1);
stats.Pmax = zeros(resolution,1);
stats.Pmean = zeros(resolution,1);

for idx = resolution:-1:1

    filename = fullfile(base_folder, ...
        sprintf(filename_pattern, idx, resolution));

    s = load(filename);
    x = s.x; % consider the whole time series since there should be no transient anymore

    % R
    local_maxima = findpeaks(x(:,1));

    if isempty(local_maxima)    % if no peaks, take max
        stats.Rmax(idx) = max(x(:,1));
    else                        % else take the average of the local maxima
        stats.Rmax(idx) = mean(local_maxima);
    end

    local_minima = findpeaks(-x(:,1));

    if isempty(local_minima)    % if no peaks, take min
        stats.Rmin(idx) = min(x(:,1));
    else                        % else take the average of the local minima
        stats.Rmin(idx) = -mean(local_minima);
    end

    stats.Rmean(idx) = mean(x(:,1));


    % C
    local_maxima = findpeaks(x(:,2));

    if isempty(local_maxima)    % if no peaks, take max
        stats.Cmax(idx) = max(x(:,2));
    else                        % else take the average of the local maxima
        stats.Cmax(idx) = mean(local_maxima);
    end

    local_minima = findpeaks(-x(:,2));

    if isempty(local_minima)    % if no peaks, take min
        stats.Cmin(idx) = min(x(:,2));
    else                        % else take the average of the local minima
        stats.Cmin(idx) = -mean(local_minima);
    end

    stats.Cmean(idx) = mean(x(:,2));

    if size(x,2) >= 3
        
        % P
        local_maxima = findpeaks(x(:,3));
    
        if isempty(local_maxima)    % if no peaks, take max
            stats.Pmax(idx) = max(x(:,3));
        else                        % else take the average of the local maxima
            stats.Pmax(idx) = mean(local_maxima);
        end
    
        local_minima = findpeaks(-x(:,3));
    
        if isempty(local_minima)    % if no peaks, take min
            stats.Pmin(idx) = min(x(:,3));
        else                        % else take the average of the local minima
            stats.Pmin(idx) = -mean(local_minima);
        end
    
        stats.Pmean(idx) = mean(x(:,3));

    end

end
end
