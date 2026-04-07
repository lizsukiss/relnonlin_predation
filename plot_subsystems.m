%%% Plot bifurcation diagrams for both subsystems R-C1-P and R-C2-P

% colors
rcolor = [0.4660, 0.6740, 0.1880];
ccolor = [0.8500, 0.3250, 0.0980];
pcolor = [0.4940, 0.1840, 0.5560];

% parameters

% model 1
mod1.a1 = 1;
mod1.h1 = 0;
mod1.a2 = 2;
mod1.h2 = 8;
mod1.aP = 0;
mod1.hP = 0;
mod1.dP = 0.10;
mod1.resolution = 30;

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


%% Model 1
mod = mod1;

base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);

rc2 = load_bifurcation(base, 'normal/d2_idx_%d_of_%d.mat', mod.resolution);
%rc2_reversed = load_bifurcation(base, 'reversed/d2_idx_%d_of_%d.mat', mod.resolution);

base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);

rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);

plot_bifurcation(rc2, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow d_2^{\mathrm{max}}$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
sgtitle("Model 1",'Interpreter','latex')

%plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
%sgtitle("Model 1",'Interpreter','latex')

%% Model 2a - C2 normal
mod = moda;

% R–C2–P (vary d2)
base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);

rc2 = load_bifurcation(base, 'linearized/d2_idx_%d_of_%d.mat', mod.resolution);
%rc2_reversed = load_bifurcation(base, 'linearized_reversed/d2_idx_%d_of_%d.mat', mod.resolution);


% R–C1–P (vary d1)
base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);

rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);


plot_bifurcation( ...
    rc2, rc1, ...
    d2_values, d1_values, ...
    rcolor, ccolor, pcolor, ...
    "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
    "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
)
sgtitle("Model 2a",'Interpreter','latex')

%plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
%sgtitle("Model 2a",'Interpreter','latex')

%% Model 2b - C2 linearized
mod = modb;

% R–C2–P (vary d2)
base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);

rc2 = load_bifurcation(base, 'linearized/d2_idx_%d_of_%d.mat', mod.resolution);
%rc2_reversed = load_bifurcation(base, 'linearized_reversed/d2_idx_%d_of_%d.mat', mod.resolution);


% R–C1–P (vary d1)
base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);

rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);


plot_bifurcation( ...
    rc2, rc1, ...
    d2_values, d1_values, ...
    rcolor, ccolor, pcolor, ...
    "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
    "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
)
sgtitle("Model 2b",'Interpreter','latex')

%plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
%sgtitle("Model 2b",'Interpreter','latex')

%% Model 3a - C2 normal 
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


plot_bifurcation( ...
    rc2, rc1, ...
    d2_values, d1_values, ...
    rcolor, ccolor, pcolor, ...
    "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
    "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
)
sgtitle("Model 3a",'Interpreter','latex')

%plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
%sgtitle("Model 3a",'Interpreter','latex')


%% Model 3b - C2 normal
mod = modb;

% R–C2–P (vary d2)
base = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a2, mod.h2, mod.aP, mod.hP, mod.dP);

rc2 = load_bifurcation(base, 'normal/d2_idx_%d_of_%d.mat', mod.resolution);
%rc2_reversed = load_bifurcation(base, 'reversed/d2_idx_%d_of_%d.mat', mod.resolution);

% R–C1–P (vary d1)
base = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
    mod.a1, mod.h1, mod.aP, mod.hP, mod.dP);

rc1 = load_bifurcation(base, 'normal/d1_idx_%d_of_%d.mat', mod.resolution);


plot_bifurcation( ...
    rc2, rc1, ...
    d2_values, d1_values, ...
    rcolor, ccolor, pcolor, ...
    "$d_2 \rightarrow d_2^{\mathrm{max}}$", ...
    "$d_1 \rightarrow d_1^{\mathrm{max}}$" ...
)
sgtitle("Model 3b",'Interpreter','latex')

%plot_bifurcation(rc2_reversed, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, "$d_2 \rightarrow 0$", "$d_1 \rightarrow d_1^{\mathrm{max}}$")
%sgtitle("Model 3b",'Interpreter','latex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_bifurcation(rc2, rc1, d2_values, d1_values, rcolor, ccolor, pcolor, d2title, d1title)

figure()

%% R–C2–P branch (vary d2)
subplot(2,1,1)
hold on

plot(d2_values, rc2.Rmin,'Color',rcolor,'LineWidth',2)
plot(d2_values, rc2.Rmax,'Color',rcolor,'LineWidth',2)
r_line = plot(d2_values, rc2.Rmean,'Color',rcolor,'LineStyle',':','LineWidth',2);

plot(d2_values, rc2.Cmin,'Color',ccolor,'LineWidth',2)
plot(d2_values, rc2.Cmax,'Color',ccolor,'LineWidth',2)
c_line = plot(d2_values, rc2.Cmean,'Color',ccolor,'LineStyle',':','LineWidth',2);

legend_handles = [r_line c_line];
legend_labels = {"R","C_2"};

if any(rc2.Pmax > 1e-200)   % predator exists
    plot(d2_values, rc2.Pmin,'Color',pcolor,'LineWidth',2)
    plot(d2_values, rc2.Pmax,'Color',pcolor,'LineWidth',2)
    p_line = plot(d2_values, rc2.Pmean,'Color',pcolor,'LineStyle',':','LineWidth',2);

    legend_handles = [legend_handles p_line];
    legend_labels{end+1} = "P";
end

legend(legend_handles,legend_labels,'interpreter','latex')

xlabel("$d_2$",'Interpreter','latex')
ylabel("density",'Interpreter','latex')
title(d2title,'Interpreter','latex')


%% R–C1–P branch (vary d1)
subplot(2,1,2)
hold on

plot(d1_values, rc1.Rmin,'Color',rcolor,'LineWidth',2)
plot(d1_values, rc1.Rmax,'Color',rcolor,'LineWidth',2)
r_line = plot(d1_values, rc1.Rmean,'Color',rcolor,'LineStyle',':','LineWidth',2);

plot(d1_values, rc1.Cmin,'Color',ccolor,'LineWidth',2)
plot(d1_values, rc1.Cmax,'Color',ccolor,'LineWidth',2)
c_line = plot(d1_values, rc1.Cmean,'Color',ccolor,'LineStyle',':','LineWidth',2);

legend_handles = [r_line c_line];
legend_labels = {"R","C_1"};

if any(rc1.Pmax > 1e-200)
    plot(d1_values, rc1.Pmin,'Color',pcolor,'LineWidth',2)
    plot(d1_values, rc1.Pmax,'Color',pcolor,'LineWidth',2)
    p_line = plot(d1_values, rc1.Pmean,'Color',pcolor,'LineStyle',':','LineWidth',2);

    legend_handles = [legend_handles p_line];
    legend_labels{end+1} = "P";
end

legend(legend_handles,legend_labels)

xlabel("$d_1$",'Interpreter','latex')
ylabel("density",'Interpreter','latex')
title(d1title,'Interpreter','latex')

end