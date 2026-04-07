% Bifurcation in the R-C2-P system  
%% parameters
params.a1 = 1;
params.a2 = 2;
params.h2 = 8;
params.aP = 1;
params.hP = 0;
params.dP = 0.25;
params.resolution = 30;

d1_values = get_grid(params.a1,0,params.resolution);
d2_values = get_grid(params.a2,params.h2,params.resolution);

%% d2 from above

initial_conditions = [0.1 0.1 0.0000001];

Rmin = zeros(params.resolution,1);
Rmax = zeros(params.resolution,1);
Rmean = zeros(params.resolution,1);

Cmin = zeros(params.resolution,1);
Cmax = zeros(params.resolution,1);
Cmean = zeros(params.resolution,1);

Pmin = zeros(params.resolution,1);
Pmax = zeros(params.resolution,1);
Pmean = zeros(params.resolution,1);



for idx = params.resolution:-1:1

    idx

    params.d2 = d2_values(idx);

    tic
    [R, C, P, last] = Bifurcation_Helper(params, initial_conditions);
    toc

    initial_conditions = last + [1 1 1] * 1e-15 + [0 0.5 0];

    Rmin(idx) = R.min;
    Rmax(idx) = R.max;
    Rmean(idx) = R.mean;

    Cmin(idx) = C.min;
    Cmax(idx) = C.max;
    Cmean(idx) = C.mean;

    Pmin(idx) = P.min;
    Pmax(idx) = P.max;
    Pmean(idx) = P.mean;
end



figure()
hold on

plot(d2_values, Rmin', "green")
plot(d2_values, Rmax', "green")
r_line = plot(d2_values, Rmean', "green");

plot(d2_values, Cmin', "yellow")
plot(d2_values, Cmax', "yellow")
c_line = plot(d2_values, Cmean', "yellow");

plot(d2_values, Pmin', "magenta")
plot(d2_values, Pmax', "magenta")
p_line = plot(d2_values, Pmean', "magenta");
legend([r_line,c_line,p_line],{"R","C","P"})
title("$d_2 \rightarrow 0$ + $P$ always starts high")
xlabel("$d_2$")
ylabel("density")



%% from below:
initial_conditions = [0.1 0.1 0.0000001];

Rmin = zeros(params.resolution,1);
Rmax = zeros(params.resolution,1);
Rmean = zeros(params.resolution,1);

Cmin = zeros(params.resolution,1);
Cmax = zeros(params.resolution,1);
Cmean = zeros(params.resolution,1);

Pmin = zeros(params.resolution,1);
Pmax = zeros(params.resolution,1);
Pmean = zeros(params.resolution,1);

for idx = 1:params.resolution

    idx

    params.d2 = d2_values(idx);

    tic
    [R, C, P, last] = Bifurcation_Helper(params, initial_conditions);
    toc

    initial_conditions = last + [1 1 1] * 1e-15;

    Rmin(idx) = min(R);
    Rmax(idx) = max(R);
    Rmean(idx) = mean(R);

    Cmin(idx) = min(C);
    Cmax(idx) = max(C);
    Cmean(idx) = mean(C);

    Pmin(idx) = min(P);
    Pmax(idx) = max(P);
    Pmean(idx) = mean(P);
end

figure()
hold on

plot(d2_values, Rmin', "green")
plot(d2_values, Rmax', "green")
r_line = plot(d2_values, Rmean', "green");

plot(d2_values, Cmin', "yellow")
plot(d2_values, Cmax', "yellow")
c_line = plot(d2_values, Cmean', "yellow");

plot(d2_values, Pmin', "magenta")
plot(d2_values, Pmax', "magenta")
p_line = plot(d2_values, Pmean', "magenta");

legend([r_line,c_line,p_line],{"R","C","P"})
title("$d_2 \rightarrow d_2^{\mathrm{max}}$")
xlabel("$d_2$")
ylabel("density")


% compute the invasion growth rates
%% compute the invasion growth rate of C1
C1_invasion_rate = zeros(params.resolution);

for idx = params.resolution:-1:1 % idx is the index of d2 but also Rmin/max/mean, Cmin/max/mean etc.
    
    C1_invasion(:,idx) = params.a1 * Rmean(idx) - d1_values - params.aP * Pmean(idx); % this only holds if P is linear 
    
end

%% compute the invasion growth rate of C2

initial_conditions = [0.1 0.1 0.1];

Rmean_c1 = zeros(params.resolution,1);

Cmean_c1 = zeros(params.resolution,1);

Pmean_c1 = zeros(params.resolution,1);

helper_params.a2 = params.a1;
helper_params.aP = params.aP;
helper_params.hP = params.hP;
helper_params.dP = params.dP;
helper_params.h2 = 0;

for idx = params.resolution:-1:1

    idx

    helper_params.d2 = d1_values(idx);

    tic
    [R, C, P, last] = Bifurcation_Helper(helper_params, initial_conditions);
    toc

    initial_conditions = last + [1 1 1] * 1e-15;

    Rmean_c1(idx) = mean(R);

    Cmean_c1(idx) = mean(C);

    Pmean_c1(idx) = mean(P);

end

figure()
hold on

r_line = plot(d1_values, Rmean_c1', "green")

c_line = plot(d1_values, Cmean_c1', "yellow")

p_line = plot(d1_values, Pmean_c1', "magenta")

legend([r_line,c_line,p_line],{"R","C","P"})

title("$d_1 \rightarrow 0$")
xlabel("$d_1$")
ylabel("density")


C2_invasion_rate = zeros(params.resolution);

for idx = params.resolution:-1:1 % idx is the index of d1 but also Rmean_c1, Cmean_c1 etc.
    
    C2_invasion(idx,:) = params.a2 * Rmean_c1(idx)/(1+params.a2*params.h2*Rmean_c1(idx)) - d2_values - params.aP * Pmean_c1(idx); % this only holds if P is linear 
    
end

C1_invasion(C1_invasion>0) = 1;
C1_invasion(C1_invasion<0) = 0;

C2_invasion(C2_invasion>0) = 2;
C2_invasion(C2_invasion<0) = 0;

%% plot the mutual invasibility

figure()
hold on

% discrete colormap for 0/1/2/3
cmap = [ ...
    0 0 0;             % 0 → white
    0, 0.4470, 0.7410;       % 1 → blue (only C1 can invade)
    0.8500, 0.3250, 0.0980;  % 2 → orange (only C2 can invade)
    0.4940, 0.1840, 0.5560   % 3 → purple (both)
];

[D1, D2] = meshgrid(d1_values, d2_values);
for k = 0:3
    mask = (C1_invasion + C2_invasion)' == k;
    scatter(D1(mask), D2(mask), 50, cmap(k+1,:), 'filled');
end



colormap(cmap)
caxis([0 3])
axis square
legend({'mutually protected','C1 can invade','C2 can invade','mutual invasibility'});