%% for R - C

[file,location] = uigetfile('*.mat');

load(fullfile(location,file))

figure(); clf;

hold on;
%plot(time,x(:,1),'Linewidth',3);
%plot(time,x(:,2),'Linewidth',3);
plot(t,log10(x(:,1)),'Linewidth',3);
plot(t,log10(x(:,2)),'Linewidth',3);

params


legend("R", "C")

box('on');


%% for R - C1+C2 - P

[file,location] = uigetfile('*.mat');

load(fullfile(location,file))

figure(); clf;

hold on;
%plot(t,x(:,1),'Linewidth',3);
%plot(t,x(:,2),'Linewidth',3);
%plot(t,x(:,3),'Linewidth',3);
%plot(t,x(:,4),'LineWidth',3);
plot(t,log10(x(:,1)),'Linewidth',3);
plot(t,log10(x(:,2)),'Linewidth',3);
plot(t,log10(x(:,3)),'Linewidth',3);
plot(t,log10(x(:,4)),'Linewidth',3);
% params

legend("$R$", "$C_1$", "$C_2$", "$P$",'Interpreter','latex')

box('on');

%% for R - C - P
[file,location] = uigetfile('*.mat');

load(fullfile(location,file))

figure(); clf;

hold on;
% plot(x(:,1),'Linewidth',3);
% plot(x(:,2),'Linewidth',3);
% plot(x(:,3),'Linewidth',3);
plot(log10(x(:,1)),'Linewidth',3);
plot(log10(x(:,2)),'Linewidth',3);
plot(log10(x(:,3)),'Linewidth',3);

legend("$R$", "$C$", "$P$",'Interpreter','latex')

box('on');

%% used for the figure showing examples of alternative stable states

[file,location] = uigetfile('*.mat');

load(fullfile(location,file))

fig = figure(); clf;

subplot(2,2,1) % important timeseries 2

hold on;
plot(t,x(:,1),'Linewidth',3);
plot(t,x(:,2),'Linewidth',3);
plot(t,x(:,3),'Linewidth',3);
plot(t,x(:,4),'LineWidth',3);
% plot(t,log10(x(:,1)),'Linewidth',3);
% plot(t,log10(x(:,2)),'Linewidth',3);
% plot(t,log10(x(:,3)),'Linewidth',3);
% plot(t,log10(x(:,4)),'Linewidth',3);
xlim([28500,30000])
ylim([0,max(max(x(:,:)))*1.1])
xticks([28500, 29000, 29500, 30000])
xticklabels({'28500', '29000', '29500', '30000'})
yticks([0,.5,1,2])
[file,location] = uigetfile('*.mat');
ylabel('population density','Interpreter','latex',FontSize=12)
title('complete food web','Interpreter','latex','fontsize',12)

load(fullfile(location,file))
subplot(2,2,3)  % important timeseries 3
hold on;
plot(t,x(:,1),'Linewidth',3);
plot(t,x(:,2),'Linewidth',3);
plot(t,x(:,3),'Linewidth',3);
plot(t,x(:,4),'LineWidth',3);
% plot(t,log10(x(:,1)),'Linewidth',3);
% plot(t,log10(x(:,2)),'Linewidth',3);
% plot(t,log10(x(:,3)),'Linewidth',3);
% plot(t,log10(x(:,4)),'Linewidth',3);
xlim([28500,30000])
ylim([0,max(max(x(:,:)))*1.1])
xticks([28500, 29000, 29500, 30000])
xticklabels({'28500', '29000', '29500', '30000'})
yticks([0,.5,1,2])
ylabel('population density','Interpreter','latex',FontSize=12)
xlabel('time','Interpreter','latex',FontSize=12)
title('absence of predator','Interpreter','latex','fontsize',12)

[file,location] = uigetfile('*.mat');

load(fullfile(location,file))
subplot(2,2,2) % important timeseries 0

hold on;
plot(t,x(:,1),'Linewidth',3);
plot(t,x(:,2),'Linewidth',3);
plot(t,x(:,3),'Linewidth',3);
plot(t,x(:,4),'LineWidth',3);
% plot(t,log10(x(:,1)),'Linewidth',3);
% plot(t,log10(x(:,2)),'Linewidth',3);
% plot(t,log10(x(:,3)),'Linewidth',3);
% plot(t,log10(x(:,4)),'Linewidth',3);
xlim([29500,30000])
ylim([0,max(max(x(:,:)))*1.1])
xticks([28500, 29000, 29500, 29750, 30000])
xticklabels({'28500', '29000', '29500', '29750','30000'})
yticks([0,.5,1,2])
[file,location] = uigetfile('*.mat');
ylabel('population density','Interpreter','latex',FontSize=12)
title('complete food web','Interpreter','latex','fontsize',12)

load(fullfile(location,file))
subplot(2,2,4)   % important timeseries 1
hold on;
plot(t,x(:,1),'Linewidth',3);
plot(t,x(:,2),'Linewidth',3);
plot(t,x(:,3),'Linewidth',3);
plot(t,x(:,4),'LineWidth',3);
% plot(t,log10(x(:,1)),'Linewidth',3);
% plot(t,log10(x(:,2)),'Linewidth',3);
% plot(t,log10(x(:,3)),'Linewidth',3);
% plot(t,log10(x(:,4)),'Linewidth',3);
xlim([29500,30000])
ylim([0,max(max(x(:,:)))*1.1])
xticks([28500, 29000, 29500, 29750, 30000])
xticklabels({'28500', '29000', '29500', '29750', '30000'})
yticks([0,.5,1,2])
ylabel('population density','Interpreter','latex',FontSize=12)
xlabel('time','Interpreter','latex',FontSize=12)
title('absence of consumer 2','Interpreter','latex','fontsize',12)

legend("$R$", "$C_1$", "$C_2$", "$P$",'Interpreter','latex','fontsize',12,'orientation','horizontal')
sgtitle('Alternative stable states','Interpreter','latex','fontsize',12)
box('on');




%%
fileindex = 0;
%% Simulate one instance

params.a1 = 1;
params.h1 = 0;
params.a2 = 2;
params.h2 = 8;
params.aP = 2.5;
params.dP = 0.1;
params.hP = 1;
params.d1 = 0.33;
params.d2 = 0.06;

initial_density = [1 0.00000000001 .1 0.1];

options = odeset('RelTol', 1e-9, 'AbsTol', 1e-11, 'NonNegative',1:4);

[t, x] = simulate(@full_system, ...
                    params, initial_density, options);

filename = sprintf('./individual_simulation/important_timeseries_%d',fileindex);

% Save the simulation
struct_to_be_saved = struct("x",x,"t",t,"params",params);
    
save(filename,'-fromstruct',struct_to_be_saved);
sprintf("Saved to: %s",filename)

fileindex = fileindex + 1;

%%%%%                         R-C1+C2-P system                        %%%%%
function dxdt = full_system(t, x, params)
    
    a1 = params.a1;
    a2 = params.a2;
    aP = params.aP;
    h1 = params.h1;
    h2 = params.h2;
    hP = params.hP;
    d1 = params.d1;
    d2 = params.d2;
    dP = params.dP;

    R  = x(1);  % resource
    C1 = x(2);  % consumer 1
    C2 = x(3);  % consumer 2
    P  = x(4);  % predator    
    
    Rdot = ( (1 - R) - a1 * C1 / (1 + a1 * h1 * R) - a2 * C2 / (1 + a2 * h2 * R) ) * R;
    C1dot = ( a1 * R / (1 + a1 * h1 * R) - d1 - aP * P / (1 + aP * hP * (C1 + C2)) ) * C1;
    C2dot = ( a2 * R / (1 + a2 * h2 * R) - d2 - aP * P / (1 + aP * hP * (C1 + C2)) ) * C2;
    Pdot = ( aP * (C1 + C2) / (1 + aP * hP * (C1 + C2)) - dP ) * P;
        
    dxdt = [Rdot;C1dot;C2dot;Pdot];
end


%%%%%                            simulation                           %%%%%
function [t, x] = simulate(ode_function, simulation_params, initial_conditions, options, time)

    tic

    % setting the time array
    if ~exist('time','var')
        tend = 30000;
        tstarteval = 0;
    else
        tend = time.tend;
        tstarteval = time.tstarteval;
    end
        
    % Define ODE function as anonymous function
    % Note: MATLAB ODE solvers expect (t, x) signature
    ode_func = @(t, x) ode_function(t, x, simulation_params);
    
    % Simulate ODE
    [t_full, x_full] = ode23(ode_func, [0 tend], initial_conditions, options);

    if t_full(end) == tend % simulation finished, save only after tstarteval
        mask = t_full >= tstarteval;
        t = t_full(mask);
        x = x_full(mask,:);
    else
        t = t_full(floor(length(t_full)/2):end); % otherwise keep the second half of the simulation until then
        x = x_full(floor(length(t_full)/2):end,:); 
    end
    toc

end
