%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                          Coexistence models                         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Setup
%%%%%

extinction_threshold = 1e-300; % now used for all models

% parameters:
a1 = 1;
a2 = 2;
h2 = 8;
dP = 0.25;
resolution = 30;

% make grid
d1 = get_grid(a1,0, resolution);
d2 = get_grid(a2,h2, resolution);

% Get ordering of indices
global ordered_indices total_iterations indices_for_lin_sat
ordered_indices = spiral_order(resolution); % might not be a good approach
total_iterations = size(ordered_indices, 1);
indices_for_lin_sat = resolution:-1:1;

%%%%%%% Set ODE simulation parameters
global options_lin_sat options_lin_lin_lin options_lin_lin_sat ...
    options_lin_sat_lin options_lin_sat_sat time_lin_lin_lin ...
    time_lin_sat time_lin_lin_sat
options_lin_lin_lin = odeset('RelTol', 1e-12, 'AbsTol', 1e-12, 'NonNegative',1:4,'Events', @(t,x) extinction_event(t,x,extinction_threshold)); % high
options_lin_lin_sat = odeset('RelTol', 1e-6, 'AbsTol', 1e-11, 'NonNegative',1:4, 'Events', @(t,x) extinction_event(t,x,extinction_threshold));
options_lin_sat = odeset('RelTol', 1e-9, 'AbsTol', 1e-9, 'NonNegative',1:2, 'Events', @(t,x) extinction_event(t,x,extinction_threshold)); % somewhat higher, very high only for a few runs
options_lin_sat_lin = odeset('RelTol', 1e-6, 'AbsTol', 1e-11, 'NonNegative',1:4, 'Events', @(t,x) extinction_event(t,x,extinction_threshold));
options_lin_sat_sat = odeset('RelTol', 1e-12, 'AbsTol', 1e-12, 'NonNegative',1:4, 'Events', @(t,x) extinction_event(t,x,extinction_threshold)); % try high for a few that crashed
% lin_lin_lin longer for fixed point
time_lin_lin_lin = struct('tend', 120000, 'tstarteval', 90000);
% only for a few re-runs that were not normal
time_lin_lin_sat = struct('tend', 500000, 'tstarteval', 470000);
% lin_sat longer took too long, back to normal
time_lin_sat = struct('tend', 60000, 'tstarteval', 30000);

                                                                      %%%%%
                                                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Run the model(s)
%%%%%
 matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
 plot_a_matrix(matrix_lin_sat,a1,a2,h2,0,0.25,resolution)
% 
% matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_lin_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_lin_sat,a1,a2,h2,8,0.25,resolution)
% 
% matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_sat_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_sat_sat,a1,a2,h2,8,0.25,resolution,1) % 1 for additional colorbar


% parameters:
a1 = 1;
a2 = 4;
h2 = 0.5;
dP = 0.25;
resolution = 30;

% make grid
d1 = get_grid(a1,0, resolution);
d2 = get_grid(a2,h2, resolution);
% 
% matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
% plot_a_matrix(matrix_lin_sat,a1,a2,h2,0,0.25,resolution)
% 
% matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_lin_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_lin_sat,a1,a2,h2,8,0.25,resolution)
% 
% matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_sat_lin,a1,a2,h2,1,0.25,resolution
% 
% matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_sat_sat,a1,a2,h2,8,0.25,resolution,1)
% 

% parameters:
a1 = 1;
a2 = 8;
h2 = 0.5;
dP = 0.25;
resolution = 30;

% make grid
d1 = get_grid(a1,0, resolution);
d2 = get_grid(a2,h2, resolution);

% matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
% plot_a_matrix(matrix_lin_sat,a1,a2,h2,0,0.25,resolution)
% 
% matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_lin_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_lin_sat,a1,a2,h2,8,0.25,resolution)
% % 
% matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_sat_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_sat_sat,a1,a2,h2,8,0.25,resolution,1)


% parameters:
a1 = 1;
a2 = 2;
h2 = 0.125;
dP = 0.25;
resolution = 30;

% make grid
d1 = get_grid(a1,0, resolution);
d2 = get_grid(a2,h2, resolution);

% matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
% plot_a_matrix(matrix_lin_sat,a1,a2,h2,0,0.25,resolution)
% 
% matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_lin_lin,a1,a2,h2,1,0.25,resolution)

% matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_lin_sat,a1,a2,h2,8,0.25,resolution)

% matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
%                                            resolution); % aP = 1, hP = 0
% plot_a_matrix(matrix_lin_sat_lin,a1,a2,h2,1,0.25,resolution)
% 
% matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
%                                            resolution); % aP = 8, hP = 3
% plot_a_matrix(matrix_lin_sat_sat,a1,a2,h2,8,0.25,resolution,1)

%{
% parameters:
a1 = 1;
a2 = 2;
h2 = 0.25;
dP = 0.25;
resolution = 30;

% make grid
d1 = get_grid(a1,0, resolution);
d2 = get_grid(a2,h2, resolution);

matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
plot_a_matrix(matrix_lin_sat,a1,a2,h2,resolution)

matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
                                           resolution); % aP = 1, hP = 0
plot_a_matrix(matrix_lin_lin_lin,a1,a2,h2,1,0.25,resolution)

matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
                                           resolution); % aP = 8, hP = 3
plot_a_matrix(matrix_lin_lin_sat,a1,a2,h2,8,0.25,resolution)

matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
                                           resolution); % aP = 1, hP = 0
plot_a_matrix(matrix_lin_sat_lin,a1,a2,h2,1,0.25,resolution)

matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
                                           resolution); % aP = 8, hP = 3
plot_a_matrix(matrix_lin_sat_sat,a1,a2,h2,8,0.25,resolution,1)

%}

                                                                      %%%%%
                                                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Function definitions
%%%%%

%%%%%                        Plotting a matrix                        %%%%%
function plot_a_matrix(M, a1, a2, h2, aP, dP, resolution, colorbar_on)

    d1 = get_grid(a1,0, resolution);
    d2 = get_grid(a2,h2, resolution);

    
    set(groot, 'defaultAxesTickLabelInterpreter','latex')
    set(groot, 'defaultLegendInterpreter','latex')
    set(groot, 'DefaultTextInterpreter', 'latex')    

    if ~exist('colorbar_on','var')
        colorbar_on = 0; % default: no colorbar
    end
    figure()
    hold on
    for idx = 1:length(d1)
        for idy = 1:length(d2)
            scatter(d1(idx),d2(idy),50,M(idx,idy),'filled');
        end
    end
    
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
    set(gca,'Color',[1 1 1])   % background color
    %set(findobj(gca,'Type','Image'),'AlphaData',~isnan(M'))
    
    axis square

    xlabel("$d_1$","FontSize",18)
    ylabel("$d_2$","FontSize",18)
    if colorbar_on

        % one shared colorbar
        cb = colorbar;
        cb.Ticks = [1/3 1 5/3];
        cb.Limits = [0 2];
        cb.TickLabels = {
            'no coexistence'
            'static'
            'dynamic'
        };
        cb.FontSize = 18;
        
        cb.Ruler.TickLabelRotation=90;
    end
    
    BoundaryConditions(gcf,[a1,a2,h2,aP,dP],'all')
end
    

%%%%%                             lin sat                             %%%%%
function coexistence_matrix = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution)
   
    global indices_for_lin_sat options_lin_sat time_lin_sat
    C1_invasion = zeros(resolution,resolution);
    C2_invasion = zeros(resolution,resolution);
    initial_density = [0.01; 0.01]; % column vector

    % Loop through indices of C1
    for idx = indices_for_lin_sat
       
        R_star = d1(idx)/a1;
        C2_invasion(idx,:) = a2*R_star/(1+a2*h2*R_star) - d2;
    
    end
    
    % Loop through indices of C2
    for idx = indices_for_lin_sat
        
        linsat_simulation_params = struct('a', a2, 'h', h2, 'd', d2(idx));
    
        % Check if saved
        dirname = sprintf('./results/timeseries/temp/lin_sat/a2=%s_h2=%s',num2str(a2),num2str(h2));
        filename = sprintf('./results/timeseries/temp/lin_sat/a2=%s_h2=%s/a2=%s_h2=%s_d2=%s.mat',...
            num2str(a2), num2str(h2),  num2str(a2),  num2str(h2), num2str(d2(idx)));
    
        if isfile(filename)
            data = load(filename);
            x = data.x;
            t = data.t;
        else
            % If not saved:

            % Simulate ODE using the default time parameters
            [t, x] = simulate(@predator_prey, linsat_simulation_params, initial_density, options_lin_sat, time_lin_sat);
            % Save the simulation
            struct_to_be_saved = struct("x",x,"t",t,"params",linsat_simulation_params);
    
            if ~exist(dirname,"dir")
                mkdir(dirname)
            end
            save(filename,'-fromstruct',struct_to_be_saved);
        end
    
        % evaluate for coexistence
        integral_value = trapz(t, x); % 2 for integrating along columns(not needed in the end)
        time_length = t(end) - t(1);
        mean_of_x = integral_value / time_length;

        if any(x(:)<0)
            'also becomes negative'
        end

        % Check for NaN or negative values
        if all(~isnan(x), 'all')  % all not nan % sometimes just negative for a short part, is that very problematic?
            C1_invasion(:,idx) = a1 * mean_of_x(1) - d1';
        else
            C1_invasion(:,idx) = nan;
        end
        
        % set new initial density
        initial_density = set_new_initial_density(x);
    end
    C1_invasion(C1_invasion<0) = 0;
    C1_invasion(C1_invasion>0) = 1;
    C2_invasion(C2_invasion<0) = 0;
    C2_invasion(C2_invasion>0) = 1;
    
    coexistence_matrix = C1_invasion .* C2_invasion * 2;

end

%%%%%                            lin lin P                            %%%%%
function coexistence_matrix = lin_lin_P_coexistence(a1,a2,h2,aP,hP,dP,d1,d2,resolution)
    global ordered_indices total_iterations options_lin_lin_lin ...
        options_lin_lin_sat time_lin_lin_lin time_lin_lin_sat
    initial_density = [0.01; 0.01; 0.01; 0.01]; % column vector
    coexistence_matrix = zeros(resolution,resolution);

    % Loop through spiral order
    for idx = 1:total_iterations
        
        % Get current i, j from spiral order
        i = ordered_indices(idx, 1);
        j = ordered_indices(idx, 2);
    
        % Linearize the saturating functional response
        alin = (1-d2(j)*h2)*a2;
        hlin = 0;
        dlin = d2(j);
                
        linlinP_simulation_params = struct('a1', a1, 'h1', 0, 'a2', alin,...
            'h2', hlin, 'd1', d1(i), 'd2', dlin, 'aP', aP, 'hP', hP, 'dP', dP);
    
        % Check if saved
        dirname = sprintf('./results/timeseries/temp/lin_lin_P/a2=%s_h2=%s_aP=%s',num2str(a2),num2str(h2),num2str(aP));
        filename = sprintf('./results/timeseries/temp/lin_lin_P/a2=%s_h2=%s_aP=%s/a1=%s_a2=%s_aP=%s_h2=%s_hP=%s_d1=%s_d2=%s_dP=%s.mat',...
            num2str(a2),num2str(h2),num2str(aP),num2str(a1), num2str(alin),...
            num2str(aP), num2str(hlin),  num2str(hP), ...
            num2str(d1(i)), num2str(dlin), num2str(dP));
    
        if isfile(filename)
            data = load(filename);
            x = data.x;
            t = data.t;
        else
            % If not saved:

            % Simulate ODE using the default time parameters
            if aP == 1
                [t, x] = simulate(@full_system, ...
                    linlinP_simulation_params, initial_density, options_lin_lin_lin, time_lin_lin_lin);
            else
                [t, x] = simulate(@full_system, ...
                    linlinP_simulation_params, initial_density, options_lin_lin_sat, time_lin_lin_sat);
            end

            % Save the simulation
            struct_to_be_saved = struct("x",x,"t",t,"params",linlinP_simulation_params);
    
            if ~exist(dirname,"dir")
                mkdir(dirname)
            end
            save(filename,'-fromstruct',struct_to_be_saved);
        end
        
        % evaluate for coexistence
        coexistence_matrix(i,j) = evaluate_coexistence(t,x);
        % set new initial density
        initial_density = set_new_initial_density(x);
    end
end

%%%%%                            lin sat P                            %%%%%
function coexistence_matrix = lin_sat_P_coexistence(a1,a2,h2,aP,hP,dP,d1,d2,resolution)
    global ordered_indices total_iterations options_lin_sat_lin options_lin_sat_sat
    initial_density = [0.01; 0.01; 0.01; 0.01]; % column vector

    coexistence_matrix = zeros(resolution,resolution);

    % Loop through spiral order
    for idx = 1:total_iterations
        
        % Get current i, j from spiral order
        i = ordered_indices(idx, 1);
        j = ordered_indices(idx, 2);
                
        linsatP_simulation_params = struct('a1', a1, 'h1', 0, 'a2', a2,...
            'h2', h2, 'd1', d1(i), 'd2', d2(j), 'aP', aP, 'hP', hP, 'dP', dP);
    
        % Check if saved
        dirname = sprintf('./results/timeseries/temp/lin_sat_P/a2=%s_h2=%s_aP=%s',num2str(a2),num2str(h2),num2str(aP));
        filename = sprintf('./results/timeseries/temp/lin_sat_P/a2=%s_h2=%s_aP=%s/a1=%s_a2=%s_aP=%s_h1=%s_h2=%s_hP=%s_d1=%s_d2=%s_dP=%s.mat',...
            num2str(a2),num2str(h2),num2str(aP),num2str(a1), num2str(a2),...
            num2str(aP), num2str(0),  num2str(h2),  num2str(hP), ...
            num2str(d1(i)), num2str(d2(j)), num2str(dP));
    
        if isfile(filename)
            data = load(filename);
            x = data.x;
            t = data.t;
        else
            % If not saved:
            %fprintf("starting simulation\n")
            
            % Simulate ODE using the default time parameters
            if aP == 1
                [t, x] = simulate(@full_system, ...
                    linsatP_simulation_params, initial_density, options_lin_sat_lin);
            else
                [t, x] = simulate(@full_system, ...
                    linsatP_simulation_params, initial_density, options_lin_sat_sat);
            end
            %fprintf("simulated!\n")
            
            % Save the simulation
            struct_to_be_saved = struct("x",x,"t",t,"params",linsatP_simulation_params);
    
            if ~exist(dirname,"dir")
                mkdir(dirname)
            end
            save(filename,'-fromstruct',struct_to_be_saved);
        end
        
        % evaluate for coexistence
        coexistence_matrix(i,j) = evaluate_coexistence(t,x);
        % set new initial density
        initial_density = set_new_initial_density(x);
    
    end
end

%%%%%                            R-C system                           %%%%%
function dxdt = predator_prey(t, x, params)
       
    a = params.a;
    h = params.h;
    d = params.d;
    
    R = x(1);  % resource
    C = x(2);  % consumer
    
    Rdot = (1 - R) * R - a * C * R / (1 + a * h * R);
    Cdot = (a * R / (1 + a * h * R) - d) * C;
    
    dxdt = [Rdot; Cdot];
end

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
        tend = 60000;
        tstarteval = 30000;
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

%%%%%                       timeseries evaluation                     %%%%%
function result = evaluate_coexistence(t,x)

    integral_value = trapz(t, x);
    time_length = t(end) - t(1);
    mean_of_x = integral_value / time_length;


    % Check for NaN or negative values
    if any(isnan(x)) % all not nan but they might be negative somewhere
        result = nan;
    elseif all(mean_of_x > 1e-15) && all(x(end,:)> 1e-100)

        std_of_x = std(x);
        cv_of_x = std_of_x./mean_of_x;
        
        if all(cv_of_x < 0.01)
            result = 1;  % fixed point
        else
            result = 2;  % cycle
        end
    else
        result = 0;
    end
end

%%%%%                    continuation/initial density                 %%%%%
function initial_density = set_new_initial_density(x)
    
    if all(~isnan(x), 'all') && all(x >= 0, 'all') % all not nan and all positive
        initial_density = x(end, :) * 1.01;
        initial_density(initial_density < 10^-15) = 10^-10; % let them re-invade
    else
        initial_density = 0.01 * ones(size(x(end,:)));
    end 
end

%%%%%                    continuation/spiral pattern                  %%%%%    
function order = spiral_order(n)

    % Generate index order for spiral continuation
    % Returns n×2 matrix where each row is [i, j] pair
    
    order = [];  % will build as rows
    x = floor(n / 2) + 1;  % MATLAB 1-based indexing
    y = floor(n / 2) + 1;
    
    if mod(n, 2) == 0
        x = x - 1;
        y = y - 1;
    end
    
    dx = [0, 1, 0, -1];    % up, right, down, left
    dy = [1, 0, -1, 0];    % clockwise spiral
    direction = 1;          % 1-indexed direction
    step_size = 1;
    index = 0;
    
    while index < n * n
        for repeat = 1:2
            for step = 1:step_size
                if index >= n * n
                    return
                end
                
                % Check bounds (optional, depends on your use case)
                if x >= 1 && x <= n && y >= 1 && y <= n
                    order = [order; x, y];
                end
                
                x = x + dx(direction);
                y = y + dy(direction);
                index = index + 1;
            end
            direction = mod(direction, 4) + 1;  % 1-based mod
        end
        step_size = step_size + 1;
    end
end

%%%%%                    extinction event function                    %%%%%
function [value, isterminal, direction] = extinction_event(t, x, threshold)

    value = min(x) - threshold;   % Trigger when smallest biomass crosses threshold
    isterminal = 1;               % Stop the integration
    direction = -1;               % Only trigger when decreasing

end