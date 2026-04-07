%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                   Plot models based on time series                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Setup
%%%%%
% parameters:
a1 = 1;
a2 = 4;
h2 = .5;
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
                                                                      %%%%%
                                                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Run the model(s)
%%%%%
matrix_lin_sat = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution);
plot_a_matrix(matrix_lin_sat)
'here'

matrix_lin_lin_lin = lin_lin_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
                                           resolution); % aP = 1, hP = 0
plot_a_matrix(matrix_lin_lin_lin)
'here 2'
%{
matrix_lin_lin_sat = lin_lin_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
                                           resolution); % aP = 8, hP = 3
plot_a_matrix(matrix_lin_lin_sat)

matrix_lin_sat_lin = lin_sat_P_coexistence(a1,a2,h2,1,0,dP,d1,d2,...
                                           resolution); % aP = 1, hP = 0
matrix_lin_sat_sat = lin_sat_P_coexistence(a1,a2,h2,8,3,dP,d1,d2,...
                                           resolution); % aP = 8, hP = 3
%}
                                                                      %%%%%
                                                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Function definitions
%%%%%

%%%%%                        Plotting a matrix                        %%%%%
function plot_a_matrix(M)
    figure()
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
    

%%%%%                             lin sat                             %%%%%
function coexistence_matrix = lin_sat_coexistence(a1,a2,h2,d1,d2,resolution)
   
    global indices_for_lin_sat
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
                
            % evaluate for coexistence
            x = x';
            mean_of_x = mean(x);
            
            if any(x(:)<0)
                'also becomes negative'
            end
    
            % Check for NaN or negative values
            if all(~isnan(x), 'all')  % all not nan % sometimes just negative for a short part, is that very problematic?
                C1_invasion(:,idx) = a1 * mean_of_x(1) - d1';
            else
                C1_invasion(:,idx) = nan;
            end
        else
            % If not saved: nothing
            C1_invasion(:,idx) = nan;
        end
    
        
        
    end
    C1_invasion(C1_invasion<0) = 0;
    C1_invasion(C1_invasion>0) = 1;
    C2_invasion(C2_invasion<0) = 0;
    C2_invasion(C2_invasion>0) = 1;
    
    coexistence_matrix = C1_invasion .* C2_invasion * 2;

end

%%%%%                            lin lin P                            %%%%%
function coexistence_matrix = lin_lin_P_coexistence(a1,a2,h2,aP,hP,dP,d1,d2,resolution)
    global ordered_indices total_iterations
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
            %fprintf("already saved!\n")
            x = x';
            % evaluate for coexistence
            coexistence_matrix(i,j) = evaluate_coexistence(x);
        else
            % If not saved: nothing
            coexistence_matrix(i,j) = nan;
        end
        
    end
end

%%%%%                            lin sat P                            %%%%%
function coexistence_matrix = lin_sat_P_coexistence(a1,a2,h2,aP,hP,dP,d1,d2,resolution)
    global ordered_indices total_iterations
    initial_density = [0.01; 0.01; 0.01; 0.01]; % column vector

    coexistence_matrix = zeros(resolution_resolution);

    % Loop through spiral order
    for idx = 1:total_iterations
        
        % Get current i, j from spiral order
        i = ordered_indices(idx, 1);
        j = ordered_indices(idx, 2);
                
        linsatP_simulation_params = struct('a1', a1, 'h1', h1, 'a2', a2,...
            'h2', h2, 'd1', d1(i), 'd2', d2, 'aP', aP, 'hP', hP, 'dP', dP);
    
        % Check if saved
        dirname = sprintf('./results/timeseries/temp/lin_sat_P/a2=%s_h2=%s_aP=%s',num2str(a2),num2str(h2),num2str(aP));
        filename = sprintf('./results/timeseries/temp/lin_sat_P/a2=%s_h2=%s_aP=%s/a1=%s_a2=%s_aP=%s_h1=%s_h2=%s_hP=%s_d1=%s_d2=%s_dP=%s.mat',...
            num2str(a2),num2str(h2),num2str(aP),num2str(a1), num2str(a2),...
            num2str(aP), num2str(h1),  num2str(h2),  num2str(hP), ...
            num2str(d1(i)), num2str(dlin), num2str(dP));
    
        if isfile(filename)
            data = load(filename);
            x = data.x;
            t = data.t;
            fprintf("already saved!\n")
            % evaluate for coexistence
            coexistence_matrix(i,j) = evaluate_coexistence(x);
        else
            % If not saved: nothing
           coexistence_matrix(i,j) = nan;
        end 
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
    
    if mod(t, 25) == 0
        t
    end

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

%%%%%                            grid making                          %%%%%
function grid = get_grid(a, h, resolution)
    % Creates numerical axis for the mortality grid
        
    maxd = a / (1 + h * a);
    grid = linspace(0, maxd, resolution + 2);
    grid = grid(2:end-1);
end

%%%%%                            simulation                           %%%%%
function [t, x] = simulate(ode_function, simulation_params, initial_conditions, time)
    % setting the time array
    if ~exist('time','var')
        tend = 50000;
        tstarteval = 30000;
        tstep = 0.1;
    else
        tend = time.tend;
        tstep = time.tstep;
        tstarteval = time.tstarteval;
    end
    
    t = tstarteval:tstep:tend;
    
    % Define ODE function as anonymous function
    % Note: MATLAB ODE solvers expect (t, x) signature
    ode_func = @(t, x) ode_function(t, x, simulation_params);
    
    % Simulate ODE
    options = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    sol = ode23(ode_func, [0 tend], initial_conditions, options);
    
    % Evaluate solution at desired time points
    x = deval(sol, t);

end

%%%%%                       timeseries evaluation                     %%%%%
function result = evaluate_coexistence(x)

    mean_of_x = mean(x);

    % Check for NaN or negative values
    if all(~isnan(x), 'all') % all not nan but they might be negative somewhere
        result = nan;
    elseif all(mean_of_x > 1e-15)

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
        initial_density(initial_density < 10^-15) = 10^-15; % let them re-invade
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

