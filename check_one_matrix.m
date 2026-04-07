a1 = 1;
h1 = 0;
a2 = 13.125;
h2 = 0.5;
resolution = 30;

% make grid
d1 = get_grid(a1,h1,resolution);
d2 = get_grid(a2,h2,resolution);

% C2 is invading if f2(R_{C_1}^*) > 0
invasionrate_C2 = zeros(resolution, resolution);

for i = 1:resolution
    R_Eq = d1(i) / a1;
    invasionrate_C2(i, :) = a2 * R_Eq / (1 + a2 * h2 * R_Eq) - d2;
end

invasionrate_C2(invasionrate_C2 > 0) = 1;
invasionrate_C2(invasionrate_C2 < 0) = 0;

% C1 is invading C2
invasionrate_C1 = zeros(resolution, resolution);

initial_density = [0.01; 0.01]; % column vector

for i = resolution:-1:1  % backwards loop
    i
    
    rc_simulation_params = struct('a', a2, 'h', h2, 'd', d2(i));

    % Check if saved
    filename = sprintf('./results/timeseries/a=%s_h=%s_d=%s.mat',num2str(a2), num2str(h2), num2str(d2(i)));
    if isfile(filename)
        data = load(filename);
        x = data.x;
        t = data.t;
        %params = data.params;
    else
        % If not saved:
        % Simulation parameters
        tend = 50000;
        tstarteval = 30000;
        tstep = 0.1;
        time_params = struct('tstarteval', tstarteval, 'tstep', tstep, 'tend', tend);
                    
        % Simulate ODE
        [t, x] = simulate(time_params, @predator_prey, rc_simulation_params, initial_density);

    end

    % Evaluate at desired time points (only last portion for speed)
    initial_density = x(:, end) * 1.01;
    
    % Check for NaN or negative values
    if any(isnan(x(:))) || any(x(:) < 0)
        invasionrate_C1(:, i) = nan;
        initial_density = [0.01; 0.01];
    else
        average_R_density = mean(x(1, :));
        invasionrate_C1(:, i) = a1 * average_R_density - d1';
    end
    
end

% Convert to binary (1 or 0)
invasionrate_C1(invasionrate_C1 > 0) = 1;
invasionrate_C1(invasionrate_C1 < 0) = 0;

% Compute coexistence
C1lin_C2sat_coexistence = invasionrate_C1 .* invasionrate_C2 *2; % *2 since it is a limit cycle


imagesc(C1lin_C2sat_coexistence')



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

% R-C1+C2-P system
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

    R  = x(0);  % resource
    C1 = x(1);  % consumer 1
    C2 = x(2);  % consumer 2
    P  = x(3);  % predator    
    
    Rdot = ( (1 - R) - a1 * C1 / (1 + a1 * h1 * R) - a2 * C2 / (1 + a2 * h2 * R) ) * R;
    C1dot = ( a1 * R / (1 + a1 * h1 * R) - d1 - aP * P / (1 + aP * hP * (C1 + C2)) ) * C1;
    C2dot = ( a2 * R / (1 + a2 * h2 * R) - d2 - aP * P / (1 + aP * hP * (C1 + C2)) ) * C2;
    Pdot = ( aP * (C1 + C2) / (1 + aP * hP * (C1 + C2)) - dP ) * P;
        
    dxdt = [Rdot,C1dot,C2dot,Pdot];
end

function grid = get_grid(a, h, resolution)
    % Creates numerical axis for the mortality grid
        
    maxd = a / (1 + h * a);
    grid = linspace(0, maxd, resolution + 2);
    grid = grid(2:end-1);
end

function [t, x] = simulate(time, ode_function, simulation_params, initial_conditions)
    % time is a struct with tstarteval, tstep and tend

    tend = time.tend;
    tstep = time.tstep;
    tstarteval = time.tstarteval;
    t = tstarteval:tstep:tend;
    
    % Define ODE function as anonymous function
    % Note: MATLAB ODE solvers expect (t, x) signature
    ode_func = @(t, x) ode_function(t, x, simulation_params);
    
    % Simulate ODE
    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-15);
    sol = ode45(ode_func, [0 tend], initial_conditions, options);
    
    % Evaluate solution at desired time points
    x = deval(sol, t);

end