%%% a2_values = [2, 4, 4, 8];
%%% h2_values = [8, 0.25, 1, 0.5];
% Define varying parameters
a2_values = 2.^linspace(-2,3,21);
h2_values = 2.^linspace(-2,3,21);

% Parameters for all sets
a1 = 1;
h1 = 0;
resolution = 30;

parameter_sets = {};
set_idx = 1;
for a2_value = a2_values
    for h2_value = h2_values
            params = struct();
            params.a1 = a1; 
            params.h1 = h1;
            params.a2 = a2_value;
            params.h2 = h2_value;
            params.resolution = resolution;
            parameter_sets{set_idx} = params;
            set_idx = set_idx + 1;
    end
end

for idx = 1:numel(parameter_sets)
    idx
    parameters = parameter_sets{idx};
    translate2matrix_basic(parameters.a1, parameters.a2, parameters.h2, parameters.resolution);
end
%{
for idx = 1:numel(parameter_sets)

    try
        basic_parameters = parameter_sets{idx};
    
        % print info
        disp('------------------------------------------------------------------')
        fields = fieldnames(basic_parameters);
        str = '';
        for j = 1:numel(fields)
            str = [str, sprintf('%s = %.2f; ', fields{j}, basic_parameters.(fields{j}))];
        end
        fprintf('New parameter set: %s\n', str);
        % setting parameters for the various models
        params_lin_sat = basic_parameters;
        params_lin_sat.aP = 0;
        params_lin_sat.hP = 0;
        params_lin_sat.dP = 0.1;
        
        params_lin_C2_lin = basic_parameters; % only later linearized
        params_lin_C2_lin.aP = 2.25;
        params_lin_C2_lin.hP = 0;
        params_lin_C2_lin.dP = 0.1;
        
        params_lin_C2_sat = basic_parameters; % only later linearized
        params_lin_C2_sat.aP = 2.5;
        params_lin_C2_sat.hP = 1;
        params_lin_C2_sat.dP = 0.1;

        % actually run the simulations
        disp('starting lin sat');
        mutual_invasibility(params_lin_sat)
        disp('starting P lin');
        mutual_invasibility(params_lin_C2_lin)
        disp('starting P sat');
        mutual_invasibility(params_lin_C2_sat)
        disp('starting C1 and C2');
        rnl_simulation(params_lin_sat)
        fprintf('Everything done for: %s\n',str);

    catch ME
        fprintf('ERROR in iteration %d\n', idx);
        fprintf('Message: %s\n', ME.message);
        fprintf('Stack:\n');
        for k = 1:length(ME.stack)
            fprintf('  File: %s | Function: %s | Line: %d\n', ...
                ME.stack(k).file, ME.stack(k).name, ME.stack(k).line);
        end
        rethrow(ME); % optional: stops everything
    end
end
%}

function mutual_invasibility(parameters)

    % grids
    d1_values = get_grid(parameters.a1, parameters.h1, parameters.resolution);
    d2_values = get_grid(parameters.a2, parameters.h2, parameters.resolution);

    % solver settings 
    RelT     = 1e-9;
    AbsT     = 1e-11;
    stepsize = 0.1;
    tspan    = 0:stepsize:150000;
    options  = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);

    % ===============================
    % === RC2P – nonlinear model ===    --> for model 1, 3a and 3b
    % ===============================
    disp("----> R–C2–P normal") % normal and reversed: same parameters, once starting from low d values, once from high
    run_bifurcation_RC2P(parameters, d2_values, ...
        parameters.a2, parameters.h2, ...
        'normal', options, tspan);
    disp("----> R–C2–P reversed")
    run_bifurcation_RC2P(parameters, d2_values, ...
        parameters.a2, parameters.h2, ...
        'reversed', options, tspan);

    % ===================================
    % === RC2P – linearized C2 model === --> for model 2a and 2b
    % ===================================
    disp("----> R–C2–P linearized")
    run_bifurcation_RC2P(parameters, d2_values, ...
        parameters.a2, parameters.h2, ...
        'linearized', options, tspan);
    disp("----> R–C2–P linearized + reversed")
    run_bifurcation_RC2P(parameters, d2_values, ...
        parameters.a2, parameters.h2, ...
        'linearized_reversed', options, tspan);

    % ===============================
    % === RC1P – nonlinear model ===  --> for all models
    % ===============================
    disp("----> R–C1–P normal")
    run_bifurcation_RC1P(parameters, d1_values, ...
        parameters.a1, parameters.h1, ...
        'normal', options, tspan);
    disp("----> R–C1–P reversed")
    run_bifurcation_RC1P(parameters, d1_values, ...
        parameters.a1, parameters.h1, ...
        'reversed',options, tspan);

end

function run_bifurcation_RC2P(parameters, d2_values, ...
                              a2, h2, subfolder, ... % subfolder could be normal, reversed, linearized or linearized_reversed
                              options, tspan)
    completed = load('completed_paths.mat');
    completed = completed.completed;
    completed = replace(completed, '/', '\');


    base_folder = sprintf('.\RC2P\a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
        a2, h2, parameters.aP, parameters.hP, parameters.dP);

    if ~isempty(subfolder)
        base_folder = fullfile(base_folder, subfolder);
    end

    if ~exist(base_folder,'dir')
        mkdir(base_folder);
    end

    initial_conditions = [0.1 0.1 0.1];

    if strcmp(subfolder,'reversed') || strcmp(subfolder,'linearized_reversed')
        index_values = parameters.resolution:-1:1;
    else 
        index_values = 1:parameters.resolution;
    end

    % sweep from max -> 0
    for idx = index_values

        d = d2_values(idx);

        % --- choose model ---
        if strcmp(subfolder,'linearized') || strcmp(subfolder,'linearized_reversed')
            a = (1 - d*h2) * a2;
            h = 0;
        else
            a = a2;
            h = h2;
        end

        filename = fullfile(base_folder, ...
            sprintf('d2_idx_%d_of_%d.mat', idx, parameters.resolution));

        % check if the file has already been downloaded (and deleted from
        % the path)
        if any(completed == filename)
            disp('in completed')
            continue
        end

        % check if the file already exists on the path
        if exist(filename,'file')
            try
                data = load(filename);
        
                if isfield(data,'x') && ~isempty(data.x) && size(data.x,1) >= 1 % check if x (the time series) is really present
                    initial_conditions = data.x(end,:) + 1e-15;
                    disp('on path')
                    continue   % skip computation
                else
                    disp('file corrupt, recompute')
                    delete(filename);  % invalid → delete and recompute
                end
        
            catch
                delete(filename);      % unreadable → delete and recompute
            end
        end
        
        % compute the timeseries
        short_params = parameters;
        short_params.d  = d;
        short_params.a  = a;
        short_params.h  = h;

        f = @(t,x) RCPEquations(t,x,short_params);

        [t,x] = ode23(f, tspan, initial_conditions, options);

        if any(x(:,1)<=0)
            RelT     = 1e-13;
            AbsT     = 1e-18;
            initial_conditions = [1 0.1 0.1];
            stepsize = 0.1;
            disp('trying second simulation')
            fprintf('a = %.2g, h = %.2g, d = %.3g, aP = %.2g, hP = %.2g, dP = %.2g\n', ...
                    short_params.a, short_params.h, short_params.d, short_params.aP, short_params.hP, short_params.dP)
            options  = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);
            [t,x] = ode23(f, tspan, initial_conditions, options);
            
            if any(x(:,1)<=0)
                disp('could not simulate accurately')
                disp('Parameters:')
                disp(short_params)
            end
        
        end
        
        initial_conditions = x(end,:) + 1e-15;

        if any(initial_conditions < 0)
            disp("negative density appearing, reset");
            initial_conditions = 0.1 * ones(size(initial_conditions));
        end
        if length(x) > 1000000
            x = x(1000000:end,:);
            t = t(1000000:end);

            save(filename, ...
                't', 'x','short_params');
        else
            warning = 'issue integrating'
            save(filename, ...
            't', 'x','short_params','warning');
        end

    end
end

function run_bifurcation_RC1P(parameters, d1_values, ...
                              a1, h1, mode, options, tspan) % mode: normal or reversed

    completed = load('completed_paths.mat');
    completed = completed.completed;
    completed = replace(completed, '/', '\');


    base_folder = sprintf('.\RC1P\a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f/%s', ...
        a1, h1, parameters.aP, parameters.hP, parameters.dP, mode);

    if ~exist(base_folder,'dir')
        mkdir(base_folder);
    end

    initial_conditions = [0.1 0.1 0.1];

    if strcmp(mode,'normal')
        indexvalues = 1:parameters.resolution;
    elseif strcmp(mode,'reversed')
        indexvalues = parameters.resolution:-1:1;
    end

    for idx = indexvalues

        d = d1_values(idx);

        short_params = parameters;
        short_params.d = d;
        short_params.a = a1;
        short_params.h = h1;

        filename = fullfile(base_folder, ...
            sprintf('d1_idx_%d_of_%d.mat', idx, parameters.resolution));
        
        % check if the file has already been downloaded (and deleted from
        % the path)
        if any(completed == filename)
            disp('in completed')
            continue
        end

        if exist(filename,'file')
            try
                data = load(filename);
        
                if isfield(data,'x') && ~isempty(data.x) && size(data.x,1) >= 1
                    initial_conditions = data.x(end,:) + 1e-15;
                    disp('on path')
                    continue   % valid → skip computation
                else
                    disp('file corrupted, recompute')
                    delete(filename);  % invalid → delete and recompute
                end
        
            catch
                delete(filename);      % unreadable → delete and recompute
            end
        end

        f = @(t,x) RCPEquations(t,x,short_params);

        [t,x] = ode23(f, tspan, initial_conditions, options);
        
        if any(x(:,1)<=0)
            RelT     = 1e-13;
            AbsT     = 1e-18;
            initial_conditions = [1 0.1 0.1];
            stepsize = 0.1;
            disp('trying second simulation')
            fprintf('a = %.2g, h = %.2g, d = %.3g, aP = %.2g, hP = %.2g, dP = %.2g\n', ...
                    a, h, d, short_params.aP, short_params.hP, short_params.dP)
            options  = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);
            [t,x] = ode23(f, tspan, initial_conditions, options);
            
            if any(x(:,1)<=0)
                disp('could not simulate accurately')
                disp('Parameters:')
                disp(short_params)
            end
        
        end
        
        initial_conditions = x(end,:) + 1e-15;

        if any(initial_conditions < 0)
            disp("negative density appearing, reset");
            initial_conditions = 0.1 * ones(size(initial_conditions));
        end
        
        if length(x) > 1000000
            x = x(1000000:end,:);
            t = t(1000000:end);

            save(filename, ...
                't', 'x','short_params');
        else
            warning = 'issue integrating'
            save(filename, ...
            't', 'x','short_params','warning');
        end
       
    end
end


function dxdt = RCPEquations(t,x,params)
    
    dxdt = zeros(size(x));
    % params = {a2, aP, h2, hP, d2, dP};
    a = params.a;
    h = params.h;
    d = params.d;
    aP = params.aP;
    hP = params.hP;
    dP = params.dP;
    
    %------- State variables ------------------------------------------------;
    
    R = x(1);
    C = x(2);
    P = x(3);
    
    % ------ Rates of change -----------------------------------------------;
    
    N = 1-R;
    dxdt(1) = (N-(a*C/(1+h*a*R)))*R; 
    dxdt(2) = (a*R/(1+h*a*R)-d-aP*P/(1+hP*aP*C))*C;
    dxdt(3) = ((aP*C/(1+hP*aP*C))-dP)*P;  

end

function dxdt = RC1C2Equations(t,x,params)
    
    dxdt = zeros(size(x));
    % params = {a1, a2, h2, d1, d2};
    a1 = params.a1;
    a2 = params.a2;
    h2 = params.h2;
    d1 = params.d1;
    d2 = params.d2;
    
    %------- State variables ------------------------------------------------;
    
    R = x(1);
    C1 = x(2);
    C2 = x(3);
    
    % ------ Rates of change -----------------------------------------------;
    
    N = 1-R;
    dxdt(1) = (N-a1*C1-a2*C2/(1+h2*a2*R))*R; 
    dxdt(2) = (a1*R-d1)*C1;
    dxdt(3) = (a2*R/(1+h2*a2*R)-d2)*C2; 

end


function rnl_simulation(parameters)

    completed = load('completed_paths.mat');
    completed = completed.completed;
    completed = replace(completed, '/', '\');


    % get the C1-C2 coex matrix and based on that only simulate when there is
    % coexistence

    base_folder = sprintf('.\RC1C2\a1=%.2f_a2=%.2f_h2=%.2f', ...
        parameters.a1, parameters.a2, parameters.h2);
    
    if ~exist(base_folder,'dir')
        mkdir(base_folder);
    end

    % grids
    d1_values = get_grid(parameters.a1, parameters.h1, parameters.resolution);
    d2_values = get_grid(parameters.a2, parameters.h2, parameters.resolution);

    % solver settings 
    RelT     = 1e-9;
    AbsT     = 1e-11;
    stepsize = 0.1;
    tspan    = 0:stepsize:150000;
    options  = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);

    initial_conditions = [0.1 0.1 0.1];

    % compute coex matrix and only simulate for the coexistence cases
    coex_matrix = translate2matrix_basic(parameters.a1, parameters.a2, parameters.h2, parameters.resolution);

    short_params.a1 = parameters.a1;
    short_params.a2 = parameters.a2;
    short_params.h2 = parameters.h2;
    
    fprintf('coexistence matrix: %s', num2str(size(coex_matrix)))
    fprintf('d1 values: %s', num2str(size(d1_values)))
    fprintf('d2 values: %s', num2str(size(d1_values)))
        

    for d1_idx = 1:parameters.resolution
        for d2_idx = 1:parameters.resolution

            if coex_matrix(d1_idx,d2_idx) == 3 % coexistence
                                
                short_params.d1 = d1_values(d1_idx);
                short_params.d2 = d2_values(d2_idx);
                
                filename = fullfile(base_folder, ...
                    sprintf('d1_idx_%d_of_%d_d2_idx_%d_of_%d.mat', d1_idx, parameters.resolution, d2_idx, parameters.resolution));
                
                % check if the file has already been downloaded (and deleted from
                % the path)
                if any(completed == filename)
                    continue
                end
                % check if the file already exists
                if exist(filename,'file')
                    try
                        data = load(filename);
                
                        if isfield(data,'x') && ~isempty(data.x) && size(data.x,1) >= 1
                            % no initial conditions used here
                            continue   % valid → skip computation
                        else
                            delete(filename);  % invalid → delete and recompute
                        end
                
                    catch
                        delete(filename);      % unreadable → delete and recompute
                    end
                end
            
                f = @(t,x) RC1C2Equations(t,x,short_params);
            
                [t,x] = ode23(f, tspan, initial_conditions, options);
                
                if any(x(:,1)==0) % R becomes 0
                    RelT     = 1e-13;
                    AbsT     = 1e-18;
                    stepsize = 0.1;
                    tspan    = 0:stepsize:150000;
                    initial_conditions = [1 0.1 0.1];
                    disp('trying second simulation')
                    fprintf('a = %.2g, h = %.2g, d = %.3g, aP = %.2g, hP = %.2g, dP = %.2g\n', ...
                            a, h, d, short_params.aP, short_params.hP, short_params.dP)
                    options  = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);
                    [t,x] = ode23s(f, tspan, initial_conditions, options); % should be better when stiff? 
                    
                    if any(x(:,1)<=0)
                        disp('could not simulate accurately')
                        disp('Parameters:')
                        disp(short_params)
                    end
                
                end

                % not implemented as a bifurcation since the coexistence
                % patch is not so simple to traverse
                            
                x = x(1000000:end,:);
                t = t(1000000:end);
                
                save(filename, ...
                    't', 'x','short_params');
            end
        end
    end

end
