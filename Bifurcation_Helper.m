
function [R, C, P, last_density] = Bifurcation_Helper(short_params,initial_conditions)

    % short_params: {'a2', 'aP', 'h2', 'hP', 'd2', 'dP'}
    % configuration and parametrization
    
    RelT=0.000000001; 
    AbsT=0.00000000001; 
    stepsize=0.1; %0.1
    options = odeset('RelTol',RelT,'AbsTol',AbsT,'MaxStep',stepsize);
    f = @(t,x) RC2PEquations(t,x,short_params);

    [t,x] = ode23(f,0:stepsize:40000,initial_conditions,options);

    R = x(:,1);
    C = x(:,2);
    P = x(:,3);
    last_density = x(end,:);

end 