from utils import check_params

# R-C system
def predator_prey(x,t,params):

    check_params(params, {'a', 'h', 'd'})

    a = params['a']
    h = params['h']
    d = params['d']


    R  = x[0]  # resource
    C = x[1]  # consumer
    
    Rdot = (1-R)*R - a*C*R/(1+a*h*R)
    Cdot = (a*R/(1+a*h*R) - d)*C
        
    return [Rdot,Cdot]

# R-C1+C2-P system
def full_system(x,t,params):
    check_params(params, ['a1', 'a2', 'aP', 'h1', 'h2', 'hP', 'd1', 'd2', 'dP'])

    a1 = params['a1']
    a2 = params['a2']
    aP = params['aP']
    h1 = params['h1']
    h2 = params['h2']
    hP = params['hP']
    d1 = params['d1']
    d2 = params['d2']
    dP = params['dP']

    R  = x[0]  # resource
    C1 = x[1]  # consumer 1
    C2 = x[2]  # consumer 2
    P  = x[3]  # predator    
    
    Rdot = (1-R)*R - a1*C1*R/(1+a1*h1*R) - a2*C2*R/(1+a2*h2*R)
    C1dot = (a1*R/(1+a1*h1*R) - d1 - aP*P/(1+aP*hP*(C1+C2)))*C1
    C2dot = (a2*R/(1+a2*h2*R) - d2 - aP*P/(1+aP*hP*(C1+C2)))*C2
    Pdot = (aP*(C1+C2)/(1+aP*hP*(C1+C2)) - dP)*P
        
    return [Rdot,C1dot,C2dot,Pdot]