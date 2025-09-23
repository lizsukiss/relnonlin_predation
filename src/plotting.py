
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import integrate as integ
from functools import partial
from utils import check_params, make_filename
from ode import full_system

def simple_d1d2(coexistence_matrix, params, title, ax=None):
    """
    Plot the coexistence region on the given axes, or create a new figure if ax is None.
    """
    a1 = params['a1']
    h1 = params.get('h1', 0)  # Default to 0 if not provided
    a2 = params['a2']
    h2 = params['h2']
    maxd1 = a1/(1+h1*a1)
    maxd2 = a2/(1+h2*a2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
    ax.set_title(title)
    ax.set_xlim([0, maxd1])
    ax.set_ylim([0, maxd2])
    ax.set_xlabel(r'death rate of $C_1$ ($d_1$)',fontsize=10)
    ax.set_ylabel(r'death rate of $C_2$ ($d_2$)',fontsize=10)
    custom_cmap = ListedColormap(['white', '#0A81D1', '#424242'])  # white, blue, dark gray
    im = ax.imshow(np.transpose(coexistence_matrix), origin='lower', cmap=custom_cmap,
                   extent=[0, maxd1, 0, maxd2], aspect='auto', vmin=0, vmax=2)
    legend_labels = ['No coexistence', 'Coexistence at a fixed point', 'Coexistence at a limit cycle/dynamic equilibrium']
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(legend_labels)
    cbar.ax.tick_params(labelsize=8)

    return ax, im

def draw_lines(params,ax):
    
    check_params(params, ['a1', 'a2', 'h2'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    
    maxd1 = a1

    axis_d1 = np.arange(0,maxd1,0.01)
    invasived2  = a2*(axis_d1/a1)/(1+h2*a2*axis_d1/a1)
    exclusiond2 = a2/a1 * axis_d1/(1+h2*a2) 
    hopfbifurcationd2 = (a2*h2-1)/(h2*(a2*h2+1))*np.ones(len(axis_d1))

    ax.plot(axis_d1,invasived2,'-',label='invasion of C_2',color=[0.5, 0.5, 0.5],linewidth=2)    
    ax.plot(axis_d1,exclusiond2,'--',label='invasion of C_2',color=[0.5, 0.5, 0.5],linewidth=2)  
    ax.plot(axis_d1,hopfbifurcationd2,'-.',label='invasion of C_2',color=[0.5, 0.5, 0.5],linewidth=2)    
    '''
    RS = 1 - a1*(dP/aP)
    dC1per = (1-(dP/aP)*a1)*a1 
    d1x = np.arange(0,maxd1,0.01)#np.arange(0, dC1per, 0.01)
    d2y = a2*RS/(1+h2*a2*RS)-a1*RS+d1x
    ax.plot(d1x, d2y, '-r', linewidth=2)

    p = (1-h2*a2)/(h2*a2)
    q = (a2*(dP/aP)-1)/(h2*a2)
    RS = -p/2 + ((p/2)**2 - q)**(1/2)
    
    dC2per = a2*RS/(1+h2*a2*RS)
    
    endPoint = dC2per-(a2*RS/(1+h2*a2*RS)-a1*(1+h2*a2)*RS/(1+h2*a2*RS))
    if np.iscomplex(endPoint):
        endPoint = 1
    d1x = np.arange(0,maxd1,0.01);#np.arange(0,endPoint,0.01);
    C1boundary = -a1*RS+a2*RS/(1+h2*a2*RS)+d1x  #a2*RS/(1+h2*a2*RS)-a1*(1+h2*a2*Rmax)*RS/(1+h2*a2*RS) + d1x;
    ax.plot(d1x,C1boundary,'--r',linewidth=2)
    '''

def coexistence_plot_with_lines(coexistence_matrix, params, title, ax=None):
    
    simple_d1d2(coexistence_matrix, params, title, ax=ax)
    draw_lines(params, ax=ax)

def add_arrows(fig, axs, color="green"):
    """
    Add arrows between subplots in a 2x3 grid.
    fig: matplotlib Figure object
    axs: array of Axes objects (shape [2,3])
    color: arrow color
    """
    arrows = [
        # (xyA, coordsA, xyB, coordsB, arrowstyle)
        ((-0.5, .5), axs[0,1], (1.125, .5), axs[0,0], "->"),
        ((-0.5, .5), axs[0,2], (1.125, .5), axs[0,1], "<-"),
        ((.5, 1.25), axs[1,0], (.5, -.25), axs[0,0], "<-"),
        ((.5, 1.25), axs[1,1], (.5, -.25), axs[0,1], "<-"),
        ((.5, 1.25), axs[1,2], (.5, -0.25), axs[0,2], "<-"),
        ((-.5, 0.5), axs[1,1], (1.125, 0.5), axs[1,0], "->"),
        ((-.5, 0.5), axs[1,2], (1.125, 0.5), axs[1,1], "<-"),
    ]
    for xyA, axA, xyB, axB, arrowstyle in arrows:
        con = ConnectionPatch(
            xyA=xyA, coordsA=axA.transAxes,
            xyB=xyB, coordsB=axB.transAxes,
            arrowstyle=arrowstyle, color=color,
            transform=fig.transFigure
        )
        fig.add_artist(con)

def summary_plot(params):

    check_params(params, ['a1', 'a2', 'h2', 'aP', 'dP', 'resolution'])

    filename =  make_filename('results/matrices/matrices',params)
    
    data = np.load(filename)
    coexistence_lin_sat= data['coexistence_lin_sat']
    coexistence_sat_sat= data['coexistence_sat_sat']
    coexistence_lin_lin_pred= data['coexistence_lin_lin_pred']
    coexistence_lin_sat_pred= data['coexistence_lin_sat_pred']
    coexistence_sat_sat_pred= data['coexistence_sat_sat_pred']

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)  # more whitespace between subplots
    coexistence_plot_with_lines(axs[0, 1], coexistence_lin_sat, params, 'Linear–saturating') # own function implemented for something else
    coexistence_plot_with_lines(axs[0, 2], coexistence_sat_sat, params, 'Saturating–saturating')
    coexistence_plot_with_lines(axs[1, 0], coexistence_lin_lin_pred, params, 'Linear–linear + predation')
    coexistence_plot_with_lines(axs[1, 1], coexistence_lin_sat_pred, params, 'Linear–saturating + predation')
    coexistence_plot_with_lines(axs[1, 2], coexistence_sat_sat_pred, params, 'Saturating–saturating + predation')

    #add_arrows(fig, axs)
    
    # Add params annotation to the figure
    param_text = "\n".join([f"{k} = {v}" for k, v in params.items()])
    fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    return fig, axs

def summary_dynamics_plots(params): 

    check_params(params, ['a1', 'a2', 'h2', 'aP', 'd1', 'd2', 'dP'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    aP = params['aP']
    d1 = params['d1']
    d2 = params['d2']
    dP = params['dP']

    # It would be difficult to reload the plots of the population dynamics because 
    # (1) they were not necessarily simulated in the first place 
    # (2) have been probably moved permanently to the external data storage
    
    tend  = 10000
    tstep = 0.1
    time_array  = np.arange(0,tend,tstep) # time for simulation

    # LinSat
    simulation_params_lin_sat = {'a1':a1, 'a2':a2, 'aP':0, 'h1':0, 'h2':h2, 'hP':0, 'd1':d1, 'd2':d2, 'dP':0}
    initial_density_lin_sat = [0.01, 0.01, 0.01, 0]

    system_lin_sat = lambda density, time: full_system(density, time, simulation_params_lin_sat)
    dynamics_lin_sat  = integ.odeint(system_lin_sat, initial_density_lin_sat, time_array, rtol = 10**(-14), atol = 10**(-12))

    # LinSatPred
    simulation_params_lin_sat_pred = {'a1':a1, 'a2':a2, 'aP':aP, 'h1':0, 'h2':h2, 'hP':0, 'd1':d1, 'd2':d2, 'dP':dP}
    initial_density_lin_sat_pred = [0.01, 0.01, 0.01, 0.01]
    
    system_lin_sat_pred = lambda density, time: full_system(density, time, simulation_params_lin_sat_pred)
    dynamics_lin_sat_pred  = integ.odeint(system_lin_sat_pred, initial_density_lin_sat_pred, time_array, rtol = 10**(-14), atol = 10**(-12))

    # SatSat
    gamma = a1/a2 + h2*d1
    sat_a = gamma*a2
    sat_h = h2/gamma
    sat_d = d1
    simulation_params_sat_sat = {'a1':sat_a, 'a2':a2, 'aP':0, 'h1':sat_h, 'h2':h2, 'hP':0, 'd1':sat_d, 'd2':d2, 'dP':0}
    initial_density_sat_sat = [0.01, 0.01, 0.01, 0]

    system_sat_sat = lambda density, time: full_system(density, time, simulation_params_sat_sat)
    dynamics_sat_sat  = integ.odeint(system_sat_sat, initial_density_sat_sat, time_array, rtol = 10**(-14), atol = 10**(-12))

    # SatSatPred
    simulation_params_sat_sat_pred = {'a1':sat_a, 'a2':a2, 'aP':aP, 'h1':sat_h, 'h2':h2, 'hP':0, 'd1':sat_d, 'd2':d2, 'dP':dP}
    initial_density_sat_sat_pred = [0.01, 0.01, 0.01, 0.01]

    system_sat_sat_pred = lambda density, time: full_system(density, time, simulation_params_sat_sat_pred)
    dynamics_sat_sat_pred  = integ.odeint(system_sat_sat_pred, initial_density_sat_sat_pred, time_array, rtol = 10**(-14), atol = 10**(-12))

    # LinLinPred
    lin_a = (1-d2*h2)*a2
    short_params_lin_lin_pred = {'a1':a1, 'a2':lin_a, 'aP':aP, 'h1':0, 'h2':0, 'hP':0, 'd1':d1, 'd2':d2, 'dP':dP}
    initial_density_lin_lin_pred = [0.01, 0.01, 0.01, 0.01]

    system_lin_lin_pred = lambda density, time: full_system(density, time, short_params_lin_lin_pred)
    dynamics_lin_lin_pred  = integ.odeint(system_lin_lin_pred, initial_density_lin_lin_pred, time_array, rtol = 10**(-14), atol = 10**(-12))

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)  # more whitespace between subplots    axs[0, 1].plot(time_array, dynamics_lin_sat)
    axs[0, 1].set_title("Linear and saturating")
    axs[0, 2].plot(time_array, dynamics_sat_sat)
    axs[0, 2].set_title("Both saturating")
    axs[1, 0].plot(time_array, dynamics_lin_lin_pred)
    axs[1, 0].set_title("Both linear + predator")
    axs[1, 1].plot(time_array, dynamics_lin_sat_pred)
    axs[1, 1].set_title("Linear and saturating")
    axs[1, 2].plot(time_array, dynamics_sat_sat_pred)
    axs[1, 2].set_title("Both saturating + predator")
    
    #add_arrows(fig, axs)
    
    # Add params annotation to the figure
    #param_text = "\n".join([f"{k} = {v}" for k, v in params.items()])
    #fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    return fig, axs
