
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

    Plot the coexistence region from the provided matrix on the given axes, or create a new figure if ax is None.
    
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
    """
    
    Draw lines for the original relative nonlinearity model on the given axis (invasion/exclusion/Hopf bifurcation)

    """

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
    """
    
    Create plot of coexistence area and add lines
    
    """

    simple_d1d2(coexistence_matrix, params, title, ax=ax)
    draw_lines(params, ax=ax)