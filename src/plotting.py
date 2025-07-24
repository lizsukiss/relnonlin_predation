
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from utils import check_params


def simple_d1d2(coexistence_matrix, maxd1, maxd2, title, ax=None):
    """
    Plot the coexistence region on the given axes, or create a new figure if ax is None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
    ax.set_title(title)
    ax.set_xlim([0, maxd1])
    ax.set_ylim([0, maxd2])
    ax.set_xlabel(r'death rate of $C_1$ ($d_1$)',fontsize=10)
    ax.set_ylabel(r'death rate of $C_2$ ($d_2$)',fontsize=10)
    im = ax.imshow(-np.transpose(coexistence_matrix), origin='lower', cmap='gray', 
                   extent=[0, maxd1, 0, maxd2], aspect='auto')
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

def plot_coexistence_subplot(ax, coexistence_matrix, params, title):
    
    check_params(params, ['a1', 'a2', 'h2'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    
    maxd1 = a1
    maxd2 = a2

    simple_d1d2(coexistence_matrix, maxd1, maxd2, title, ax=ax)
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

    check_params(params, ['a1', 'a2', 'h2', 'aP', 'dP'])

    a1 = params['a1']
    a2 = params['a2']
    h2 = params['h2']
    aP = params['aP']
    dP = params['dP']

    filename = f'matrices_{a1}_{a2}_{aP}_{h2}_{dP}.npz'
    
    data = np.load(filename)
    coexistence_lin_sat= data['coexistence_lin_sat']
    coexistence_sat_sat= data['coexistence_sat_sat']
    coexistence_lin_lin_pred= data['coexistence_lin_lin_pred']
    coexistence_lin_sat_pred= data['coexistence_lin_sat_pred']
    coexistence_sat_sat_pred= data['coexistence_sat_sat_pred']

    fig, axs = plt.subplots(2, 3, figsize=(9, 3))
    plot_coexistence_subplot(axs[0, 1], coexistence_lin_sat, params, 'Linear–saturating')
    plot_coexistence_subplot(axs[0, 2], coexistence_sat_sat, params, 'Saturating–saturating')
    plot_coexistence_subplot(axs[1, 0], coexistence_lin_lin_pred, params, 'Linear–linear + predation')
    plot_coexistence_subplot(axs[1, 1], coexistence_lin_sat_pred, params, 'Linear–saturating + predation')
    plot_coexistence_subplot(axs[1, 2], coexistence_sat_sat_pred, params, 'Saturating–saturating + predation')

    add_arrows(fig, axs)
    
    # Add params annotation to the figure
    param_text = "\n".join([f"{k} = {v}" for k, v in params.items()])
    fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.show()

    return fig, axs
