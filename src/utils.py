import numpy as np

def get_equilibriums(a1, a2, aP, h2, d1, d2, dP):

    G_p = (a1 - a2 - (d1 - d2) * h2 * a2) / (a1 * h2 * a2)
    G_q = -(d1 - d2) / (a1 * h2 * a2)
    G1_Rstar = -G_p / 2 + np.sqrt(G_p**2 / 4 - G_q) # "positive" root
    G2_Rstar = -G_p / 2 - np.sqrt(G_p**2 / 4 - G_q) # "negative" root
    G1_C2 = (G1_Rstar + a1 * dP / aP - 1) / (a1 - a2 / (1 + h2 * a2 * G1_Rstar))
    G2_C2 = (G2_Rstar + a1 * dP / aP - 1) / (a1 - a2 / (1 + h2 * a2 * G2_Rstar))
    G1_C1 = dP / aP - G1_C2
    G2_C1 = dP / aP - G2_C2
    G1_P = (a1 * G1_Rstar - d1) / aP
    G2_P = (a1 * G2_Rstar - d1) / aP
    x0_G1 = np.array([G1_Rstar, G1_C1, G1_C2, G1_P]) # "positive" root
    x0_G2 = np.array([G2_Rstar, G2_C1, G2_C2, G2_P]) # "negative" root
    return x0_G1, x0_G2

def check_params(params, required):
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing parameters: {', '.join(missing)}")