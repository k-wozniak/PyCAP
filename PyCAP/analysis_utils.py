"""
Contains functions used to analysise obtained distributions, such as post convolution analysis 
"""

import numpy as np
import matplotlib.pyplot as plt

def post_conv_analysis(signals, qs, distribution, plot_pairs = [(0, 1)], show_plot = True):
    # Signlas - all signals used to generate distribution
    # qs - Q matrices used to generate the distribution
    # distribution - obtained diss
    # Pairs of signals to plot
    # show_plot - if false will show only SSD between the signals
    
    for p1, p2 in plot_pairs:
        s1 = signals[p1]
        s2 = signals[p2]

        ct1 = np.convolve(s1, (qs[p2+1] - qs[p2])@distribution)
        ct2 = np.convolve(s2, (qs[p1+1] - qs[p1])@distribution)

        ssd = np.sum((ct1 - ct2)**2) # Fix it 
        print("SSD between signals: (" + str(p1) + ", " + str(p2) + ") is " + str(ssd))

        if show_plot:
            plt.figure()
            plt.plot(ct1)
            plt.plot(ct2)
            plt.show()