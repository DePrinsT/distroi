import matplotlib.pyplot as plt


def set_matplotlib_params():
    # setting some matplotlib parameters
    plt.rc('font', size=9)  # controls default text sizes
    plt.rc('axes', titlesize=11)  # fontsize of the axes title
    plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
    plt.rc('legend', fontsize=8)  # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title
    return
