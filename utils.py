import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dill as pickle
from mpl_toolkits.mplot3d import Axes3D

sns.set()

def plot(Q, actions):

    pRange = list(range(1 ,22))
    dRange = list(range(1, 11))
    vStar = list()
    for p in pRange:
        for d in dRange:
            vStar.append([p, d, np.max(Q[p, d, :])])

    df = pd.DataFrame(vStar, columns= ['player', 'dealer', 'value'])

    fig = plt.figure()
    
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(30, 45)
    
    plt.show()

    
def plotMseLamdas(data, lamdas):
    df = pd.DataFrame(data, columns=['MSE'])
    df['lambda'] = lamdas

    sns.pointplot(x=df['lambda'], y=df['MSE'])
    plt.title("Mean Squared Error per Lambda")
    plt.show()