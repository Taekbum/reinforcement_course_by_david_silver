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

    ax.view_init(30, -45)
    
    
    plt.show()
    # plt.savefig('valfunc_table_mc', dpi=fig.dpi)


def plotMseLamdas(data, lamdas):
    df = pd.DataFrame(data, columns=['MSE'])
    df['lambda'] = lamdas

    # fig = plt.figure()

    sns.pointplot(x=df['lambda'], y=df['MSE'])
    plt.title("Mean Squared Error per Lambda")
    plt.show()
    # plt.savefig('td_mse_lambda', dpi=fig.dpi)
    # plt.savefig('lfa_mse_lambda', dpi=fig.dpi)


def plotMseEpisodesLambdas(arr):
    
    m,n = arr.shape
    I,J = np.ogrid[:m,:n]
    out = np.empty((m,n,3), dtype=arr.dtype)
    out[...,0] = I
    out[...,1] = J
    out[...,2] = arr
    out.shape = (-1,3)

    df = pd.DataFrame(out, columns=['lambda', 'Episode', 'MSE'])
    df['lambda'] = df['lambda'] / 10
    #df = df.loc[df.index % 100 == 0]
    g = sns.FacetGrid(df, hue="lambda", size=8, legend_out=True)
    #g.map(plt.scatter, "episode", "mse")
    g = g.map(plt.plot, "Episode", "MSE").add_legend()

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Mean Squared Error per Episode')

    plt.show()