import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def bar_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # MAE plot
    fig.add_subplot(2, 2, 1)
    df.loc["mae"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mae"].sort_values().index], )
    plt.legend()
    plt.title("MAE Metric, lower is better")
    fig.add_subplot(2, 2, 2)
    df.loc["rmse"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["rmse"].sort_values().index], )
    plt.legend()
    plt.title("RMSE Metric, lower is better")
    fig.add_subplot(2, 2, 3)
    df.loc["mape"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mape"].sort_values().index], )
    plt.legend()
    plt.title("MAPE Metric, lower is better")
    fig.add_subplot(2, 2, 4)
    df.loc["r2"].sort_values(ascending=False).plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333")
            for x in df.loc["r2"].sort_values(ascending=False).index
        ],
    )
    plt.legend()
    plt.title("R2 Metric, higher is better")
    plt.tight_layout()
    plt.savefig("results/metrics.png")


def plot_prediction(X, Xf, Ys, method_list, styles, region_list, region_mask):
    """
    [___________][#####]
                        [___________][#####]
                                            [___________][#####]
    X -      (sections, seq_length, regions)
    Xf -     (sections, seq_length, regions)
    Ys -     (sections, methods, predict_len, regions)
    
    """
    # plt.close('all')
    # plt.figure(figsize=(20, 10))
    # print(X.shape, Xf.shape)
    # for Y in Ys:
    #     print(Y.shape)
    # print(region_mask)
    #
    sns.set(font_scale=1.5)
    dfs=[]
    idx = 0
    for i in range(len(X)):
        x = X[i]
        xf = Xf[i]
        y = Ys[i]
                
        x_n, lines_x = x.shape
        methods, y_n, lines_y = y.shape

        assert (lines_x == lines_y)

        d = dict()
        d['Days'] = []
        d['New cases'] = []
        d['Region'] = []
        d['Data type'] = []
        d['Preprocessing'] = []
        d['Size'] = []
        for c in range(lines_x):
            if region_mask[c] == 0:
                  continue
            
            d['Days'] += [idx + i for i in range(x_n)]
            d['New cases'] += [i for i in x[:, c]]
            d['Region'] += [region_list[c]] * x_n
            d['Data type'] += ['Train'] * x_n
            d['Size'] += [styles['X']['Size']]*x_n
            d['Preprocessing'] +=[styles['X']['Preprocessing']] *x_n
            
            d['Days'] += [idx + i for i in range(x_n)]
            d['New cases'] += [i for i in xf[:, c]]
            d['Region'] += [region_list[c]] * x_n
            d['Data type'] += ['Train'] * x_n
            d['Size'] += [styles['Xf']['Size']]*x_n
            d['Preprocessing'] +=[styles['Xf']['Preprocessing']] *x_n
            
            for m in range(methods):
                
                d['Region'] += [region_list[c]] * (y_n + 2)
                d['Data type'] += [method_list[m]] * (y_n + 2)
                d['Size'] += [styles[method_list[m]]['Size']]*(y_n + 2)
                preprocessing = styles[method_list[m]]['Preprocessing']
                d['Preprocessing'] +=[preprocessing] * (y_n + 2)
                
                d['Days'] += [idx+i for i in range(x_n-1, x_n+y_n+1)]
                if preprocessing=='Filtered':
                    d['New cases'] += [xf[-1, c]] + [i for i in y[m, :, c]]
                    if i+1 != X.shape[0]:
                        d['New cases'] += [Xf[i+1,0,c]]
                    else:
                        d['New cases'] += [y[m, -1, c]]
                else:
                    d['New cases'] += [x[-1, c]] + [i for i in y[m, :, c]]
                    if i+1 != X.shape[0]:
                        d['New cases'] += [X[i+1,0,c]]
                    else:
                        d['New cases'] += [y[m, -1, c]]
                
                
        idx += x_n+y_n

        df = pd.DataFrame(d)
        df.set_index('Days')
        dfs.append(df)
        if i==0:
            legend = 'brief'
        else:
            legend = False
        ax = sns.lineplot(data=df, x='Days', y='New cases', 
                          style='Preprocessing',
                          hue='Data type', size='Size', linewidth=3,
                          # estimator=lambda x: x if len(x)==1 else list(x)[1],
                          markers=True, dashes=True,
                          legend=legend)
        
        # if legend != False:
#             plt.legend(loc='upper left')
#             handles, labels = ax.get_legend_handles_labels()
#             a = 2
#             b = 6
#             ax.legend(handles=handles[:a+len(method_list)]+handles[b+len(method_list):], labels=labels[:a+len(method_list)]+labels[b+len(method_list):], loc='lower left')
    ax.set_xlabel("Days")
    ax.set_ylabel("Cases")
    # plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    return dfs