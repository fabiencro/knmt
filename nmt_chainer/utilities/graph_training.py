#!/usr/bin/env python
"""graph_training.py: visualize training process"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import sqlite3


def build_prefix_list(lst, op=max):
    res = [None] * len(lst)
    res[0] = lst[0]
    for k in xrange(1, len(lst)):
        res[k] = op(res[k - 1], lst[k])
    return res


def define_parser(parser):
    parser.add_argument("sqldb")
    parser.add_argument("dest")
#     parser.add_argument("--lib", default = "plotly", choices = ["bokeh", "plotly"])


def do_graph(args):
    generate_graph(args.sqldb, args.dest)


def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    define_parser(parser)
    args = parser.parse_args()
    do_graph(args)


def generate_graph(sqldb, dest, title="Training Evolution"):
    db = sqlite3.connect(sqldb)
    c = db.cursor()

    c.execute("SELECT date, bleu_info, iteration, loss, bleu, dev_loss, dev_bleu, avg_time, avg_training_loss FROM exp_data")
    date, bleu_info, iteration, test_loss, test_bleu, dev_loss, dev_bleu, avg_time, avg_training_loss = zip(
        *list(c.fetchall()))
#     all = [[], [], [], []]
#     for line in c.fetchall():
#         print line
# #         _, _, _, test_loss, test_bleu, dev_loss, dev_bleu = line
#         test_loss, test_bleu, dev_loss, dev_bleu, avg_training_loss = line
#         all[0].append(test_loss)
#         all[1].append(test_bleu)
#         all[2].append(dev_loss)
#         all[3].append(dev_bleu)
    if avg_training_loss[0] == 0:
        avg_training_loss = list(avg_training_loss)
        avg_training_loss[0] = None

    import plotly
    #         import plotly.plotly as py
    import plotly.graph_objs as go
    import numpy as np
    #         N = len(all[0])
    # #         random_x = np.linspace(0, , N)
    #         random_y0 = np.random.randn(N)+5
    #         random_y1 = np.random.randn(N)
    #         random_y2 = np.random.randn(N)-5
    #
    # Create traces
    text_trg = ["i:%i\n%s\nt:%r" % (i, d, t) for i, d, t in zip(iteration, date, avg_time)]
    text_bleu = []
    for bli in bleu_info:
        if bli is None:
            text_bleu.append("NO INFO")
        else:
            bli_splitted = bli.split()
            ng_info = bli_splitted[1:5]
            size_info = bli_splitted[-1]
            text_bleu.append("\n" + "\n".join(ng_info + [size_info]))

    trace0 = go.Scatter(
        #             x = random_x,
        y=test_loss,
        mode='lines',
        name='test_loss',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )

    trace0min = go.Scatter(
        #             x = random_x,
        y=build_prefix_list(test_loss, op=min),
        mode='lines',
        name='min_test_loss',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1, dash="dot")
    )

    trace1 = go.Scatter(
        #             x = random_x,
        y=test_bleu,
        mode='markers',
        name='test_bleu',
        yaxis='y2',
        text=text_bleu,

        marker=dict(
            size=4,
            color='rgba(205, 12, 24, .8)')


        #                 line = dict(
        #         color = ('rgb(205, 12, 24)'),
        #         width = 2)
    )

    trace1max = go.Scatter(
        y=build_prefix_list(test_bleu, op=max),
        mode='lines',
        name='max_test_bleu',
        yaxis='y2',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
            dash="dot")
    )

    trace2 = go.Scatter(
        y=dev_loss,
        mode='lines',
        name='dev_loss',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2,)
    )

    trace2min = go.Scatter(
        y=build_prefix_list(dev_loss, op=min),
        mode='lines',
        name='min_dev_loss',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=1, dash="dot")
    )

    trace3 = go.Scatter(
        y=dev_bleu,
        mode='markers',
        name='dev_bleu',
        yaxis='y2',
        marker=dict(
            size=4,
            color='rgba(22, 96, 167, .8)')
        #                 line = dict(
        #         color = ('rgb(22, 96, 167)'),
        #         width = 2,)
    )

    trace3max = go.Scatter(
        y=build_prefix_list(dev_bleu, op=max),
        mode='lines',
        name='max_dev_bleu',
        yaxis='y2',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=1, dash="dot")
    )

    trace4 = go.Scatter(
        #             x = random_x,
        y=avg_training_loss,
        mode='lines',
        text=text_trg,
        name='avg_training_loss',
        line=dict(
            color=('rgb(22, 220, 67)'),
            width=2,)
    )

    layout = go.Layout(
        title=title,
        yaxis=dict(
            title='loss',
            gridcolor='#bdbdbd'
        ),
        yaxis2=dict(
            title='bleu',
            overlaying='y',
            side='right'
        )
    )

    data = [trace0, trace1, trace2, trace3, trace4, trace0min, trace1max, trace2min, trace3max]
    #         data = [trace2, trace3, trace4, trace2min, trace3max]
    fig = go.Figure(data=data, layout=layout)
    # Plot and embed in ipython notebook!
    plotly.offline.plot(fig, filename=dest, auto_open=False)


if __name__ == '__main__':
    commandline()
