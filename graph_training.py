#!/usr/bin/env python
"""make_data.py: prepare data for training"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import sqlite3

def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sqldb")
    parser.add_argument("dest")
    parser.add_argument("--lib", default = "bokeh", choices = ["bokeh", "plotly"])
    args = parser.parse_args()
    
    db = sqlite3.connect(args.sqldb)
    c = db.cursor()
    
    c.execute("SELECT loss, bleu, dev_loss, dev_bleu FROM exp_data")
    all = [[], [], [], []]
    for line in c.fetchall():
        print line
#         _, _, _, test_loss, test_bleu, dev_loss, dev_bleu = line
        test_loss, test_bleu, dev_loss, dev_bleu = line
        all[0].append(test_loss)
        all[1].append(test_bleu)
        all[2].append(dev_loss)
        all[3].append(dev_bleu)
        
    if args.lib == "bokeh":
        import bokeh.charts
        bokeh.charts.output_file(args.dest)
        p = bokeh.charts.Line(all)
        bokeh.charts.show(p)
    else:
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
        trace0 = go.Scatter(
#             x = random_x,
            y = all[0],
            mode = 'lines',
            name = 'test_loss',
                line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4)
        )
        trace1 = go.Scatter(
#             x = random_x,
            y = all[1],
            mode = 'markers',
            name = 'test_bleu',
            yaxis='y2',
                line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4)
        )
        trace2 = go.Scatter(
#             x = random_x,
            y = all[2],
            mode = 'lines',
            name = 'dev_loss',
                line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,)
        )
        trace3 = go.Scatter(
#             x = random_x,
            y = all[3],
            mode = 'markers',
            name = 'dev_bleu',
            yaxis='y2',
                line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,)
        )
        
        layout = go.Layout(
            title='Double Y Axis Example',
            yaxis=dict(
                title='loss'
            ),
            yaxis2=dict(
                title='bleu',
                overlaying='y',
                side='right'
            )
        )
        
        data = [trace0, trace1, trace2, trace3]
        fig = go.Figure(data=data, layout=layout)
        # Plot and embed in ipython notebook!
        plotly.offline.plot(fig, filename = args.dest, auto_open = False)
    
if __name__ == '__main__':
    commandline()