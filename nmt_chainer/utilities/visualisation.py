#!/usr/bin/env python
"""visualisation.py: visualize attention model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.models.layouts import Column
from bokeh.models import HoverTool, ColumnDataSource

# adapted from http://bokeh.pydata.org/en/0.7.1/tutorial/solutions/gallery/les_mis.html

def make_alignment_figure(src, tgt, alignment, title="Attention Model", toolbar_location='right', plot_width=800, plot_height=800):
    
    alignment = alignment[:,::-1]
#     tgt = list(reversed(tgt))
    
    src = ["%i %s"%(num, x) for num, x in enumerate(src)]
    tgt = ["%s %i"%(x, num) for num, x in reversed(list(enumerate(tgt)))]
    
    xname = []
    yname = []
    color = []
    alpha = []
    
    count = [] 
    for i, n1 in enumerate(src):
        for j, n2 in enumerate(tgt):
            xname.append(n1)
            yname.append(n2)
            alpha.append(alignment[i,j])
            color.append('black')
            count.append(alignment[i,j])
    
#     for x, y, a in zip(xname, yname, alpha):
#         print x, y, a
    
    source = ColumnDataSource(
        data=dict(
            xname=xname,
            yname=yname,
            colors=color,
            alphas=alpha,
            count=count,
        )
    )
    
    # create a new figure
    p = figure(title=title,
               x_axis_location="above", tools="resize,hover", toolbar_location=toolbar_location,
               x_range=src, y_range=tgt,
               plot_width=plot_width, plot_height=plot_height)
    
    p.rect('xname', 'yname', 0.9, 0.9, source=source,
           color='colors', alpha='alphas', line_color=None)
    
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3
    
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ('tgt/src', '@yname, @xname'),
        ('attn', '@count'),
    ]
    
    return p

if __name__ == '__main__':
    src = ["le", "est", "un"]
    tgt = ["the", "one", "is", "about"]
    alignment = np.array(
                         [[0, 1, 2, 5],
                          [2, 4, 6, 7],
                          [9, 1, 3, 3]]
                         )/10.0
    print alignment
    p1 = make_alignment_figure(src, tgt, alignment)
    p2 = make_alignment_figure(src, tgt, alignment)
    p_all = Column(p1, p2)
    show(p)
