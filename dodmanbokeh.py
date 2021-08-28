# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:42:32 2021

@author: AfterthoughtC
"""

import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, LabelSet


flip_meaning = pd.read_csv('flip meaning.csv')
pathds = pd.read_csv('pathds.csv')
pointds = pd.read_csv('pointds.csv')


flip_dict = {}
for i in range(len(flip_meaning)):
    flip_dict['Node_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Node']
    flip_dict['Termini_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Termini']
    flip_dict['Intersection_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Intersection']


startlists = [[],[],[],[]]
otherlists = [[],[],[],[]]
intersectlists = [[],[],[],[]]
pointdict = {}

for p in range(len(pointds)):
    pointdict[pointds.loc[p,'Point Index']] = {'X':pointds.loc[p,'X'],
                                               'Y':pointds.loc[p,'Y'],
                                               'Point Type':pointds.loc[p,'Point Type']}
    lockey = pointds.loc[p,'Point Type']
    flips = pointds.loc[p,'Flips']
    
    if lockey == 'Start':
        startlists[0].append(pointds.loc[p,'X'])
        startlists[1].append(pointds.loc[p,'Y'])
        startlists[2].append("Start")
        startlists[3].append(pointds.loc[p,'Point Index'])
        pointdict[pointds.loc[p,'Point Index']]['Location'] = 'Start'
    else:
        pointdict[pointds.loc[p,'Point Index']]['Location'] = flip_dict[lockey + "_" + flips]
        if lockey == 'Intersection':
            intersectlists[0].append(pointds.loc[p,'X'])
            intersectlists[1].append(pointds.loc[p,'Y'])
            intersectlists[2].append(pointdict[pointds.loc[p,'Point Index']]['Location'])
            intersectlists[3].append(pointds.loc[p,'Point Index'])
        else:
            otherlists[0].append(pointds.loc[p,'X'])
            otherlists[1].append(pointds.loc[p,'Y'])
            otherlists[2].append(pointdict[pointds.loc[p,'Point Index']]['Location'])
            otherlists[3].append(pointds.loc[p,'Point Index'])
        

linelists = [[],[]]
for l in range(len(pathds)):
    p1 = pathds.loc[l,'Point Index 1']
    p2 = pathds.loc[l,'Point Index 2']
    linelists[0].append([pointdict[p1]['X'],pointdict[p2]['X']])
    linelists[1].append([pointdict[p1]['Y'],pointdict[p2]['Y']])


output_file('dodman.html',mode='inline')

point_tooltip = [
        ("Mouse Coord","($x, $y)"),
        ("Point","@point_no"),
        ("Point Coord","(@x, @y)"),
        ("Contains","@desc")
    ]

point_hover = HoverTool(names=['node_n_termini_point','intersection_point','start_point'],
                        tooltips=point_tooltip)

p = figure(plot_width = 500,plot_height=500,tools=['pan', 'box_zoom', 'wheel_zoom', 'save',
                                 'reset', point_hover])

# nodes and termini
pointsource = ColumnDataSource(data = {
        'x':otherlists[0],
        'y':otherlists[1],
        'desc':otherlists[2],
        'point_no':otherlists[3]
    })
node_term_glyph = p.circle('x','y',size=5,color="blue",alpha=1,source=pointsource,name='node_n_termini_point')
p.text([a + 7 for a in otherlists[0]],
       [a + 7 for a in otherlists[1]],
       text=["%d" % q for q in otherlists[3]],
       text_baseline="middle",
       text_align="center")


# intersections
intsource = ColumnDataSource(data = {
        'x':intersectlists[0],
        'y':intersectlists[1],
        'desc':intersectlists[2],
        'point_no':intersectlists[3]
    
    })
intersection_glyph = p.circle('x','y',size=20,color="blue",fill_alpha=0,line_alpha=1,source=intsource,name='intersection_point')
p.text([a + 7 for a in intersectlists[0]],
       [a + 7 for a in intersectlists[1]],
       text=["%d" % q for q in intersectlists[3]],
       text_baseline="middle",
       text_align="center")


# start points
startsource = ColumnDataSource(data = {
        'x':startlists[0],
        'y':startlists[1],
        'desc':startlists[2],
        'point_no':startlists[3]
    
    })
start_glyph = p.x('x','y',size=20,color="blue",alpha=1,source=startsource,name='start_point')
p.text([a + 7 for a in startlists[0]],
       [a + 7 for a in startlists[1]],
       text=["%d" % q for q in startlists[3]],
       text_baseline="middle",
       text_align="center")


p.multi_line(linelists[0],linelists[1],color='black')
show(p)