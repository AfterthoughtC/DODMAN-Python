# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:42:32 2021

@author: AfterthoughtC
"""

import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource


flip_meaning = pd.read_csv('flip meaning.csv')
pathds = pd.read_csv('pathds.csv')
pointds = pd.read_csv('pointds.csv')


flip_dict = {}
for i in range(len(flip_meaning)):
    flip_dict['Node_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Node']
    flip_dict['Termini_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Termini']
    flip_dict['Intersection_'+flip_meaning.loc[i,'Label']] = flip_meaning.loc[i,'Intersection']


startlists = [[],[],[]]
otherlists = [[],[],[]]
intersectlists = [[],[],[]]
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
        pointdict[pointds.loc[p,'Point Index']]['Location'] = 'Start'
    else:
        pointdict[pointds.loc[p,'Point Index']]['Location'] = flip_dict[lockey + "_" + flips]
        if lockey == 'Intersection':
            intersectlists[0].append(pointds.loc[p,'X'])
            intersectlists[1].append(pointds.loc[p,'Y'])
            intersectlists[2].append(pointdict[pointds.loc[p,'Point Index']]['Location'])
        else:
            otherlists[0].append(pointds.loc[p,'X'])
            otherlists[1].append(pointds.loc[p,'Y'])
            otherlists[2].append(pointdict[pointds.loc[p,'Point Index']]['Location'])
        

linelists = [[],[]]
for l in range(len(pathds)):
    p1 = pathds.loc[l,'Point Index 1']
    p2 = pathds.loc[l,'Point Index 2']
    linelists[0].append([pointdict[p1]['X'],pointdict[p2]['X']])
    linelists[1].append([pointdict[p1]['Y'],pointdict[p2]['Y']])


output_file('dodman.html',mode='inline')

TOOLTIPS = [
        ("Mouse Coord","($x, $y)"),
        ("Point Coord","(@x, @y)"),
        ("Contains","@desc")
    ]

p = figure(plot_width = 500,plot_height=500,tooltips=TOOLTIPS)
pointsource = ColumnDataSource(data = {
        'x':otherlists[0],
        'y':otherlists[1],
        'desc':otherlists[2]
    
    })
p.circle('x','y',size=20,color="blue",alpha=0.8,source=pointsource)
intsource = ColumnDataSource(data = {
        'x':intersectlists[0],
        'y':intersectlists[1],
        'desc':intersectlists[2]
    
    })
p.circle_x('x','y',size=20,color="green",alpha=0.25,source=intsource)
startsource = ColumnDataSource(data = {
        'x':startlists[0],
        'y':startlists[1],
        'desc':startlists[2]
    
    })
p.x('x','y',size=20,color="black",alpha=1,source=startsource)
p.multi_line(linelists[0],linelists[1],color='black')
show(p)