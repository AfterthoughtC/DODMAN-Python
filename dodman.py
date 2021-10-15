# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:12:42 2021

@author: AfterthoughtC
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# we start by determining the width and height of the map
width = 400 # width of the map
height = 400 # height of the map
startcoord = [(0.5,0.5)] # if we want to have multiple start coordinates
randcoord = False # Only relevant if there are multiple coordinates.
                # If True when the next startcoord will be randomly chosen
                # if False the next startcoord will go in order (ex. startcoord 1, 2, 3)
maptype = 'ellipse' # 2 types; circle / ellipse and square / rectangle
angle_no = 12 # if more than 0, will only generate angles whose size is of multiple 2pi/angle_no
size = 5 # to prevent points from being too close to each other, size sets a minimum diameter
            # this diameter is also used to 'round' path lengths and directions to points that fall
            # within the circle
zeropow = 11 # due to accuracy loss, any number that is less than 10^-zeropow will be converted straight to 0
iterations = 30 # how many times to flip your coin
#seed = None
seed = 2335 # the seed to use. If none the file will generate its own seed
flipres = ['H','T'] # the results for flipping the coin
maxflip = 3 # the maximum number of flips there can be
#flipres = ['1','2','3','4','5','6'] # the results for casting the dice
#maxflip = 2 # the maximum number of flips there can be



rng = np.random.default_rng(seed)
#rng = np.random.default_rng()


def minnumber(number,zeropow):
    if abs(number) < 10**-zeropow:
        return(0)
    return(number)


# function for generating the angle
def generate_angle(angle_no,rng = np.random.default_rng()):
    if angle_no == 0:
        return(rng.random()*2*np.pi)
    return(2*np.pi/angle_no*rng.integers(0,angle_no))


def generate_vector(angle):
    x = minnumber(np.cos(angle),zeropow)
    y = minnumber(np.sin(angle),zeropow)
    return(np.array([x,y]))


# the class for a square boundary
class SquareBoundary():
    def __init__(self,width,height):
        self.xmin = -width/2
        self.xmax = width/2
        self.ymin = -height/2
        self.ymax = height/2
    
    def intersect_disp(self,startpoint,vector):
        dx = self.max_disp(startpoint[0],
                           vector[0],
                           self.xmin,
                           self.xmax)
        dy = self.max_disp(startpoint[1],
                           vector[1],
                           self.ymin,
                           self.ymax)
        d = min(dx,dy)
        return(d)

    def max_disp(self,start,vect,minval,maxval):
        if vect == 0:
            return(float('+inf'))
        if vect > 0:
            return((maxval-start)/vect)
        if vect < 0:
            return((minval-start)/vect)


# the class for an ellipse
class EllipseBoundary():
    def __init__(self,width,height):
        self.a = width/2
        self.b = height/2
    
    def intersect_disp(self,startpoint,vector):
        xm = vector[0]
        ym = vector[1]
        xo = startpoint[0]
        yo = startpoint[1]
        negbcomp = xm*xo*self.b**2 + ym*yo*self.a**2
        sqrtcomp = 2*xm*xo*ym*yo + (self.b*xm)**2 + (self.a*ym)**2 - (xm*yo)**2 - (xo*ym)**2
        sqrtcomp = minnumber(sqrtcomp,zeropow)
        twoacomp = (self.b*xm)**2 + (self.a*ym)**2
        dp,dn = (-negbcomp+self.a*self.b*sqrtcomp**0.5)/twoacomp,(-negbcomp-self.a*self.b*sqrtcomp**0.5)/twoacomp
        dp = minnumber(dp,zeropow)
        dn = minnumber(dn,zeropow)
        print('dpdn',dp,dn)
        d = []
        if dp > 0:
            d.append(dp)
        if dn > 0:
            d.append(dn)
        if len(d) == 0:
            d = 0
        else:
            d = min(d)
        print(self.within_boundary(startpoint[0]+d*vector[0],startpoint[1]+d*vector[1]))
        return(d)
    
    def within_boundary(self,xo,yo):
        return((xo/self.a)**2+(yo/self.b)**2<=1)

# the default point class
class Point():
    def __init__(self,index,x,y,flips,size,pointtype):
        self.index = index
        self.x = minnumber(x,zeropow)
        self.y = minnumber(y,zeropow)
        self.flips = flips
        self.size = size
        self.pointtype = pointtype
        
    def coord(self):
        return(np.array([self.x,self.y]))

    def point_touch(self,P1,vector):
        P2 = P1 + vector
        P0 = self.coord()
        #vector = vector/((vector[0]**2+vector[1]**2)**0.5)
        top = (P2[0]-P1[0])*(P1[1]-P0[1])-(P1[0]-P0[0])*(P2[1]-P1[1])
        bottom = ((P2[0]-P1[0])**2+(P2[1]-P1[1])**2)**0.5
        sdist = abs(top)/bottom
        if sdist > self.size:
            return(False,None)
        touchpoint = P1+sum((P0-P1)*vector)*vector
        totouch = touchpoint - P1
        dist = 0
        N = 0
        for i in range(len(vector)):
            if vector[i] != 0:
                dist += totouch[i]/vector[i]
                N += 1
        dist = dist/N
        return(dist>=0,dist)


def check_lines_intersect(P1,P2,P3,P4):
    
    V1 = P2-P1
    D1 = ((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)**0.5
    xdy1 = None if V1[1]==0 else V1[0]/V1[1]
    ydx1 = None if V1[0]==0 else V1[1]/V1[0]
    V2 = P4-P3
    D2 = ((P3[0]-P4[0])**2 + (P3[1]-P4[1])**2)**0.5
    xdy2 = None if V2[1]==0 else V2[0]/V2[1]
    ydx2 = None if V2[0]==0 else V2[1]/V2[0]
    if xdy1 == xdy2 and ydx1 == ydx2:
        return(False,None)
    
    a1 = -V1[1]
    b1 = V1[0]
    k1 = a1*P2[0]+b1*P2[1]
    a2 = -V2[1]
    b2 = V2[0]
    k2 = a2*P3[0]+b2*P3[1]
    numerator = k1*a2-k2*a1
    denominator = b1*a2-b2*a1
    if abs(denominator) <= 10**-zeropow:
        return(False,None)
    ym = numerator/denominator
    if abs(a1) > 10**-zeropow:
        xm = (k1-b1*ym)/a1
    else:
        xm = (k2-b2*ym)/a2
    matchcoord = [xm,ym]
    d1list = []
    d2list = []
    for i in range(2):
        if V1[i] != 0:
            d1list.append((matchcoord[i]-P1[i])/V1[i])
            #d1list.append(D1-(P2[i]-matchcoord[i])/V1[i])
        if V2[i] != 0:
            d2list.append((matchcoord[i]-P3[i])/V2[i])
            #d2list.append(D2-(P4[i]-matchcoord[i])/V2[i])
    d1 = sum(d1list)/len(d1list)*D1
    d2 = sum(d2list)/len(d2list)*D2
    if d1 <= D1 and d1 > 0 and d2 <= D2 and d2 > 0:
        return(True,d1)
    return(False,d1)



pointlist = []
for i in range(len(startcoord)):
    pointlist.append(Point(i,(startcoord[i][0]-0.5)*width,(startcoord[i][1]-0.5)*height,'S',10,'Start'))
linelist = []
if maptype == 'square':
    bound = SquareBoundary(width,height)
elif maptype == 'ellipse':
    bound = EllipseBoundary(width,height)
if len(startcoord) == 1 or randcoord == False:
    current_start = 0
else:
    current_start = rng.integers(len(startcoord))

current = current_start

for i in range(iterations):
    print('iter',i+1)
    print('current',current)
    # generate the angle and vector
    angle = generate_angle(angle_no,rng)
    flip = flipres[rng.integers(len(flipres))]
    vector = generate_vector(angle)
    print('current point',pointlist[current].coord(),pointlist[current].flips)
    print('vector',vector)
    # create the maximum distance
    maxd = bound.intersect_disp(pointlist[current].coord(),vector)
    
    if abs(maxd) <= size:
        # if that displacement will take us off the paper / will not exit the
        # point 'collision shape', add a point
        print('The vector will not take us very far from the current node')
        if pointlist[current].flips == 'S':
            print('Start point will not be changed')
        elif len(pointlist[current].flips) >= maxflip:
            print('Current point already has '+str(maxflip)+' flips; returning to start')
            if len(startcoord) == 1:
                current_start = 0
            elif randcoord:
                current_start = rng.integers(len(startcoord))
            else:
                current_start = (current_start + 1) % len(startcoord)
            current = current_start
        else:
            pointlist[current].flips += flip
            print('updated point',pointlist[current].coord(),pointlist[current].flips)
    else:
        closestpoint = None
        closestdist = float('+inf')
        # iterate every pre-existing path to see which has the closest intersection
        for l in range(len(linelist)):
            if current in linelist[l]:
                # if that line is connecting the current point check the angle between
                # the generated angle and connected angle
                # or we can just ignore since the 'does it touch node' part already
                # handles this
                pass
            else:
                P1ac = pointlist[current].coord()
                P1bc = P1ac + vector*maxd
                P2a,P2b = linelist[l]
                P2ac = pointlist[P2a].coord()
                P2bc = pointlist[P2b].coord()
                
                
                linetouch,touchdist = check_lines_intersect(P1ac,P1bc,P2ac,P2bc)
                if linetouch:
                    if touchdist < closestdist:
                        print(P2a,P2b)
                        print('line intersects at dist',touchdist)
                        closestpoint = ('Line',l)
                        closestdist = touchdist
                # if not check if the current line intersects the other line
                # if it does add the distance between the 2 points
        # iterate every point to find the closest touching point
        for p in range(len(pointlist)):
            # ignore if back to current point
            if p == current:
                continue
            cantouch,touchdist = pointlist[p].point_touch(pointlist[current].coord(),vector)
            if cantouch:
                # if there is a closer point that can touch add that point
                if closestdist > touchdist:
                    closestpoint = ('Point',p)
                    closestdist = touchdist
                    print('will collide with point',p,'(',pointlist[p].coord(),')')
                # point intersections take precedence over line intersections
                elif minnumber(closestdist-touchdist,zeropow) >= 0:
                    if closestpoint[0] == 'Line':
                        closestpoint = ('Point',p)
                        closestdist = touchdist
                        print('will collide with point',p,'(',pointlist[p].coord(),')')
        newpoint = pointlist[current].coord()+vector*maxd
        # if closets point is a line, split the line at the touch point
        # then add a point there
        if type(closestpoint) == tuple:
            inttype,ind = closestpoint
            if inttype == 'Line':
                index = len(pointlist)
                pointlist.append(Point(index,
                                       pointlist[current].coord()[0]+closestdist*vector[0],
                                       pointlist[current].coord()[1]+closestdist*vector[1],
                                       flip,size,
                                       'Intersection'))
                cutlines,cutlinee = linelist[ind]
                linelist[ind] = (cutlines,index)
                linelist.append((index,cutlinee))
                linelist.append((current,index))
            if inttype == 'Point':
                goback = False
                index = ind
                #pointlist[index].flips += flip
                if (current,index) not in linelist and (index,current) not in linelist:
                    linelist.append((current,index))
                else:
                    print('no new path')
                if pointlist[index].flips == 'S':
                    print('went back to the start')
                elif len(pointlist[index].flips) >= maxflip:
                    print('revisiting a place with '+str(maxflip)+' flips - back to the start')
                    goback = True
                else:
                    pointlist[index].flips += flip
        else:
            index = len(pointlist)
            pointlist.append(Point(index,newpoint[0],newpoint[1],flip,size,'Termini'))
            linelist.append((current,index))

        # change termini to node if there are more than 2 connecting paths
        prevnode = {}
        if pointlist[current].pointtype == 'Termini':
            prevnode[current] = 0
        if pointlist[index].pointtype == 'Termini':
            prevnode[index] = 0
        if len(prevnode) > 0:
            for lines in linelist:
                for prev in prevnode:
                    if prev in lines:
                        prevnode[prev] += 1
        for prev in prevnode:
            if prevnode[prev] > 1:
                pointlist[prev].pointtype = 'Node'
            
        current = index
        if type(closestpoint) == tuple:
            if inttype == 'Line':
                print('cut line',(cutlines,cutlinee),'to',(cutlines,index),'and',(index,cutlinee))
                print('new intersection',pointlist[current].coord(),pointlist[current].flips)
            if inttype == 'Point':
                print('revisit point',pointlist[current].coord(),pointlist[current].flips)
                if goback:
                    print('went back to the start as visited point already had 3 flips')
                    if len(startcoord) == 1:
                        current = 0
                    else:
                        current = rng.integers(len(startcoord))
        else:
            print('new point',pointlist[current].coord(),pointlist[current].flips)
    print()

pathds = []
pointds = []
# draw each line
for l in range(len(linelist)):
    P1,P2 = linelist[l]
    P1 = pointlist[P1].coord()
    P2 = pointlist[P2].coord()
    pathds.append({'Line Index':l,'Point Index 1':linelist[l][0],'Point Index 2':linelist[l][1]})
    plt.plot([P1[0],P2[0]],[P1[1],P2[1]],color='k',linewidth=1)

pathds = pd.DataFrame(pathds)
pathds.to_csv('pathds.csv',index=False)

for p in range(len(pointlist)):
    P = pointlist[p]
    print('Point',p,P.coord(),P.pointtype,P.flips)
    if p == current:
        color = 'r'
    else:
        color = 'b'
    if P.pointtype == 'Intersection':
        plt.scatter(P.x,P.y,marker='o',edgecolors=color,facecolors='none',s=60)
    else:
        plt.plot(P.x,P.y,marker='o',color=color)
    pointds.append({'Point Index':p,'Flips':P.flips,'X':P.x,'Y':P.y,'Point Type':P.pointtype})
    plt.text(P.x+0.01*width,P.y+0.01*height,str(P.flips))
    #plt.text(P.x+0.01*width,P.y+0.01*height,str(P.index))
pointds = pd.DataFrame(pointds)
pointds.to_csv('pointds.csv',index=False)
plt.show() 
    