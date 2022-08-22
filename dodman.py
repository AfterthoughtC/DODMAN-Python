# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:12:42 2021

@author: AfterthoughtC
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Union


def minnumber(number,zeropow):
    if abs(number) < 10**-zeropow:
        return(0)
    return(number)


# function for generating the angle
def generate_angle(angle_no,rng = np.random.default_rng()):
    if angle_no == 0:
        return(rng.random()*2*np.pi)
    return(2*np.pi/angle_no*rng.integers(0,angle_no))


def generate_vector(angle,zeropow):
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
    
    def get_width(self):
        return(self.xmax-self.xmin)
    
    def get_height(self):
        return(self.ymax-self.ymin)
    
    def intersect_disp(self,startpoint,vector,zeropow):
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
    
    def get_width(self):
        return(self.a*2)
    
    def get_height(self):
        return(self.b*2)
    
    def intersect_disp(self,startpoint,vector,zeropow):
        xm = vector[0]
        ym = vector[1]
        xo = startpoint[0]
        yo = startpoint[1]
        print([xm,ym],[xo,yo])
        quad_a = self.b*self.b*xm*xm+self.a*self.a*ym*ym
        quad_b = 2*(self.b*self.b*xo*xm+self.a*self.a*yo*ym)
        quad_c = self.b*self.b*xo*xo+self.a*self.a*yo*yo-self.a*self.a*self.b*self.b
        print(quad_a,quad_b,quad_c)
        sqrtpart = quad_b*quad_b - 4*quad_a*quad_c
        sqrtpart = minnumber(sqrtpart,zeropow)
        if sqrtpart < 0:
            print('Square root part when calculating displacement is <0.')
            print('Taking it as no possible d value that can give an intersect')
            return(0)
        dp = (-quad_b + sqrtpart**0.5)/2/quad_a
        dn = (-quad_b - sqrtpart**0.5)/2/quad_a
        d = []
        if dp > 0:
            d.append(dp)
        if dn > 0:
            d.append(dn)
        if len(d) == 0:
            d.append(0)
        return(min(d))
        """
        negbcomp = xm*xo*self.b**2 + ym*yo*self.a**2
        sqrtcomp = 2*xm*xo*ym*yo + (self.b*xm)**2 + (self.a*ym)**2 - (xm*yo)**2 - (xo*ym)**2
        sqrtcomp = minnumber(sqrtcomp,zeropow)
        twoacomp = (self.b*xm)**2 + (self.a*ym)**2
        dp = (-negbcomp+self.a*self.b*sqrtcomp**0.5)/twoacomp
        if pd.isna(dp):
            print(negbcomp,self.a,self.b,sqrtcomp,twoacomp)
            print(-negbcomp+self.a*self.b*sqrtcomp**0.5)
            1/0
        dn = (-negbcomp-self.a*self.b*sqrtcomp**0.5)/twoacomp
        dp = minnumber(dp,zeropow)
        dn = minnumber(dn,zeropow)
        #print('dpdn',dp,dn)
        d = []
        if dp > 0:
            d.append(dp)
        if dn > 0:
            d.append(dn)
        if len(d) == 0:
            d = 0
        else:
            d = min(d)
        """
        return(d)
    
    def within_boundary(self,xo,yo):
        return((xo/self.a)**2+(yo/self.b)**2<=1)

# the default point class
class Point():
    def __init__(self,index,x,y,flips,pointrad,pointtype,zeropow):
        self.index = index
        self.x = minnumber(x,zeropow)
        self.y = minnumber(y,zeropow)
        self.flips = flips
        self.pointrad = pointrad
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
        if sdist > self.pointrad:
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


def check_lines_intersect(P1,P2,P3,P4,zeropow):
    
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


int_or_float = Union[int,float]


class DodmanGen():
    
    def __init__(self,width:int_or_float=400,height:int_or_float=400,
                 start_coord=[(0.5,0.5)],random_start:bool=False,
                 map_shape:str='ellipse',angle_no:int=12,point_rad:int_or_float=20,
                 zero_pow:int_or_float=11,seed:Union[int,type(None)]=None,
                 flip_results:Union[tuple,list]=['H','T'],max_label_length:int=3):
        #self.width = width
        #self.height = height
        self.point_list = [] # for storing the list of points
        self.start_count = len(start_coord) # how many possible start points
        for i in range(len(start_coord)):
            self.point_list.append(Point(i,(start_coord[i][0]-0.5)*width,(start_coord[i][1]-0.5)*height,'S',10,'Start',zero_pow))
        self.path_list = [] # for storing the list of paths
        
        # bound is the shape of the map
        # width and height will be stored inside the bound
        self.random_start = random_start
        if map_shape == 'ellipse':
            self.bound = EllipseBoundary(width,height)
        else:
            self.bound = SquareBoundary(width,height)
        # angle_no is how many possible angles it can generate
        # if angle_no is zero, it is a uniform distribution
        self.angle_no = angle_no
        # point_rad is the radius of each point
        self.point_rad = point_rad
        # maximum possible 10^-zero_pow value before a number gets snapped to zero
        self.zero_pow = zero_pow
        # the random number generator
        self.rng = np.random.default_rng(seed)
        self.flip_results = flip_results
        self.max_label_length = max_label_length
        
        if len(start_coord) == 1 or random_start == False:
            self.current_start = 0
        else:
            self.current_start = self.rng.integers(len(self.start_count))
        
        # current coordinate
        self.current = self.current_start
        
        # storing the rolls made
        self.rolls_made = []
    
    def get_width(self):
        return(self.bound.get_width())
    
    def get_height(self):
        return(self.bound.get_height())

    def flip_once(self,printmid=False):
        this_roll = []
        # generate the angle and vector
        angle = generate_angle(self.angle_no,self.rng)
        flip = self.flip_results[self.rng.integers(len(self.flip_results))]
        vector = generate_vector(angle,self.zero_pow)
        this_roll.append((angle,flip))
        
        if printmid:
            print('current point',self.point_list[self.current].coord(),self.point_list[self.current].flips)
            print('vector',vector)
            
        # create the maximum distance
        maxd = self.bound.intersect_disp(self.point_list[self.current].coord(),vector,self.zero_pow)

        # if that displacement will take us off the paper / will not exit the
        # point 'collision shape', add a point
        if abs(maxd)<=self.point_rad:
            if printmid:
                print('The vector will not take us very far from the current node')
            
            # do not change start point if we are already back at the start
            if self.point_list[self.current].flips == 'S':
                if printmid:
                    print('Start point will not be changed')
            
            # if we are at a non-start point, check if we can add more flips
            # if we cannot return to the start
            elif len(self.point_list[self.current].flips) >= self.max_label_length:
                if printmid:
                    print('Current point already has %s flips; returning to start'%str(len(self.point_list[self.current].flips)))
                if self.start_count == 1:
                    self.current_start = 0
                elif self.random_start:
                    self.current_start = self.rng.integers(self.start_count)
                    this_roll.append(self.current_start)
                else:
                    self.current_start = (self.current_start + 1) % self.start_count
                self.current = self.current_start
            # if we can, add flip
            else:
                self.point_list[self.current].flips += flip
                if printmid:
                    print('updated point',self.point_list[self.current].coord(),self.point_list[self.current].flips)
        # if we can move, move
        else:
            closestpoint = None
            closestdist = float('+inf')
            # iterate every pre-existing path to see which has the closest intersection
            for l in range(len(self.path_list)):
                if self.current in self.path_list[l]:
                    # if that line is connecting the current point check the angle between
                    # the generated angle and connected angle
                    # or we can just ignore since the 'does it touch node' part already
                    # handles this
                    pass
                else:
                    # if not check if the current line intersects the other line
                    P1ac = self.point_list[self.current].coord()
                    P1bc = P1ac + vector*maxd
                    P2a,P2b = self.path_list[l]
                    P2ac = self.point_list[P2a].coord()
                    P2bc = self.point_list[P2b].coord()
                    
                    linetouch,touchdist = check_lines_intersect(P1ac,P1bc,P2ac,P2bc,self.zero_pow)
                    if linetouch:
                        # if it does find the distance between the point and line
                        if touchdist < closestdist:
                            if printmid:
                                print(P2a,P2b)
                                print('line intersects at dist',touchdist)
                            closestpoint = ('Line',l)
                            closestdist = touchdist
            # iterate every point to find the closest touching point
            for p in range(len(self.point_list)):
                # ignore if back to current point
                if p == self.current:
                    continue
                cantouch,touchdist = self.point_list[p].point_touch(self.point_list[self.current].coord(),vector)
                if cantouch:
                    # if there is a closer point that can touch add that point
                    if closestdist > touchdist:
                        closestpoint = ('Point',p)
                        closestdist = touchdist
                        if printmid:
                            print('will collide with point %s (%s)'%(p,self.point_list[p].coord()))
                    # point intersections take precedence over line intersections
                    elif minnumber(closestdist-touchdist,self.zero_pow) >= 0:
                        if closestpoint[0] == 'Line':
                            closestpoint = ('Point',p)
                            closestdist = touchdist
                            if printmid:
                                print('will collide with point %s (%s)'%(p,self.point_list[p].coord()))
            newpoint = self.point_list[self.current].coord()+vector*maxd
            # if closets point is a line, split the line at the touch point
            # then add a point there
            if type(closestpoint) == tuple:
                inttype,ind = closestpoint
                if inttype == 'Line':
                    index = len(self.point_list)
                    self.point_list.append(Point(index,
                                           self.point_list[self.current].coord()[0]+closestdist*vector[0],
                                           self.point_list[self.current].coord()[1]+closestdist*vector[1],
                                           flip,self.point_rad,
                                           'Intersection',self.zero_pow))
                    cutlines,cutlinee = self.path_list[ind]
                    self.path_list[ind] = (cutlines,index)
                    self.path_list.append((index,cutlinee))
                    self.path_list.append((self.current,index))
                if inttype == 'Point':
                    goback = False
                    index = ind
                    #pointlist[index].flips += flip
                    if (self.current,index) not in self.path_list and (index,self.current) not in self.path_list:
                        self.path_list.append((self.current,index))
                    else:
                        if printmid:
                            print('no new path')
                    if self.point_list[index].flips == 'S':
                        if printmid:
                            print('went back to the start')
                    elif len(self.point_list[index].flips) >= self.max_label_length:
                        if printmid:
                            print('revisiting a place with %s flips - back to the start'%self.max_label_length)
                        goback = True
                    else:
                        self.point_list[index].flips += flip
            else:
                index = len(self.point_list)
                self.point_list.append(Point(index,newpoint[0],newpoint[1],flip,self.point_rad,'Termini',self.zero_pow))
                self.path_list.append((self.current,index))
                
            # change termini to node if there are more than 2 connecting paths
            prevnode = {}
            if self.point_list[self.current].pointtype == 'Termini':
                prevnode[self.current] = 0
            if self.point_list[index].pointtype == 'Termini':
                prevnode[index] = 0
            if len(prevnode) > 0:
                for lines in self.path_list:
                    for prev in prevnode:
                        if prev in lines:
                            prevnode[prev] += 1
            for prev in prevnode:
                if prevnode[prev] > 1:
                    self.point_list[prev].pointtype = 'Node'
                
            self.current = index
            if type(closestpoint) == tuple:
                if inttype == 'Line':
                    if printmid:
                        print('cut line',(cutlines,cutlinee),'to',(cutlines,index),'and',(index,cutlinee))
                        print('new intersection',self.point_list[self.current].coord(),self.point_list[self.current].flips)
                if inttype == 'Point':
                    if printmid:
                        print('revisit point',self.point_list[self.current].coord(),self.point_list[self.current].flips)
                    if goback:
                        if printmid:
                            print('went back to the start as visited point already had %s flips'%self.max_label_length)
                        if self.start_count == 1:
                            self.current = 0
                        else:
                            self.current = self.rng.integers(self.start_count)
                            this_roll.append(self.current)
            else:
                if printmid:
                    print('new point',self.point_list[self.current].coord(),self.point_list[self.current].flips)
        if printmid:
            print('Flips Made:',this_roll)
        self.rolls_made.append(this_roll)



if __name__ == '__main__':
    
    # we start by determining the width and height of the map
    width = 400 # width of the map
    height = 400 # height of the map
    start_coord = [(0.5,0.5)] # if we want to have multiple start coordinates
    random_start = False # Only relevant if there are multiple coordinates.
                    # If True when the next startcoord will be randomly chosen
                    # if False the next startcoord will go in order (ex. startcoord 1, 2, 3)
    map_shape = 'ellipse' # 2 types; circle / ellipse and square / rectangle
    angle_no = 12 # if more than 0, will only generate angles whose size is of multiple 2pi/angle_no
    point_rad = 20 # to prevent points from being too close to each other, size sets a minimum diameter
                # this diameter is also used to 'round' path lengths and directions to points that fall
                # within the circle
    zero_pow = 11 # due to accuracy loss, any number that is less than 10^-zeropow will be converted straight to 0
    rollno = 20 # how many times to flip your coin/ roll your dice
    #seed = None
    seed = None # the seed to use. If none the file will generate its own seed
    flip_results = ['H','T'] # the results for flipping the coin
    max_label_length = 3 # the maximum number of flip results a label can store
    #flip_results = ['1','2','3','4','5','6'] # the results for casting the dice
    #max_label_length = 2 # the maximum number of flip results a label can store
    
    
    dodgen = DodmanGen(width=width,height=height,
                       start_coord=start_coord,random_start=random_start,
                       map_shape=map_shape,angle_no=angle_no,point_rad=point_rad,
                       zero_pow=zero_pow,seed=seed,flip_results=flip_results,
                       max_label_length=max_label_length)
    while len(dodgen.rolls_made) < 20:
        dodgen.flip_once(True)
        print()
    path_list = dodgen.path_list
    point_list = dodgen.point_list
    current = dodgen.current
    width = dodgen.get_width()
    height = dodgen.get_height()
    
    pathds = []
    pointds = []
    # draw each line
    for l in range(len(path_list)):
        P1,P2 = path_list[l]
        P1 = point_list[P1].coord()
        P2 = point_list[P2].coord()
        pathds.append({'Line Index':l,'Point Index 1':path_list[l][0],'Point Index 2':path_list[l][1]})
        plt.plot([P1[0],P2[0]],[P1[1],P2[1]],color='k',linewidth=1)
    
    pathds = pd.DataFrame(pathds)
    pathds.to_csv('pathds.csv',index=False)
    
    for p in range(len(point_list)):
        P = point_list[p]
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
    plt.savefig('plotimg.png')
    plt.show()
    print("Map generated and saved. Press enter to finish the script.")
    