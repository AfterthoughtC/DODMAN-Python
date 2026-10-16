from PyDodman.Vector import Vector2
from typing import Literal
import pandas as pd


class Point():
    coordinate: Vector2
    point_type: Literal['origin','node','termini','intersection']
    indexes: list

    def __init__(self,x:float,y:float,point_type:Literal['origin','node','termini','intersection'],indexes:list[int]|int):
        self.coordinate = Vector2(x,y)
        self.point_type = point_type
        if type(indexes) == int:
            self.indexes = [indexes]
        else:
            self.indexes = [int(i) for i in indexes]


    def get_dictionary(self,separator:str='|') -> dict:
        return {'i':None,
                'x':self.coordinate.x,'y':self.coordinate.y,
                'point_type':self.point_type,
                'indexes': separator.join([str(a) for a in self.indexes])}


class DodmanMapGenerator():
    # The list of points
    points: list[Point]
    paths: set[tuple[int,int]]


    def make_databases(self,index_separator:str='|') -> tuple[pd.DataFrame,pd.DataFrame]:
        PointDs:list[dict] = []
        for i in range(len(self.points)):
            point: Point = self.points[i]
            PointDs.append(point.get_dictionary(index_separator))
            PointDs[-1]['i'] = i
        PointDs: pd.DataFrame = pd.DataFrame(PointDs)

        PathDs:list[dict] = []
        for path in self.paths:
            start,end = path
            PathDs.append({'start':start,'end':end})
        PathDs: pd.DataFrame = pd.DataFrame(PathDs)
        return PointDs,PathDs