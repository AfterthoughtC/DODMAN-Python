from abc import ABC, abstractmethod
from PyDodman.Vector import Vector2
from PyDodman.util import is_almost_equal, RELATIVE_TOLERANCE, EPSILON


class AbstractBoundary(ABC):

    x_start: float
    y_start: float
    x_length: float
    y_length: float


    def __init__(self,x_start:float,y_start:float,x_length:float,y_length:float):
        self.x_start = x_start
        self.y_start = y_start
        self.x_length = x_length
        self.y_length = y_length


    def get_origin(self) -> Vector2:
        """Get the coordinate of the centre of the boundary

        Returns:
            Vector2: The coordinate of the centre of the boundary
        """
        return Vector2(self.x_start + self.x_length / 2, self.y_start + self.y_length / 2)


    def get_x_min_max(self) -> tuple[float,float]:
        """Get the min and max possible value for x

        Returns:
            tuple[float,float]: The min and max possible value for x
        """
        start_val: float = self.x_start
        end_val: float = self.x_start+self.x_length
        return min(start_val,end_val),max(start_val,end_val)


    def get_y_min_max(self) -> tuple[float,float]:
        """Get the min and max possible value for y

        Returns:
            tuple[float,float]: The min and max possible value for y
        """
        start_val: float = self.y_start
        end_val: float = self.y_start+self.y_length
        return min(start_val,end_val),max(start_val,end_val)


    @abstractmethod
    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        pass


    @abstractmethod
    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None, rel_tol:float=RELATIVE_TOLERANCE, abs_tol:float=EPSILON) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.
            rel_tol (float, optional):  Relative tolerance; the maximum allowed difference between 0 and the distance from the boundary, relative to the larger absolute value of the distance. Defaults to RELATIVE_TOLERANCE.
            abs_tol (float, optional): The minimum absolute tolerance when comparing the distance from the boundary to 0. abs_tol must be at least zero. Defaults to EPSILON.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        pass


    @abstractmethod
    def max_possible_distance(self,start_point:Vector2,direction:Vector2) -> float:
        """The maximum possible distance a line can extend from start_point in the given direction before hitting a border.

        Args:
            start_point (Vector2): _description_
            direction (Vector2): _description_

        Returns:
            float: _description_
        """
        pass



class RectangleBoundary(AbstractBoundary):

    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary. In the case of a rectangle it is the distance from the start corner coordinate to the end corner coordinate.

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        return (self.x_length**2 + self.y_length**2)**0.5


    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None, rel_tol:float=RELATIVE_TOLERANCE, abs_tol:float=EPSILON) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.
            rel_tol (float, optional):  Relative tolerance; the maximum allowed difference between 0 and the distance from the boundary, relative to the larger absolute value of the distance. Defaults to RELATIVE_TOLERANCE.
            abs_tol (float, optional): The minimum absolute tolerance when comparing the distance from the boundary to 0. abs_tol must be at least zero. Defaults to EPSILON.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        if type(arg1) == Vector2:
            coord: Vector2 = arg1
        else:
            coord: Vector2 = Vector2(arg1,arg2)

        # get the min and max possible values
        x_min, x_max = self.get_x_min_max()
        y_min,y_max = self.get_y_min_max()


        # get smallest difference between extreme sides and the coordinate
        x_diff: float = min(abs(coord.x-x_min),abs(coord.x-x_max))
        y_diff: float = min(abs(coord.y-y_min),abs(coord.y-y_max))

        if is_almost_equal(x_diff,0,RELATIVE_TOLERANCE,EPSILON) or is_almost_equal(y_diff,0,RELATIVE_TOLERANCE,EPSILON):
            return 0
        elif x_min < coord.x and coord.x < x_max and \
            y_min < coord.y and coord.y < y_max:
            return 1
        else:
            return -1


    def max_possible_distance(self,start_point:Vector2,direction:Vector2) -> float:
        # TODO: Complete this function
        pass
        


class EllipseBoundary(AbstractBoundary):

    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary. In the case of an ellipse it is the longest of the x and y length values.

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        return max(abs(self.x_length),abs(self.y_length))


    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None, rel_tol:float=RELATIVE_TOLERANCE, abs_tol:float=EPSILON) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.
            rel_tol (float, optional):  Relative tolerance; the maximum allowed difference between 0 and the distance from the boundary, relative to the larger absolute value of the distance. Defaults to RELATIVE_TOLERANCE.
            abs_tol (float, optional): The minimum absolute tolerance when comparing the distance from the boundary to 0. abs_tol must be at least zero. Defaults to EPSILON.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        if type(arg1) == Vector2:
            coord: Vector2 = arg1
        else:
            coord: Vector2 = Vector2(arg1,arg2)

        a: float = 0.5*self.x_length
        b: float = 0.5*self.y_length

        # formula for an ellipse is (x/a)^2 + (x/b)^2 = 1 where a and b are the x and y radii respectively
        left_side: float = (coord.x / a)**2 + (coord.y / b)**2

        if is_almost_equal(left_side,1,RELATIVE_TOLERANCE,EPSILON):
            return 0
        elif left_side < 1:
            return 1
        else:
            return -1


    def max_possible_distance(self,start_point:Vector2,direction:Vector2) -> float:
        # TODO: Complete this function
        pass