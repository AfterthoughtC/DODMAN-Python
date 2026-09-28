from abc import ABC, abstractmethod
from PyDodman.Vector import Vector2


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


    @abstractmethod
    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        pass


    @abstractmethod
    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        pass



class RectangleBoundary(AbstractBoundary):

    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary. In the case of a rectangle it is the distance from the start corner coordinate to the end corner coordinate.

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        return (self.x_length**2 + self.y_length**2)**0.5


    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        if type(arg1) == Vector2:
            coord: Vector2 = arg1
        else:
            coord: Vector2 = Vector2(arg1,arg2)

        # get the min and max possible values
        x_min: float = min(self.x_start,
                           self.x_start+self.x_length)
        x_max: float = max(self.x_start,
                           self.x_start+self.x_length)
        y_min: float = min(self.y_start,
                           self.y_start+self.y_length)
        y_max: float = max(self.y_start,
                           self.y_start+self.y_length)

        # get smallest difference between extreme sides and the coordinate
        x_diff: float = min(abs(coord.x-x_min),abs(coord.x-x_max))
        y_diff: float = min(abs(coord.y-y_min),abs(coord.y-y_max))

        if x_diff == 0 or y_diff == 0:
            return 0
        elif x_min < coord.x and coord.x < x_max and \
            y_min < coord.y and coord.y < y_max:
            return 1
        else:
            return -1


class EllipseBoundary(AbstractBoundary):

    def longest_line_length(self) -> float:
        """Returns the longest possible straight line that can fit in the boundary. In the case of an ellipse it is the longest of the x and y length values.

        Returns:
            float: The longest possible line that can fit in the boundary
        """
        return max(abs(self.x_length),abs(self.y_length))


    def is_inside_boundary(self,arg1:Vector2|float,arg2:None|float=None) -> int:
        """Returns an integer indicating if the point is within the boundary, on the perimeter or outside.

        Args:
            arg1 (Vector2 | float): The x-axis of the Vector of the whole vector itself.
            arg2 (None | float, optional): The y-axis of the Vector if arg1 is not the Vector2 itself. Defaults to None.

        Returns:
            int: 1 if the coordinate is in the boundary, 0 if on the border and -1 if outside.
        """
        if type(arg1) == Vector2:
            coord: Vector2 = arg1
        else:
            coord: Vector2 = Vector2(arg1,arg2)

        a: float = 0.5*self.x_length
        b: float = 0.5*self.y_length

        left_side: float = (coord.x / a)**2 + (coord.y / b)**2

        if left_side == 1:
            return 0
        elif left_side < 1:
            return 1
        else:
            return -1

        