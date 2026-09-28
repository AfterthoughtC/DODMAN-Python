from PyDodman.Vector import Vector2

class LineEquation():

    A: float
    B: float
    C: float

    def __init__(self,A: float, B: float, C: float):
        self.A = A
        self.B = B
        self.C = C
        # A*x + B*y = C
        # B*y = C - A*x
        # y = (-A / B)*x + (C - B)


    def __repr__(self):
        return f"LineEquation({self.A}, {self.B}, {self.C})"


    def __str__(self):
        return f"Line({self.A}*x+{self.B}*y={self.C})"


    def get_direction(self) -> Vector2:
        """Returns the direction Vector of the equation

        Returns:
            Vector2: The direction Vector, Vector2(B,-A)
        """
        return Vector2(self.B, -self.A)


    def get_x_range(self,y:float) -> list[float]:
        """Returns the range of possible x values given the y value

        Args:
            y (float): The y value

        Returns:
            list[float]: A list containing 0, 1 or 2 float values. 2 float values means 'x is between these 2 values'. 1 float value means 'x is exactly this value'. 0 means that the Line Equation is invalid in some way.
        """
        if self.A == 0 and self.B == 0:
            return []
        elif self.A == 0:
            return [float('-inf'),float('inf')]
        else:
            return [(self.C - (self.B*y)) / self.A]


    def get_y_range(self,x:float) -> list[float]:
        """Returns the range of possible y values given the x value

        Args:
            x (float): The x value

        Returns:
            list[float]: A list containing 0, 1 or 2 float values. 2 float values means 'y is between these 2 values'. 1 float value means 'y is exactly this value'. 0 means that the Line Equation is invalid in some way.
        """
        if self.B == 0 and self.A == 0:
            return []
        elif self.B == 0:
            return [float('-inf'),float('inf')]
        else:
            return [(self.C - (self.A*x)) / self.B]


def two_coords_to_line_equation(coord1:Vector2,coord2:Vector2) -> LineEquation:
    """Using two coordinates, generate the line equation for a line that connects the two coordinates.
    Source: https://www.geeksforgeeks.org/dsa/program-for-point-of-intersection-of-two-lines/

    Args:
        coord1 (Vector2): The first coordinate
        coord2 (Vector2): The second coordinate

    Returns:
        LineEquation: The line equation for a line that connects the two points
    """
    a = coord2.y - coord1.y
    b = coord1.x - coord2.x
    c = a*coord1.x + b*coord1.y
    return LineEquation(a,b,c)


def coord_and_vector_to_line_equation(coord:Vector2,direction:Vector2,next_coord_pos:float=2) -> tuple[float,float,float]:
    """Using a coordinate and direction vector, generate the line equation for a line that passes through the coordinate and is parallel to the direction.


    Args:
        coord (Vector2): The origin coordinate
        direction (Vector2): The displacement vector
        next_coord_pos (float, optional): Distance of the derived second coordinate from the origin relative to the current direction. Cannot be zero. Defaults to 2.

    Returns:
        LineEquation: The line equation for a line that passes through coord and is parallel to direction
    """
    next_coord: Vector2 = coord + direction*next_coord_pos
    return two_coords_to_line_equation(coord,next_coord)