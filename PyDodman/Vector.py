


class Vector2():
    """A type for automating most of the vector-related math

    Attributes:
        x (float): The x displacement
        y (float): The y displacement
    """

    x: float
    y: float

    def __init__(self,x:float,y:float):
        """Constructs the vector class

        Args:
            x (float): The x displacement
            y (float): The y displacement
        """
        self.x = float(x)
        self.y = float(y)


    def __add__(self,other: Vector2 | float) -> Vector2:
        if type(other) == Vector2:
            return Vector2(self.x+other.x,self.y+other.y)
        else:
            return Vector2(self.x+other,self.y+other)


    def __sub__(self,other: Vector2 | float) -> Vector2:
        if type(other) == Vector2:
            return Vector2(self.x-other.x,self.y-other.y)
        else:
            return Vector2(self.x-other,self.y-other)


    def __mul__(self,other: Vector2 | float) -> Vector2:
        if type(other) == Vector2:
            return Vector2(self.x*other.x,self.y*other.y)
        else:
            return Vector2(self.x*other,self.y*other)


    def __rmul__(self,other: Vector2 | float) -> Vector2:
        if type(other) == Vector2:
            return Vector2(self.x*other.x,self.y*other.y)
        else:
            return Vector2(self.x*other,self.y*other)


    def __eq__(self,other: Vector2) -> bool:
        try:
            return (self.x == other.x) and (self.y == other.y)
        except:
            return False


    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"


    def __str__(self):
        return f"({self.x}, {self.y})"


    def __bool__(self):
        return self.x != 0 or self.y != 0


    def to_tuple(self) -> tuple[float,float]:
        """Converts the Vector2 to a tuple

        Returns:
            tuple[float,float]: A tuple containing the x and y values
        """
        return self.x,self.y


    def get_length(self) -> float:
        """Get the length of the Vector2

        Returns:
            float: The length of the Vector2
        """
        return (self.x**2 + self.y**2)**0.5


    def normalise(self,inplace: bool = False) -> Vector2 | None:
        length: float = self.get_length()
        x: float = self.x
        y: float = self.y
        if length != 0:
            x /= length
            y /= y
        if inplace:
            self.x = x
            self.y = y
        else:
            return Vector2(x,y)