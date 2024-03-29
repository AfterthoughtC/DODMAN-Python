# DODMAN-Python
DODMAN is a map generation system designed by Alfred Valley for the Pleasure-not-Business Card RPG Jam. DODMAN-Python is my attempt at recreating his algorithm in Python.

You can find Alfred Valley's DODMAN [here](https://alfredvalley.itch.io/dodman) and the jam submission page [here](https://itch.io/jam/pleasurecardrpg/rate/962103).

# Prerequisites
The python scripts were programmed in Python 3.7.

The following packages are used by the scripts:
* Numpy: Trigonometry functions, random number generator and multi-dimensional arrays to handle and simplify coordinate calculation.
* Matplotlib: Map preview.
* Pandas: Creating and reading csv files.
* Bokeh: Package used to try and plot the map in a more interactive environment.

# Additional Features
The python implementation used here also supports some features over Alfred Valley's DODMAN:
## Multiple Start Points
Rather than being stuck with just one start point, users can set multiple start points within the map. The point that one returns to once no more moves can be made from the current point can be decided either randomly or in order.

# Scripts
There are two main scripts; dodman.py and dodmanbokeh.py.

## dodman.py
This contains the main algorithm that generates the map.
The following values are to be set by the user before generation:
| Value       | Value Type           | Description|
| ------------- |-------------| -----|
|width|integer or float|How big the map will be horizontally. The horizontal coordinates will start from -width/2 and end at width/2.|
|height|integer or float|How big the map will be vertically. The vertical coordinates will start from -height/2 and end at height/2.|
|start_coord|list of tuples / lists, size 2 each, containing floats of value between 0 and 1|The coordinates of all start points on the map relative to map size. (0.5, 0.5) places the start at the map centre.|
|random_start|boolean|Only applicable if you have more than 1 start coordinates. If True the start coordinate to start/return to if no more moves can be made is determined randomly. If False coordinate to start/return to will be the order of the coordinates in startcoord (ex. 0, 1, 2, ...)|
|map_shape|string|What type of border will be used for the map. Values include 'square' and 'circle'.|
|angle_no|integer|If zero, the angles will be generated via a uniform distribution. If any number above 0 the angles generated will be multiples of 360 degrees / angle_no (ex. if you place 12 it will only generate angles of 0, 30, 60, 90 ...)|
|point_rad|integer or float|The radius of each point. If the shortest path between the path and projected path (the normal) is smaller than size, that point will be considered a possible candidate for snapping that projected path to.|
|epsilon|any float larger than zero|When calclating, any value smaller than epsilon will be converted to zero.|
|seed|integer or None|The random seed to use for generating your direction and coin flip/dice throw. Leaving the value as None will use the system's own random seed.|
|flip_results|list of strings|The possible results of the flips.|
|max_label_length|integer|Maximum number of flips a point can have.|

The algorithm will generate a preview image of the map and three files; pointds.csv (contains information on the generated points), pathds.csv (contains information on the paths) and plotimg.png (the image file in png format).

![Example Preview Map](https://github.com/AfterthoughtC/DODMAN-Python/blob/main/images/DODMAN%20Preview.png)

Fig 1: A possible preview image generated after running dodman.py. Red point is the last visited point.

![Example Preview Two Start Points](https://github.com/AfterthoughtC/DODMAN-Python/blob/main/images/DODMAN%20Preview%20-%20Multiple%20Start%20Points.png)

Fig 2: A possible preview image when 2 start points are used.

## dodmanbokeh.py
An attempt at creating the map in bokeh with a more interactive interface. Requires both pathds.csv, pointds.csv and flip meaning.csv to be used.

![Example Bokeh](https://github.com/AfterthoughtC/DODMAN-Python/blob/main/images/Bokeh%20Map%20Preview.png)

Fig 3: How the interactive map looks like on Bokeh.

# Future Plans
* Improve the functions to reduce need for zeropow and hence improve accuracy
* Better documentation
* Recreate the algorithm in Godot or some user interface
