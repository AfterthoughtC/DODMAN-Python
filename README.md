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

# Scripts
There are two main scripts; dodman.py and dodmanbokeh.py.

## dodman.py
This contains the main algorithm that generates the map.
The following values are to be set by the user before generation:
| Value       | Value Type           | Description|
| ------------- |-------------| -----|
|width|integer or float|How big the map will be horizontally|
|height|integer or float|How big the map will be vertically|
|maptype|string|What type of border will be used for the map. Currently only 'square' exists though I have plans to support circular borders.|
|angle_no|integer|If zero, the angles will be generated via a uniform distribution. If any number above 0 the angles generated will be multiples of 360 degrees / angle_no (ex. if you place 12 it will only generate angles of 0, 30, 60, 90 ...)|
|size|integer or float|The radius of each point. If the shortest path between the path and projected path (the normal) is smaller than size, that point will be considered a possible candidate for snapping that projected path to.|
|zeropow|integer larger than 0|Accuracy is sometimes lost due to the irrational nature of the numbers. This value is used to help snap some very small numbers into zero.|
|iterations|integer|Number of 'flips' the algorithm will do.|
|flipres|list of strings|The possible results of the flips.|
|maxflip|integer|Maximum number of flips a point can have.|

The algorithm will generate a preview image of the map and two files; pointds.csv (contains information on the generated points) and pathds.csv (contains information on the paths).

## dodmanbokeh.py
An attempt at creating the map in bokeh with a more interactive interface. Requires both pathds.csv, pointds.csv and flip meaning.csv to be used.

# Future Plans
* Allow for a circular boundaries
* Improve the functions to reduce need for zeropow and hence improve accuracy
* Better documentation
* Recreate the algorithm in Godot
