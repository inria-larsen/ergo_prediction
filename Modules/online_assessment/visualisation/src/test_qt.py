#!/usr/bin/python3
#
# Scatter plot example using pyqtgraph with PyQT5
#
# Install instructions for Mac:
#   brew install pyqt
#   pip3 install pyqt5 pyqtgraph
#   python3 pyqtgraph_pyqt5.py

import sys

import numpy as np
import pyqtgraph as pg

# Set white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Generate random points
n = 1000
print('Number of points: ' + str(n))
data = np.random.normal(size=(2, n))

# Create the main application instance
app = pg.mkQApp()

# make plot with a line drawn in
plt = pg.plot([1,5,2,4,3,2], pen='r')

# add an image, scaled
img = pg.ImageItem(np.random.normal(size=(100,100)))
img.scale(0.2, 0.1)
img.setZValue(-100)
plt.addItem(img)
# Gracefully exit the application
sys.exit(app.exec_())