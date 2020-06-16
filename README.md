# Classical Automatic Vanishing Point Detection

by running src/vanishingpoint.py you will apply automatic vanishing point detection over images in data folder.

## Algorithmic Details

At this code I have tried to automatically detect Vanishing Points. Basically I have done this task in n steps:

### GrayScaling
To reduce color noise and make it easier to detect edges.

### Reduce Noise
Using GaussianBlur technique with a larger kernel (filter) I have reduced the noise to have smoother image with less edges this way eadge detection could just detect prominent edges and not noises. Also there is another layer of bluring inside Canny Edge Detector.

### Edge Detection
Using Canny edge detection I have extracted edges to Hough Transformer be able to extract lines.

### Hough Transform
By applying Probabilistic Hough Transform we can quickly extract desired shape that in our case are lines. 

### Intersect Calculation
Finding all possible intersect between the lines that Hough Transform detected.

### Select and Shift Intersections
I have selected intersections based on three criteria. 
1. Happens inside our desired region. 
2. Take place by two lines that have a sharp angle to each other. Best result: 0 < theta < 22.5
3. After dividing our image into grids (could be 100), just one intersection per grid is accepted and that intersection was the intersection between two longest lines at that grid.


