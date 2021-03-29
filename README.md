# openCv

### erode and dilate 

morphological operations

applications: remove noise, isolate elements join disparate elements , find intensity of bumps and holes in image 

dilation - makes object larger/thicker

eroosion makes the lines thinner

### smoothing 

blur, gaussianBlur, medianBlu, bilateralFilter

takes neighbors of each pixel and normalizes to the neighbors 

### noise reduction using opening

erode followed by dilate 

for closing, dilate followed by erosion (removes holes within the shape 

### morphological gradient

dilate minus erode, gets outline of objects

### hit or miss transform 

basis of thinning and pruning

merge of two erodes (original image and complement of image)

looks for a single pattern in entire imageå

### extracting horizontal and vertical lines 

uses erode, dilate, and getStructuring element

structuring element of [[1,1,1,1,1,1,1]] to find horizontal lines

structuring element of 
[[1],[1],[1],[1],[1],[1]]

to find vertical lines 

### image pyramids

for up/down sampling (zoom in/out)

gaussian pyramid for downsample
laplacian pyramid for upsampling 

downsampling loses information of image 

### thresholding 

a segmentation method 

application: separate out regions of images corresponding to objects we want to analyze 

in terms of plot of intensities of all pixels, draw a horizontal line thru it
all items below the line get a value and all lines above the line get another value

can also allow one side to be 0 and the rest to remain as original 

### HSV vs RGB

color only uses one channel - easier to process color 

rgb uses 3 channels

### create a filter

linear filter uses filter2D function
filter2D ultimately sharpens the image using Laplacian filtering

### sobel derivatives

used to detect edges - a jump in intensity between pixels

### laplacian 

uses second derivative - finds peaks - also used to detect edges

### canny

even betteer edge detector than the above 2

### hough line 

for detecting straight lines

### hough circle 

For detecting circles 

### histogram equalization

improves contrast in image

makes intensity frequency graph more spread out, giving differing intensities, instead of a single peak of the same intensity

### finding and drawing contours and others

discovers the outlines of shapes in the picture 

convex hull - gets outlines the shapes without concave parts 

boundingRect and minEnclosingCircle for getting bounding box and bounding circles for shapes in image
a.  use canny to get the outlines
b.  find contours from canny output
c.  approximate enclosing polygons using approxPolyDP - applies approximation, boundingRect and minEnclosingCircle uses this output 

### debluring 

PSF - point spread function, e.g. circuar PSF - has one param, radius R - goal - to get estimate of original image 
PSF of linear motion takes in LEN and THETA 

Weiner filter - way to restore blurred image 

SNR - signal to noise ratio 

example application - debluring a license plate that was in motion 

### remove periodic noise 

e.g. dots that are evenly spread throughout the image 

### camera calibration

openCV allows you to calculate distance of 3D geometry from camera

read in 3d textured object model and object mesh 

model registration 
    a.  take image and register as 3d mesh - must provide parameters for camera 
    b.  3d points outputed into YAML format 

model detection 
    a.  estimate object pose given 3d textured model 

### slow motion video effect

1.  vc = VideoCapture(srcFile)
2.  write to four videos with VideoWriter
3.  VideoWriter_fourcc specifies an encoding format 
4.  set the fps of one of the writers to a lower number  
5.  vc.aread() reads a single frame 
6.  video_writer.write writes a single frame 
7.  after the three videos have been written , use those three videos to write to the final fourth video 

### machine learning with open cv 

example: weather
inputs - temperature, cloud density, humidity
output - probability of rain
algorithm - basic baynesian 

1.  simplify the inputs, humidity becomes three levels of humidity , cloud density becomes 3 levels(ranges) as well, weather (rain or shine) becomes two inputs
2.  a.  bayesian algorithm: P = P(W2|G,H) = prob (weather output) given a humidity * probability of weather output / Probability of cloud density * prob of output  given humidity input * probability of the output / probability of the humidity
    b.  the general algorithm: P(A|B) aka probability of A given B = P(B|A) * P(a) / P(b)
    c.  example with numbers:
        i.  given 50% of rainy start cloudy
        ii.  40% of mornings are cloudy
        iii.  10% of days are rainy
        iv.  what is chance of rain?
        v.  p(r|c) = .5 * .1 / .4 = .125
3.  to calculate P（H|W2) and P(G|W2), calculated from past statistics, a large amount of sample data, 10 years of it, lots of work
4.  using 10 years of data, split data into two parts
    a.  feature matrix - accumulates all vectors from the data
    b.  response vector - output as a category (rainy or not)
5.  with a mapping between feature matrix and response vector, you can input today's features, to make a response vector.  then we can use predict() to get the probability of the two categories with a correctness score 
6.  machine learning's main workload:  categorization, regression
7.  we get a ML model to predict outputs for unknowns
8.  iris classificatino example
    a.  load iris data
    b.  import model feeding in data (training data and test data) - splits into feature matrix and response vector (training sample should be slightly larger than testing sample) .6 to .4 split
    c.  use guassian naive bayes 
    d.  fit() used using model, inputing x_train and y_train - no output used, this trains the gnb model
    e.  test() used finally with test data to check accuracy of the model 

9.  so we now have a trained ML model.  now what?  can use predict() method - input being an array, output is a response vector 
10.  applications - predict who will win a football match 
recommendation engines 

### definitions

affine transformation - matrix multiplication followed by vector addition 
used for rotations, translations, scale

correlation: operation between image or part of image and operator (kernel) 

kernel - fixed size array coefficients with anchor point in center

morphological operations - operations that process images based on shapes

structuring element - matrix of 0s and 1s - used by morphological operations
origin is the center pixel 
same size and shape as objects u want to process

upsample - scale up

ML defs

sampling - input data, output data aka labels

dataset - training data, validation data, testing data

features - rows of the input feature matrix 
in terms of images, input could be every pixel, which would have way too many dimensions.  instead, group up inputs into features to reduce the number of dimensions also improves the performance of the model 

model - algorithm plus parameters after training

loss function - used to measure the accuracy of a model
for binary classification, the loss function output is 1 or 0 

optimization function - minimize the loss function by training the parameters more.  e.g. classic gradient descent - most common optimization function - 

gradient descent - from wikipedia - finds local minimum of differentiable function.  takes repeated step in opposite direction of gradient - Haskell Curry studied for use in optimization functions in 1944
gradient - represented with upside down triangle.  vector calculus concept - vector field whose value at point is a vector compmosed of partial derivatives at p å - direction and rate of fastest increase.  points in a field where rate of change is fastest

image moments (image processing) - weighted avg of image pixel intensities 

