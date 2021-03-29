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
2.  bayesian algorithm: P = P(W2|G,H) = prob (weather output) given a humidity * probability of weather output / Probability of cloud density * prob of output  given humidity input * probability of the output / probability of the humidity


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
