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

### definitions

correlation: operation between image or part of image and operator (kernel) 

kernel - fixed size array coefficients with anchor point in center

morphological operations - operations that process images based on shapes

structuring element - matrix of 0s and 1s - used by morphological operations
origin is the center pixel 
same size and shape as objects u want to process

upsample - scale up
