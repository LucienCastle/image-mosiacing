# Image-mosiacing

## Abstract
The purpose of this project was to estimate a homography between two images by finding corresponding corners, and use it to create a mosaic between two images. In order to detect corners, the Harris corner detection algorithm was implemented. Next, correspondences were found using Normalized Cross-Correlation (NCC) between images and thresholding output values. After this, the homography from one image to another was estimated by using RANSAC to select the best correspondences and then performing least-squares regression on all inliers. Finally, the images were warped with the homography and overlapped in order to form a mosaic. The results of this project were a successful mosaic of two images, and a panorama of all images.

## Algorithms
### Harris Corner Detection
<img src="https://user-images.githubusercontent.com/47452095/228600945-72dd5604-6ca1-4bd3-91d6-1df873bdcdcf.png" width=20% height=20%>

```
Algorithm HarrisCorner()  {
1. Convert images to grayscale
2. Convolve with sobel mask in x and y direction to obtain dx and dy
3. Obtain elements of Harris matrix using dx*dx, dx*dy, and dy*dy
4. Compute the sum of Ixx, Ixy, and Iyy over a local window using a averaging filter
5. Compute determinant and trace of Harris matrix
6. Compute Harris response and threshold it
}
```
### Non-Maxima Suppression
```
Algorithm nonMaximaSuppression()  {
1. Get the dimensions of the harris response matrix
2. Consider the 8 neighbors and construct a 3x3 window, centered at a pixel
3. If the current pixel has the highest intensity within the window
 3.1 Assign it as a corner
4. Else skip the corner by assigning it 0
5. Return the corner matrix
} 
```
### Finding correspondences
```
Algorithm findCorrespondences(cornersImg1, cornersImg2)  {
1. Retrieve x,y coordinates of corners with non-zero intensities
2. For each non-zero corner i in image1 do
 2.1 Construct a patch1 centered at the corner (pad with zeros if patch goes out of
     bounds) and set max_ncc, best match to -1
 2.2 For each non-zero corner j in image2 do
   2.2.1 Construct a similar patch2 centered at the corner and padded if required 
  2.2.2 Calculate the NCC value of patch1 and patch2
  2.2.3 If the current NCC value is best, set corner j as best match and max_ncc as
        Current ncc value
 2.3 If the max_ncc if greater than the ncc threshold, append i, j to matches array
3. Return the matches
} 
```
### Homography Estimation
```
Algorithm findHomography(srcPts, dstPts)  {
1. Initialize the A matrix
2. For every point in srcPts, dstPts do 
 2.1 Append [-xsrc, -ysrc, -1, 0, 0, 0, xsrc*xdst, ysrc*xdst, xdst]
 2.2 And [0, 0, 0, -xsrc, -ysrc, -1, xsrc*ydst, ysrc*ydst, ydst]
3. Perform SVD on A
4. Get the column of V corresponding to the smallest singular value(Optional: 
   Normalize it)
5. Reshape it into 3x3 matrix and return it
} 

Algorithm estHomographyRANSAC(matches, cornersImg1, cornersImg2)  {
1. For every iteration do
 1.1 Pick randomly 4 correspondences from the matches list
 1.2 Find the homography using these points
 1.3 Initialize array of inliers and for every match in matches do
  1.3.1 Get all the matched corners(x,y) from both source and destination images
  1.3.2 Transform the points from source image using the calculate homography
  1.3.3 Calculate distance between transformed points and the points from the 
        Destination image
  1.3.4 If the distance is greater than threshold consider it inlier and add it to the
  	 List of Inliers
 1.4 If the number of current inliers is maximum, set current inliers as the largest
     Largest set of inliers
2. Get the points from source and destination images from the largest set of inliers
3. Find homography using these points
} 
```

## Results
![res_flow](https://user-images.githubusercontent.com/47452095/228603626-f59f9534-d459-4dea-9c79-ea84bfb7a94b.jpg)

## Future Works
- [ ] Improve the blending/warping algorithm
