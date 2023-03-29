# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
from typing import Tuple

def read_imgs(
    root_dir:str, # path to images
    scale_factor:int=0.9, # scaling factor for images
)->np.ndarray:
    '''
    reads images, scales them and returns a array of images
    '''
    img_paths = os.listdir(root_dir)
    img_paths.sort()

    imgs = []
    for img_path in img_paths:
      # read image
      img = cv2.imread(os.path.join(root_dir, img_path))

      # Resize images if they are large
      img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
      imgs.append(img)

    imgs = np.stack(imgs, axis=0)
    return imgs

def harris_corner_detector(
    img:np.ndarray, # image for corner detection 
    filter_size:int=3, # size of the averaging filter 
    ksize:int=3, # size of the sobel operator 
    k:float=0.04, # k values to determine the corner response 
    threshold=0.01, # threshold of corner responses
)->np.ndarray:
    '''
    converts the image to grayscale, applies sobel operator,
    calculates Ixx, Ixy, Iyy, applies box/gaussian filter
    and returns thresholded harris response matrix
    '''
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Compute x and y derivatives of the image using Sobel operator
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute elements of the Harris matrix M
    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2

    # Compute the sum of Ixx, Ixy, and Iyy over a local window using a box filter
    weight = np.ones((filter_size, filter_size)) / (filter_size**2)
    Sxx = cv2.filter2D(Ixx, -1, weight)
    Sxy = cv2.filter2D(Ixy, -1, weight)
    Syy = cv2.filter2D(Iyy, -1, weight)

    # Compute the determinant and trace of M
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    # Compute the Harris response
    harris_response = det - k * (trace ** 2)

    # Threshold the Harris response
    harris_response[harris_response < threshold * harris_response.max()] = 0

    return harris_response

def non_maxima_supression(
    harris_response_mat:np.ndarray, # harris corner response matrix
)->np.ndarray:
    '''
    applies non-maxima suppression to 
    reduce multiple/redundunt detected corners
    '''
    # Non-maximum suppression
    height, width = harris_response_mat.shape

    corners = np.zeros_like(harris_response_mat)

    for i in range(1, height-1):
      for j in range(1, width-1):
        window = harris_response_mat[i-1:i+2, j-1:j+2]
        if harris_response_mat[i, j] == np.max(window):
          corners[i, j] = np.max(window)

    return corners

def get_corners(
    imgs:np.ndarray, # images detect corners
    filter_size:int=7, # size of the averaging filter 
    ksize:int=7, # size of the sobel operator 
    k:float=0.04, # k values to determine the corner response 
    response_thresh=0.01, # threshold of corner responses
    visualize:bool=False, # visualizes detected corners on the original images
)->np.ndarray:
    '''
    Read images, perform harris corner detection
    and Non-maxima suppresion
    '''
    harris_responses = []

    # get the harris response matrix for each image
    for img in imgs:
      harris_responses.append(harris_corner_detector(img, 
                                                     filter_size=filter_size,
                                                     ksize=ksize,
                                                     k=k,
                                                     threshold=response_thresh))

    harris_responses = np.stack(harris_responses, axis=0)

    nms_corners = []

    # perform NMS on corners detected in every image
    for idx, harris_response in enumerate(harris_responses):
      nms_corner = non_maxima_supression(harris_response)
      
      # draw the corners after NMS
      if visualize:
        img = imgs[idx].copy()
        (X, Y) = np.where(nms_corner > 0)
        
        for i,j in zip(X,Y):
          cv2.circle(img, (j, i), 2, (0, 0, 255), -1)
        cv2_imshow(img)

      nms_corners.append(nms_corner)

    nms_corners = np.stack(nms_corners, axis=0)
    return nms_corners

def get_ncc(
    patch1:np.ndarray, # patch 1
    patch2:np.ndarray, # patch 2
)->float:
    '''
    Calculates NCC between the patches
    '''
    patch1 = patch1 - np.mean(patch1)
    patch2 = patch2 - np.mean(patch2)

    ncc = np.sum(patch1 * patch2) / (np.sqrt(np.sum(patch1**2)) * (np.sqrt(np.sum(patch2**2))))

    return ncc

def find_correspondences(
    corners1:np.ndarray, # corner features of image1
    corners2:np.ndarray, # corner features of image2
    img1:np.ndarray, # image 1 
    img2:np.ndarray, # image 2 
    patch_size:int=15, # size of the template for matching
    threshold:float=0.95, # threshold for NCC value
)->Tuple[list, np.ndarray, np.ndarray]:
    '''
    performs feature matching using the detected corners by calculate 
    NMS between patches
    '''
    matches = []

    (X1, Y1) = np.where(corners1 > 0)
    (X2, Y2) = np.where(corners2 > 0)
    patch_half_size = patch_size // 2

    for idx, (x1, y1) in enumerate(zip(X1, Y1)):
      x1min = max(x1 - patch_half_size, 0)
      x1max = min(x1 + patch_half_size + 1, img1.shape[0])
      y1min = max(y1 - patch_half_size, 0)
      y1max = min(y1 + patch_half_size + 1, img1.shape[1])
      
      # create patch centered at a corner in img1
      # pad with zero to handle out of bounds error
      patch1 = np.zeros((patch_size, patch_size), dtype=img1.dtype)
      patch1[x1min - x1 + patch_half_size:x1max - x1 + patch_half_size,
             y1min - y1 + patch_half_size:y1max - y1 + patch_half_size] = img1[x1min:x1max, y1min:y1max, ]
      
      max_ncc = -1
      best_match = -1
      for jdx, (x2, y2) in enumerate(zip(X2, Y2)):
        x2min = max(x2 - patch_half_size, 0)
        x2max = min(x2 + patch_half_size + 1, img2.shape[0])
        y2min = max(y2 - patch_half_size, 0)
        y2max = min(y2 + patch_half_size + 1, img2.shape[1])

        # create patch centered at a corner in img2
        # pad with zero to handle out of bounds error
        patch2 = np.zeros((patch_size, patch_size), dtype=img2.dtype)
        
        patch2[x2min - x2 + patch_half_size:x2max - x2 + patch_half_size,
               y2min - y2 + patch_half_size:y2max - y2 + patch_half_size] = img2[x2min:x2max, y2min:y2max]

        # calculate ncc between patches
        ncc_val = get_ncc(patch1, patch2)

        if ncc_val > max_ncc:
          max_ncc = ncc_val
          best_match = jdx
      
      # save the best matches patch i.e. indices of the corner
      # at which the patches were centered
      if max_ncc > threshold:
        matches.append((idx, best_match))

    return matches, np.stack([X1.astype(int), Y1.astype(int)], axis=1), np.stack([X2.astype(int), Y2.astype(int)], axis=1)

def find_homography(
    src_pts:np.ndarray, # source points 
    dst_pts:np.ndarray, # destination points
)->np.ndarray:
    '''
    Calculates homography between src and dst points
    '''
    n = src_pts.shape[0]
    A = []

    for i in range(n):
      x, y = src_pts[i]
      x_dash, y_dash = dst_pts[i]

      A.append([-x, -y, -1, 0, 0, 0, x*x_dash, y*x_dash, x_dash])
      A.append([0, 0, 0, -x, -y, -1, x*y_dash, y*y_dash, y_dash])

    A = np.array(A)

    U, S, VT = np.linalg.svd(A)
    h = VT[-1, :]/VT[-1, -1]
    H = np.reshape(h, (3, 3))

    return H

def est_homography_ransac(
    matches:list, # matched indices of corners 
    c1:np.ndarray, # corners of image 1 
    c2:np.ndarray, # corners of image 2 
    threshold:float=5, # distance threshold 
    max_iterations:int=1000, # maximum number of iterations
)->Tuple[np.ndarray, list]:
    '''
    Estimate homography between 2 images using correspondences
    by applying RANSAC
    '''
    best_homography = None
    largest_num_inliers = 0
    max_inliers = []

    for iters in range(max_iterations):
      # randomly select 4 correspondences
      rand_mat_idx = np.random.choice(len(matches), 4, replace=True)

      src_pts = np.array([c1[matches[idx][0]] for idx in rand_mat_idx]).astype(np.float32)
      dst_pts = np.array([c2[matches[idx][1]] for idx in rand_mat_idx]).astype(np.float32)

      # estimate homography
      homography = find_homography(src_pts, dst_pts)

      # counting inlier points
      inliers = []
      for j in range(len(matches)):
          src_pt = np.float32(c1[matches[j][0]]).reshape(-1, 1, 2)
          dst_pt = np.float32(c2[matches[j][1]]).reshape(-1, 1, 2)
          pred_dst_pt = cv2.perspectiveTransform(src_pt, homography)
          dist = np.linalg.norm(dst_pt - pred_dst_pt)
          if dist < threshold:
            inliers.append(j)

      # update max set of inliers
      if len(inliers) > len(max_inliers):
          max_inliers = inliers
        
    xy = np.array([c1[matches[idx][0]] for idx in max_inliers])
    xy_dash = np.array([c2[matches[idx][1]] for idx in max_inliers])
    

    H = find_homography(xy, xy_dash)
    
    return H, max_inliers

def get_all_homography(
    imgs:np.ndarray, # Images 
    corners:list, # Detected corners in the images
    patch_size:int=15, # patch size for feature matching
    ncc_thresh:float=0.95, # threshold for ncc
    dist_thresh:float=5, # distance threshold for RANSAC
    max_ransac_iters:int=1000, # maximum iterations for RANSAC
    viz_correspondences:bool=False, # If true, visualizes the correspondences
)->dict:
  '''
  Estimate homography between successive images
  '''
  img_id = 0
  H_all = {}

  while img_id < len(imgs)-1:
    # find correspondences
    matches, c1, c2 = find_correspondences(
        corners[img_id], 
        corners[img_id+1], 
        cv2.cvtColor(imgs[img_id], cv2.COLOR_BGR2GRAY), 
        cv2.cvtColor(imgs[img_id+1], cv2.COLOR_BGR2GRAY),
        patch_size=patch_size,
        threshold=ncc_thresh
    )

    # visualize correspondences
    if viz_correspondences:
      img1 = imgs[img_id].copy()
      img2 = imgs[img_id+1].copy()

      h1, w1, _ = img1.shape
      h2, w2, _ = img2.shape
          
      out = np.zeros((max([h1, h2]), w1+w2, 3), dtype='uint8')

      # Place the first image to the left
      out[:h1, :w1, :] = img1

      # Place the next image to the right of it
      out[:h2, w1:w1+w2, :] = img2

      for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat[0]
        img2_idx = mat[1]

        (y1, x1) = c1[img1_idx]
        (y2, x2) = c2[img2_idx]

        # draw the matching features on the corresponding images
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), -1)   
        cv2.circle(out, (int(x2)+w1,int(y2)), 4, (0, 0, 255), -1)

        cv2.line(out, (int(x1),int(y1)), (int(x2)+w1,int(y2)), (255, 0, 0), 1)

      cv2_imshow(out)

    # estimate the homography using RANSAC
    homography, inliers = est_homography_ransac(matches, c1, c2, threshold=dist_thresh, max_iterations=max_ransac_iters)
    H_all['{}{}'.format(img_id, img_id+1)] = homography

    img_id += 1
  
  return H_all

def stitch_images(
    imgs:np.ndarray, # images for panorama
    H_all:dict, # homographies between successive images
)->Tuple[np.ndarray, dict]:
  '''
  Warps and create panorama from the given images
  and homographies
  '''
  H = np.identity(3)
  iter = 0
  H_all_copy = H_all.copy()

  n = len(imgs)
  mid = n//2

  # multiply homographies to calculate H between 1st and any image
  for key, homography in H_all.items():
    iter += 1
    H = np.matmul(H, homography)
    H_all_copy['{}{}'.format(0, iter)] = H

  for iter in range(n):
    H_all_copy['{}{}'.format(iter, iter)] = np.identity(3)

  for iter in range(mid, n):
    if '{}{}'.format(mid, iter) in H_all_copy.keys():
      continue
    else:
      H_all_copy['{}{}'.format(mid, iter)] = np.matmul(H_all_copy['{}{}'.format(mid, iter-1)], H_all_copy['{}{}'.format(iter-1, iter)])

  # Get corners of first image
  points0 = np.array(
        [[0, 0], [imgs[0].shape[0], 0], [imgs[0].shape[0], imgs[0].shape[1]], [0, imgs[0].shape[1]]],
        dtype=np.float32)
  points0 = points0.reshape((-1, 1, 2))
  prev_points = {}
  prev_points['pts0'] = points0

  # transform corners of images using H
  for i in range(1, len(imgs)):
    points = np.array(
        [[0, 0], [imgs[i].shape[0], 0], [imgs[i].shape[0], imgs[i].shape[1]], [0, imgs[i].shape[1]]],
        dtype=np.float32)
    points = points.reshape((-1, 1, 2))

    points = cv2.perspectiveTransform(points, np.linalg.inv(H_all_copy['{}{}'.format(i-1, i)]))
    prev_points['pts{}'.format(i)] = points

  # calculate the size of output panorama
  all_pts = [pts for key, pts in prev_points.items()]
  all_pts = np.concatenate(all_pts, axis=0)

  [x_min, y_min] = (all_pts.min(axis=0).ravel() - 0.5).astype(np.int32)
  [x_max, y_max] = (all_pts.max(axis=0).ravel() + 0.5).astype(np.int32)

  output_img = np.zeros(( x_max - x_min, y_max - y_min, 3))

  # transform and paste the images on the left
  # using homography
  for i in range(mid):
    position = prev_points['pts{}'.format(i+1)]
    img_init = imgs[i]

    h_translation = np.array([[1, 0, position.max(axis=0)[0,0]-img_init.shape[0]], [0, 1, (position.min(axis=0)[0,1])], [0, 0, 1]])

    warped_img = cv2.warpPerspective(np.transpose(img_init, (1, 0, 2)),  h_translation.dot(H_all_copy['{}{}'.format(i,mid)]),
                                     ( x_max - x_min, y_max - y_min,))
    left_img = np.transpose(warped_img, (1, 0, 2))

    for i in range(output_img.shape[0]):
      for j in range(output_img.shape[1]):
        output_img[i, j, ] = output_img[i, j, :] if all(output_img[i, j, ]) > 0 else left_img[i, j, ]
  
    # cv2_imshow(output_img)
  
  # paste the middle image as it is and transform
  # the images on its left by inverse homography
  for i in range(mid,n):
    position = prev_points['pts{}'.format(i)]
    img_init = imgs[i]

    h_translation = np.array([[1, 0, position.max(axis=0)[0,0]-img_init.shape[0]], [0, 1, position.min(axis=0)[0,1]], [0, 0, 1]])

    H = np.linalg.inv(H_all_copy['{}{}'.format(mid,i)])

    warped_img = cv2.warpPerspective(np.transpose(img_init, (1, 0, 2)),  h_translation.dot(H),
                                     ( x_max - x_min, y_max - y_min,))
    right_img = np.transpose(warped_img, (1, 0, 2))

    for i in range(output_img.shape[0]):
      for j in range(output_img.shape[1]):
        output_img[i, j, ] = output_img[i, j, :] if all(output_img[i, j, ]) > 0 else right_img[i, j, ]
    
    # cv2_imshow(output_img)
  
  panorama = output_img

  return panorama, H_all_copy

def warp_two_images(
    left_img_path:str, # path of the left image 
    right_img_path:str, # path of the right image 
    scale_factor:int=0.9, # scaling factor
)->np.ndarray:
  '''
  Reads 2 images, left and right, perform feature matching using
  Harris corner detection and uses RANSAC to estimate best 
  homoraphy between the two, and finally warps them.
  '''
  # read image
  left_img = cv2.imread(left_img_path)
  right_img = cv2.imread(right_img_path)

  # Resize images if they are large
  left_img = cv2.resize(left_img, None, fx=scale_factor, fy=scale_factor)
  right_img = cv2.resize(right_img, None, fx=scale_factor, fy=scale_factor)

  imgs = np.stack([left_img, right_img], axis=0)

  # detect corner features
  nms_corners = get_corners(imgs, visualize=True)

  # estimate the homography
  H_all = get_all_homography(imgs, nms_corners, viz_correspondences=True)

  img1_id, img2_id = 0, 1

  img1 = imgs[img1_id].copy()
  img2 = imgs[img2_id].copy()

  # calculate inverse homography
  xh = np.linalg.inv(H_all['{}{}'.format(img1_id, img2_id)])

  # calculate the size of the output by takinng corners of the image boundary
  # and transforming them
  points0 = np.array(
          [[0, 0], [img1.shape[0], 0], [img1.shape[0], img1.shape[1]], [0, img1.shape[1]]],
          dtype=np.float32)
  points0 = points0.reshape((-1, 1, 2))
  points1 = np.array(
          [[0, 0], [img2.shape[0], 0], [img2.shape[0], img2.shape[1]], [0, img2.shape[1]]],
          dtype=np.float32)
  points1 = points1.reshape((-1, 1, 2))

  points2 = cv2.perspectiveTransform(points1, xh)
  points = np.concatenate((points0, points2), axis=0)

  [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(np.int32)
  [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(np.int32)

  # estimate the correct location of the warped image on the output canvas
  h_translation = np.array([[1, 0, x_max-img2.shape[0]], [0, 1, y_min], [0, 0, 1]])

  output_img = cv2.warpPerspective(np.transpose(img2, (1, 0, 2)), h_translation.dot(xh),
                                      (x_max - x_min, y_max - y_min, ))
  output_img = np.transpose(output_img, (1, 0, 2))

  # position the left image on the canvas
  h_translationn = np.array([[1, 0, points2.max(axis=0)[0, 0]-img1.shape[0]], [0, 1, (points2).min(axis=0)[0, 1]], [0, 0, 1]])
  imggg = cv2.warpPerspective(np.transpose(img1, (1, 0, 2)),  h_translationn.dot(H_all['{}{}'.format(img1_id, img2_id)]),
                                      ( x_max - x_min, y_max - y_min,))
  imggg = np.transpose(imggg, (1, 0, 2))

  for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
      output_img[i, j, ] = output_img[i, j, :] if any(output_img[i, j, ]) > 0 else imggg[i, j, ]

  return output_img

if __name__=='__main__':
  path_to_images = input('Enter path to image folder to create panorama')

  # read images
  imgs = read_imgs(path_to_images, scale_factor=0.9)

  # Uncomment if the order of images in the folder isnt from left to right
  # imgs = np.flip(imgs, 0)

  # detect corners
  nms_corners = get_corners(imgs, visualize=True)

  # estimate homography
  H_all = get_all_homography(imgs, nms_corners, viz_correspondences=False)

  # stitch together images to create panorama
  stitched_img, H_all_copy = stitch_images(imgs, H_all)
  cv2_imshow(stitched_img)

  # '''
  # To perform warping on 2 images
  # '''
  left_img = input("Enter path of left image: ")
  right_img = input("Enter path of right image: ")

  warped_images = warp_two_images(left_img, right_img)
  cv2_imshow(warped_images)

