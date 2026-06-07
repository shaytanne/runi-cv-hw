# %% [markdown]
# # <span style="color:blue"> Computer Vision — Spring 2026
# 
# ## Exercise 3
# 
# **Deadline:** June 7, 23:55. \
# If you need more time, please send a private Piazza message letting me know that you'll submit late. An extention of up to 1 week is automatically approved.
# 
# ---
# 
# 
# 
# In this exercise, you will practice projection matrices and epipolar geometry related tasks.
# 
# ## Submission guidelines (Note the change in the submission filename!)
# 
# 1. Your submission should include the following files only:
#    - `hw3_Group_<GROUP_NUMBER>_<ID>_<ID>.ipynb` \
#      Replace \<ID\> with your ID number(s). \
#      Replace \<GROUP_NUMBER\> with the your moodle group number.
#    - All image files used in your experiments.
# 3. Use Jupyter Notebook.
# 4. Submit this assignment in pairs (no triplets, singles ok).
#    * One student should submit the homework, and the other should not submit anything.
#    * If you are not the one submitting, make sure that your collaborator indeed submits!
# 
# ## Read the following instructions carefully
# 
# 1. Write **efficient vectorized** code.
# 2. You are responsible for the correctness of your code and should add as many tests as you see fit.
# 3. Use `Python 3` and `NumPy 1.3.2` or above. Before submitting the exercise, restart the kernel and run the notebook from start to finish to make sure everything works.
# 4. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [NumPy](https://www.numpy.org/devdocs/reference/) only. Any other imports are forbidden unless provided by us.
# 5. Your code must run without errors. **Code that fails to run will not be graded.**
# 6. Document your code properly.


# %%
import cv2

# This opens an inteactive figure - use it in part B
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

# This specifies the way plots behave in jupyter notebook
%matplotlib inline
plt.rcParams['figure.figsize'] = (8.0, 8.0) # set default size of plots
plt.rcParams['image.cmap'] = 'gray'


# %%
import platform
print("Python version: ", platform.python_version())
print("Numpy version: ", np.__version__)
print("OpenCV version: ", cv2.__version__)

# %% [markdown]
# ## <span style="color:blue">Section A: Projection
# 
# In this part you will go over projection matrix,  and use them to project 3D points to an image.
# 
# 

# %% [markdown]
# ## <span style="color:blue">Part A1: Projection Matrix 
# Fill the missing values, given partial values of the parameters of the left and right cameras.
# 
# 
# 

# %% [markdown]
# **Right image parameters:**
# The projection matrix of the right image:

# %%
MR = np.array([[1100.504780,          0,   331.023000,   0],
               [0,          1097.763735,   259.386377,   0],
               [0,                    0,            1,   0]])

# %% [markdown]
# The rotation matrix of the right image:

# %%
RR = np.array([[1,0,0],
               [0,1,0],
               [0,0,1]])

# %% [markdown]
# The focal length of the right image:

# %%
fR = 1.0

# %% [markdown]
# From here on, replace "none" with your answers to the questions. In addition, if there are more than a single possible solution, choose one.
# Compute the right image center (principal point):
# 

# %%
OxR = None
OyR = None

# %% [markdown]
# Compute the right image scale factor which is consistent with MR:

# %%
SxR = None
SyR = None

# %% [markdown]
# Compute the right image intrinsic matrix which is consistent with MR:

# %%
MintR = None

# %% [markdown]
#  
# **Left image parameters**
#  
# Left image center (principal point):

# %%
OxL = 320.798101 
OyL = 236.431326

# %% [markdown]
# Scale factor:

# %%
SxL = 1095.671499
SyL = 1094.559584 

# %% [markdown]
# Focal length of the left image: 

# %%
fL = 1

# %% [markdown]
# Translation vector w.r.t. the world origin:

# %%
TL = -np.array([[178.2218,18.8171,-13.7744]]).T

# %% [markdown]
# Rotation matrix of the left image:

# %%
RL = np.array([[ 0.9891,    0.0602,   -0.1346],
               [-0.0590,    0.9982,    0.0134],
               [0.1351,   -0.0053,    0.9908]])

# %% [markdown]
# Compute the intrinsic projection matrix of the left camera: 

# %%
MintL = None 

# %% [markdown]
# Compute the projection matrix of the left camera

# %%
ML = None

# %% [markdown]
# Compute the COP of the left and the right images, in Cartesian coordinates:   
# 
# (You may use the the function *null_space* from *scipy.linalg*) 

# %%
CL = None
CR = None

# %% [markdown]
# Compute the distance between CL and CR:
#     

# %%
D = None 

# %% [markdown]
# ## <span style="color:blue">Part A2: Hands on Triangulation
# 
# Write a function p = proj(M,P) that recieves as input the 3D point P in Euclidean coordinates and a projection matrix M, and outputs the 2D  Euclidean coordinates of the projected point.
# 

# %%
def proj(M,P):
    # your code here
    ...

# %% [markdown]
# **<span style="color:blue">Answer Quesion:**\
# Given object points in the world coordinate system,  $P=(-140,50,1200)$ and $Q=(30,100,2000)$.
# 
# a.	What are the coordinates (Euclidean) of the points in the left camera coordinate system?\
# b.	What are the coordinates (Euclidean) of the points in the right camera coordinate system?
#     
# Note: the camera coordinate system rather than the image coordinate system. (PL means the 3D coordinates in the left **camera** cordinates system, and pL means the 2D coordinates in the left **image** coordinates system.)
#    

# %% [markdown]
# **<span style="color:blue">Your answer:**

# %%
PL = None
PR = None
QL = None
QR = None

# %%
P = np.array([[-140],[50],[1200]])
pL = proj(ML,P)
pR = proj(MR,P)

Q = np.array([[30],[100],[2000]]) 
qL = proj(ML,Q)
qR = proj(MR,Q)

# %% [markdown]
# ### Read two images and display the projections of P and Q on the two given images ###

# %%
imL = cv2.imread('left.tif', cv2.IMREAD_GRAYSCALE)
imR = cv2.imread('right.tif', cv2.IMREAD_GRAYSCALE)
    
plt.rcParams['figure.figsize'] = (14.0, 14.0) 
f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

ax1.imshow(imL, cmap='gray'), ax1.set_title('Left image'), ax1.scatter(pL[0], pL[1], color='r'), \
    ax1.scatter(qL[0],qL[1], color = 'b')
ax2.imshow(imR, cmap='gray'), ax2.set_title('Right image'), ax2.scatter(pR[0], pR[1], color = 'r'), \
    ax2.scatter(qR[0],qR[1], color = 'b')

# %% [markdown]
# **<span style="color:blue"> Answer Question:**\
# Look at the projection of each of the points in the two images. One pair looks as expected, and the other does not. Please give a short explanation of what may have caused it.

# %% [markdown]
# **<span style="color:blue">Your answer:**\
#    ...
#     
#     
#     

# %% [markdown]
# ## <span style="color:blue"> Part B: Epipolar Geometry
# Compute the fundamental matrix F and the epipoles eL and eR of the left and right images, using their projection matrices.\
# Note, you should normalize F by F(3,3) for improved precision.
# 
# For the epipoles' computation use the MR and ML and the Center of projections.
# 
# **<span style="color:blue">Answer Question:**
# Can you double check if they are correct using F? If so, check it.
# 

# %% [markdown]
# **<span style="color:blue">Your answer:**\
#    ...
#     
#     
#     

# %%
eL = None
eR = None
F = None

# %% [markdown]
# ## Epipolar lines ##
# 
# Click on three different points of the **right** image, and check if the epipolar lines on the left image pass through a pixel that corresponds to the one you picked in the right image. Output the set of epipolar lines overlayed on the pair of  images as shown below.
# 
# To do so you can use:
# 1. The code below opens the images in a seperate window. You can click on the right image and  capture the click's coordinates by using the function *plt.ginput*.
# 2. Take each point (this can be done by a loop) and calculate its epipolar line  on the left image using F.
# 3. Compute the two endpoints of the line in the image to plot it on the left image. \
#     **Hint**: you have linear coefficients - (a,b,c). Calculate the y value in the image for x=0, and x=image.width and plot the result.\
#     Use: ax2.plot((x0. xWidth),(yx0, yxWidth))
# 4. Use the set of the points of the right image that you collected, and draw the epipolar lines on the right image.

# %%
# This sould open a new figure window outside of jupyter notebook
%matplotlib qt  

imL = cv2.imread('left.tif', cv2.IMREAD_GRAYSCALE)
imR = cv2.imread('right.tif', cv2.IMREAD_GRAYSCALE)
    
plt.rcParams['figure.figsize'] = (14.0, 14.0) 
f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

ax1.imshow(imL, cmap='gray'), ax1.set_title('Left image')
ax2.imshow(imR, cmap='gray'), ax2.set_title('Right image')

data = plt.ginput(3)

x_val = [x[0] for x in data]
y_val = [x[1] for x in data]

ax2.scatter(x_val, y_val, color='r')

for x in data: 
    # Write your own implementation here.
    pass
    

# %%
%matplotlib inline

# %% [markdown]
# ### This is what you should see:
# ![Epipolar](epipolarLines1.png "Epipolar Lines example")

# %% [markdown]
# ##  <span style="color:blue">Part C : SIFT and RANSAC/LMedS</span>
# 
# **Follow the matching to compute F.**
# 
# Make sure you remember RANSAC, and read a bit about LMedS. Ask yourself: what are the commonalities between the two methods? What are the differences? (Not for submission.)
# 
# Go over this epipolar geometry tutorial:
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html 
# 
# Below, we find the corresponding featues using the SIFT algorithm and match the closet points. The plotted figure showes the best 300 matches.
# 
# ❗️Important: SIFT was a patented algorithm and was thus not distributed with vanilla OpenCV. This changed when the patent expired in 2020. If you are getting errors (e.g., missing `cv2.SIFT_create()`) make sure you are using a recent version of OpenCV.

# %%
imL = cv2.imread('left.tif', cv2.IMREAD_GRAYSCALE)
imR = cv2.imread('right.tif', cv2.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imL, None)
kp2, des2 = sift.detectAndCompute(imR, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
# create FlannBasedMatcher object
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Match descriptors.
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []
matching = []
# Building a list of points screened by ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        matching.append(m)
        
# Sort them in the order of their distance.
matching = sorted(matching, key = lambda x:x.distance)
        
# Draw first 300 matches.
img3 = np.array([])
img3 = cv2.drawMatches(imL, kp1, imR, kp2, matching[:300], outImg = img3, flags=2)

plt.rcParams['figure.figsize'] = (14.0, 14.0) 
f, ((ax1)) = plt.subplots(1, 1, sharex='col', sharey='row')
ax1.imshow(img3, cmap='gray'), ax1.set_title('Matches')

# %% [markdown]
# <span style="color:blue"> Not for submission:</span>
# 
# Look at the obtained results.
# 
# 1. Do you think all matches are correct?
# 2. In which regions of the scene most of the reliable matches were found?
# 3. Try the worst 200 mathces as well (`matching[-200:]`).
# 
# Now, we will use the found matches to compute **F** using `cv2.findFundamentalMat()`.

# %%
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Computing the F matrix
F_calc, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# %%
np.set_printoptions(formatter={'float' : '{:0.7f}'.format})
print(F_calc.T)
print(F)

# %% [markdown]
# And now lets check the computed F_calc:
# 1. Use it to draw the epipolar line as in the example above (change F to F_calc.T)
# 2. Compute the distance between the computed epipoles by F and by F_calc in each of the images.
# 
# You can use `scipy.linalg import null_space`

# %% [markdown]
# **<span style="color:blue">Answer Question:**\
#     Do you see any differences?
#     
# **<span style="color:blue">Your answer:**\
#     ...

# %% [markdown]
# ### <span style="color:blue"> Your part in this section :) ###
# 
# #### Take two pictures using your own camera and compute the epipolar geometry using LMedS ####
# 
# Please submit: 5 corresponding epipolar lines overlayed on   your pair of images.


