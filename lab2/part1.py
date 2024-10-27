#!/usr/bin/env python
# coding: utf-8

# In[16]:


from scipy.stats import multivariate_normal
import cv2
import scipy
import cv23_lab2_2_utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label,map_coordinates,convolve1d, rank_filter
from numpy import unravel_index
import math
import glob
import base64
from numpy.linalg import norm
import scipy.io
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from numpy import linalg as LA
from pathlib import Path
from scipy.stats import multivariate_normal
import cv2
import scipy
import cv23_lab2_2_utils as utils
import numpy as np
import matplotlib.patches as patches
from scipy.ndimage import label,map_coordinates,convolve1d, rank_filter
from numpy import unravel_index
import math
import glob
import base64
from numpy.linalg import norm
import scipy.io
import os
get_ipython().run_line_magic('matplotlib', 'inline')


def GaussKernel(sigma):
    n = int(np.ceil(3*sigma)*2+1)
    
    k = cv2.getGaussianKernel(n, sigma) 
    kernel = k @ k.T
    
    kernel /= np.sum(kernel) #normalize
    return kernel


def gaussian_kernel(sigma):
    n = int(np.ceil(3*sigma)*2+1)
    
    k = cv2.getGaussianKernel(n, sigma) 
    kernel = k @ k.T
    
    kernel /= np.sum(kernel) #normalize
    return kernel



def disk_strel(n):
    '''
        Return a structural element, which is a disk of radius n.
    '''
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)


# In[17]:


# Φορτωση ΜΑΤ αρχειου και μετατροπη rbg εικονας σε YCrCb για την ανιχνευση
skin_mat = scipy.io.loadmat('cv23_lab2_material/part1 - GreekSignLanguage/skinSamplesRGB.mat') 
skin_rgb =  skin_mat['skinSamplesRGB']
skin_ycrcb = cv2.cvtColor(skin_rgb, cv2.COLOR_RGB2YCrCb)
#απομονωση cr και cb 
cr = skin_ycrcb[:, :, 2]
cb = skin_ycrcb[:, :, 1]
#το μ βρισκεται απο τις μεσες τιμες των cr, cb
mu = [np.mean(cb), np.mean(cr)]
cov = np.array(np.cov(cb.flatten(), cr.flatten()))


# In[18]:


def fd(I,mu,cov):                          
   

    #μετατροπη σε YCrCb
    i_ycrcb = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)
    
    #απομονωση συνιστωσων
    cr = i_ycrcb[:, :, 2]
    cb = i_ycrcb[:, :, 1]
    

    ar = np.array([cb, cr]).T
    skin_pr = multivariate_normal.pdf(ar, mu, cov)
    skin_pr=skin_pr.T
    
    #κραταω μονο τις τιμες που περνανε το κατωφλι 0.0005 και τις θετω ολες ισες με 0.25 για ομοιομορφια
    thresholded_skin = np.where(skin_pr > 0.0005, skin_pr, 0)
    plt.imshow(thresholded_skin)
    
    
    #ανοιγμα και γεμισμα οπων:
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening_result = cv2.morphologyEx(thresholded_skin, cv2.MORPH_OPEN, kernel_1)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closing_result = cv2.morphologyEx(opening_result, cv2.MORPH_CLOSE, kernel_2)
    skin = closing_result
    #plt.imshow(closing_result, cmap='gray')
    
    if (plot):                                 #plot skin before opening, after opening and after closing
        fig,ax = plt.subplots(1, 3, figsize = (20, 20))

        ax[0].set_title('Skin image before morphological filtering',fontsize=16)
        ax[0].imshow(thresholded_skin)

        ax[1].set_title('Skin image after opening',fontsize=16)
        ax[1].imshow(opening_result)
        
        ax[2].set_title('Skin image after opening, closing',fontsize=16)
        ax[2].imshow(skin)
        
    labeled_array, num_features = label(skin)              
    
    regions=[]                                          
    for i in range(num_features):
        region = np.argwhere(labeled_array == i + 1)
        
        up_left_x = np.min(region[:,0])                    
        up_left_y = np.min(region[:,1])
        
        
        up_right_x = np.min(region[:,0])                     
        up_right_y = np.max(region[:,1])
        
        down_left_x = np.max(region[:,0])                    
        down_left_y = np.min(region[:,1])
        
        down_right_x = np.max(region[:,0])                   
        down_right_y = np.max(region[:,1])
        
        width = up_right_y-up_left_y + 1
        height = down_right_x - up_right_x + 1
        
        regions.append([up_left_x, up_left_y, width, height])
    
    return np.array(regions)     


I = cv2.imread('cv23_lab2_material/part1 - GreekSignLanguage/{}.png'.format(1))
#μετατροπη εικονας σε RBG
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)                           
plot = True                    

#κληση συναρτησης
skin_detection = fd(I, mu, cov) 




# In[19]:


#πλοταρουμε την εικονα και τα παραλληλογραμμα πανω της
I = cv2.imread("cv23_lab2_material/part1 - GreekSignLanguage/1.png")     
  
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)                            
plot = True                    
boundingBox=fd(I, mu, cov)    


#kefali
head_x, head_y, height, width = boundingBox[0, :]
rect_head = patches.Rectangle((head_y, head_x), height, width, linewidth=1, edgecolor='r', facecolor='none')

#aristero xeri
left_hand_x, left_hand_y, height, width=boundingBox[1,:]
rect_left = patches.Rectangle((left_hand_y,left_hand_x), height, width, linewidth=1, edgecolor='g', facecolor='none')

# deksi xeri
right_hand_x, right_hand_y, height, width = boundingBox[2,:]
rect_right = patches.Rectangle((right_hand_y, right_hand_x), height, width, linewidth=1, edgecolor='b', facecolor='none')

fig, ax=plt.subplots(figsize=(7,7))
ax.set_title('Image with skin areas shown', fontsize=16)
                               

ax.add_patch(rect_head)
ax.add_patch(rect_left)
ax.add_patch(rect_right)

ax.imshow(I)


# In[20]:


#πραγματικες συντεταγμενες των ορθογωνιων

face = [154, 102, 67, 115]
left = [93, 272, 56, 83]
right = [201, 270, 56, 83]


# In[21]:


def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    
    
#     dx_final = []
#     dy_final = []
    dx_final = np.zeros(np.shape(I1))
    dy_final = np.zeros(np.shape(I1))
    
    
    for f in features:
        
        x, y = f[0][0], f[0][1]
        print("x = ", x)
        print("y = ", y)

#         if x <= 1 or y <=1:
#             continue

#         I1 = I1_[max(fx-2,0):min(fx+2+1,np.shape(I1_)[0]), max(fy-2,0):min(fy+2+1,np.shape(I1_)[1])]
#         I2 = I2_[max(fx-2,0):min(fx+2+1,np.shape(I2_)[0]), max(fy-2,0):min(fy+2+1,np.shape(I2_)[1])]
        w_1 = I1[max(x-2,0):min(x+2+1,np.shape(I1)[0]), max(y-2,0):min(y+2+1,np.shape(I1)[1])]
        w_2 =I2[max(x-2,0):min(x+2+1,np.shape(I2)[0]), max(y-2,0):min(y+2+1,np.shape(I2)[1])]
#         print("w_1 = ", w_1)
#         print("w_2 = ", w_2)

        thr = 0.001
        dx = d_x0
        dy = d_y0

        diff = 100
        j = 0
        #          print("j = ", j)

        while j < 1000 and abs(diff) > abs(thr):
            j = j + 1



            x0, y0 = np.meshgrid(np.arange(w_1.shape[1]), np.arange(w_1.shape[0]))

#             print("y0 = ")
#             print(y0)
#             print("y0 shape = ", y0.shape)

        #             print("dy = ")
        #             print(dy)
        #             print("dy shape = ", dy.shape)

#             print("x0 = ")
#             print(x0)
#             print("x0 shape = ", x0.shape)

        #             print("dx = ")
        #             print(dx)
        #             print("dx shape = ", dx.shape)
        #             print(" i = ", i)
        #             i = i +1


        #           interp_values = scipy.ndimage.map_coordinates(w_1, [y0 + dy, x0 + dx], order = 1)

            gradient = np.gradient(w_1)
            w_1_y = gradient[0]
            w_1_x = gradient[1]


            partial_x = scipy.ndimage.map_coordinates(w_1_x, [np.ravel(y0+dy), np.ravel(x0+dx)], order=1).reshape(w_1.shape)
            partial_y = scipy.ndimage.map_coordinates(w_1_y, [np.ravel(y0+dy), np.ravel(x0+dx)], order=1).reshape(w_1.shape)







            interp_values = scipy.ndimage.map_coordinates(w_1,[np.ravel(y0+dy), np.ravel(x0+dx)], order=1) 

            In_minus_1 = interp_values.reshape(w_1.shape)

#             print("interp_values=", interp_values)

#             print("hi1")


        #             interp_values_x1 = map_coordinates(w_1, [y0 + dy, x0 + dx - 1], order=1)
        #             interp_values_x2 = map_coordinates(w_1, [y0 + dy, x0 + dx + 1], order=1)
        #             partial_x_In_minus_1 = (interp_values_x2 - interp_values_x1) / 2
        #           

            #partial_y_In_minus_1, partial_x_In_minus_1 = np.gradient(In_minus_1)

            A1 = partial_x

#             print("A1=", A1)

#             print("hi2")

        #             interp_values_y1 = map_coordinates(w_1, [y0 + dy - 1, x0 + dx], order=1)
        #             interp_values_y2 = map_coordinates(w_1, [y0 + dy + 1, x0 + dx], order=1)
        #             partial_y_In_minus_1 = (interp_values_y2 - interp_values_y1) / 2
            A2 = partial_y


#             print("hi3")

            E = w_2 - In_minus_1
            G2D = GaussKernel(rho)

#             print("hi4")

            a11 = cv2.filter2D(A1**2, -1, G2D) + epsilon
            a_11 = a11[2][2]
            a12 = cv2.filter2D(A1*A2, -1, G2D)
            a_12 = a12[2][2]
            a21 = a12
            a_21 = a_12
            a22 = cv2.filter2D(A2**2, -1, G2D) + epsilon
            a_22 = a22[2][2] 


#             print("a11=", a11)
#             print("a22 = ", a12)
#             print("a22 = ", a22)


            # *****calculating the inverse matrix*******
            det = a_11*a_22 - a_12*a_21
            inv_11 = a_22 / det
            inv_12 = -a_21 / det
            inv_21 = -a_12 / det
            inv_22 = a_11 / det
            # ***end of calculation of the inverse matrix*****

#             print("hi6")

            b1 = cv2.filter2D(A1*E, -1, G2D)
            b_1 = b1[2][2]
            b2 = cv2.filter2D(A2*E, -1, G2D)
            b_2 = b2[2][2]
#             print("b1 = ", b1)
            u_1 = inv_11 * b_1 + inv_12 * b_2
            u_2 = inv_21 * b_1 + inv_22 * b_2

            dx_pr = dx
            dy_pr = dy

#             print("hi7")
            dx = dx + u_1
            dy = dy + u_2

#             print("hi8")
            diff = np.linalg.norm([u_1,u_2])
#             print("diff - thr = ", diff-thr, "and j = ", j)

            print("u_1 = ", u_1, "and u_2 = ", u_2)


        dx_final[x, y] = dx
        dy_final[x, y] = dy



    return dx_final, dy_final




# In[22]:


box = [93, 272, 56, 83]

image_path = "cv23_lab2_material/part1 - GreekSignLanguage/1.png"
I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#I1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
I1 = I[272:272+83, 93:93+56]

shift=5
I2 = I[272+shift:272+83+shift, 93:93+56]

plt.imshow(I1)
plt.figure()
plt.imshow(I2)

rho = 5
epsilon = 0.05
dx_0, dy_0 = 0,0

# Exract features to track movement
corners = cv2.goodFeaturesToTrack(I1.astype('uint8'),15,0.01,10)
corners = np.int0(corners)

dx, dy = lk(I1, I2, corners, 5, 0.05, 0, 0)

plt.figure()
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal') 
plt.quiver(-dy,-dx,angles='xy',scale=100)


box = [93, 272, 56, 83]


# In[23]:


image_path = "cv23_lab2_material/part1 - GreekSignLanguage/1.png"
I1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

I1 = I1[272:272+83, 93:93+56]

image_path = "cv23_lab2_material/part1 - GreekSignLanguage/2.png"
I2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
I2 = I2[272:272+83, 93:93+56]

plt.imshow(I1)
plt.figure()
plt.imshow(I2)


# In[24]:


rho = 5
epsilon = 0.05
dx_0, dy_0 = 0,0

# Σημεια ενδιαφεροντος
features = cv2.goodFeaturesToTrack(I1.astype('uint8'),15,0.01,10)
features = np.int0(features)

dx, dy = lk2(I1, I2, features, rho, epsilon, dx_0, dy_0)


# In[28]:


frame1 = cv2.imread('cv23_lab2_material/part1 - GreekSignLanguage/1.png')
frame2 = cv2.imread('cv23_lab2_material/part1 - GreekSignLanguage/2.png')

hi, psi, width, height = 93, 272, 60, 87

im1 = frame1[psi : psi + height, hi: hi + width]
im2 = frame2[psi + 5 : psi + height + 5, hi: hi + width]
gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
features = cv2.goodFeaturesToTrack(gray, 15, 0.15, 3)
print("features shape =", features.shape)


features_x = features[:, 0, 0]
features_y = features[:, 0, 1]

im_copy = np.copy(im2)

for i in features:
    x, y = i.ravel()
    cv2.circle(im_copy, (x, y), 1, 255, -1)

plt.imshow(im_copy)
plt.title('Image corners with Shi Thomasi Detector')
plt.show()


# In[29]:


# Calculate optical flow with cv2 function
oflow = cv2.DualTVL1OpticalFlow_create(nscales=1)
flow = oflow.calc(I1, I2, None)

features = cv2.goodFeaturesToTrack(I1.astype('uint8'),15,0.01,10)
features = np.int0(features)


# dx, dy mono gia ta deatures
dxs = []
dys = []
# to kratw gia to plot meta
fxs = []
fys = []

for feature in features:
    fy = feature[0][0]
    fx = feature[0][1]
    
    
    
for feature in features:
    fy = feature[0][0]
    fx = feature[0][1]
    fxs.append(fx)
    fys.append(fy)
   
    dxs.append(dx[fx, fy])
    dys.append(dy[fx, fy])


# In[30]:


dxs = np.zeros(np.shape(I1))
dys = np.zeros(np.shape(I2))

for i in range(np.shape(I1)[0]):
    for j in range(np.shape(I1)[1]):
        if (i,j) in zip(fxs, fys):
            dxs[i,j] = dx[i,j]
            dys[i,j] = dy[i,j]
        else:
            flow[i, j, 0]=0
            flow[i, j, 1]=0
            
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal') 
plt.quiver(-1*dxs,-1*dys,angles='xy',scale=100)


plt.figure()
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal') 
plt.quiver(-20*flow[:,:,0], -20*flow[:,:,1], angles='xy', scale=100)

plt.figure()
plt.imshow(I1)


# In[31]:


def displ(d_x, d_y):
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    
    threshold = 0.5
    d = d_x**2 + d_y**2
        
    cond = (d >= threshold*np.max(d)).reshape(d_x.shape)
    
    dx = cond*d_x
    dy = cond*d_y
    
    dx_mean = np.sum(dx)/np.sum(cond)
    dy_mean = np.sum(dy)/np.sum(cond)
    
    return dx_mean, dy_mean
######


# In[37]:


def GaussKernel(sigma):
    n = int(np.ceil(3*sigma)*2+1)
    
    k = cv2.getGaussianKernel(n, sigma) 
    kernel = k @ k.T
    
    kernel /= np.sum(kernel) #normalize
    return kernel


def read_image(name):
    img1 = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    #Normalizing to 0.0-1.0 float values
    img1=img1.astype(np.float)/255
    return img1


def compute_optical_flow(pyramid_1, pyramid_2, feature_pyramid, rho, epsilon, dx_previous, dy_previous, scale):
    dx_current = dx_previous
    dy_current = dy_previous
    
    for i in range(scale - 1, -1, -1):
        dx_current *= 2
        dy_current *= 2
        dx_current = np.resize(dx_current, pyramid_1[i].shape)
        dy_current = np.resize(dy_current, pyramid_2[i].shape)
        dx_current, dy_current = lk(pyramid_1[i], pyramid_2[i], feature_pyramid[i], rho, epsilon, dx_current, dy_current)
    
    return dx_current, dy_current



def lk_multi(I1, I2, features, rho=3, epsilon=0.05, dx_initial=None, dy_initial=None, scale=3):
    kernel_size = 3
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, rho)
    gaussian_filter = gaussian_kernel @ gaussian_kernel.T
    
    pyramid_1 = [I1]
    pyramid_2 = [I2]
    feature_pyramid = [features]

    for i in range(scale - 1):
        I1_filtered = cv2.filter2D(pyramid_1[i], -1, gaussian_filter)
        I2_filtered = cv2.filter2D(pyramid_2[i], -1, gaussian_filter)
        pyramid_1.append(I1_filtered[::2, ::2])
        pyramid_2.append(I2_filtered[::2, ::2])
        if feature_pyramid[i] is not None:
            feature_pyramid.append((feature_pyramid[i] // 2).astype(int))
        else:
            feature_pyramid.append(None)
   
    dy_previous = np.ones_like(pyramid_1[-1]) * 0 if dy_initial is None else np.resize(dy_initial, pyramid_1[-1].shape)
    dx_previous = np.ones_like(pyramid_1[-1]) * 0 if dx_initial is None else np.resize(dx_initial, pyramid_1[-1].shape)
    
    dx_final, dy_final = compute_optical_flow(pyramid_1, pyramid_2, feature_pyramid, rho, epsilon, dx_previous, dy_previous, scale)
    
    return dx_final, dy_final


# In[38]:


def track(param, single_scale, rho=20, epsilon=0.02, dx0=0, dy0=0, scale=3):
    # Initialize bounding box
    if param == 'Head':
        bound_box = [154, 102, 67, 115]
    elif param == 'Left hand':
        bound_box = [93, 272, 56, 83]
    elif param == 'Right hand':
        bound_box = [201, 270, 56, 83]
    
    if single_scale:
        scaling = 'single'
    else:
        scaling = 'multi'
    
    I_cur = read_image('cv23_lab2_material/part1 - GreekSignLanguage/1.png')

    # Initialize transform
    dx = dx0
    dy = dy0

    # Create directory
    os.makedirs("optical_flow/{}/{}".format(scaling, param), exist_ok=True)
    
    for i in range(2, 70):  # For all frames
        # ------------------------------- Feature Extraction -------------------------------------------
        
        # Read and extract features from the next frame
        I_features = cv2.imread('cv23_lab2_material/part1 - GreekSignLanguage/{}.png'.format(i))
        im2 = I_features[bound_box[1]:bound_box[1]+bound_box[3], bound_box[0]:bound_box[0]+bound_box[2]]
        gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, 2000, 0.15, 3)
        im_copy = np.copy(im2)

        # ---------------------------------- Lucas Kanade --------------------------------------------
        
        # Previous frame
        I_prev = np.copy(I_cur)

        # Read the next frame for Lucas Kanade
        I_cur = read_image('cv23_lab2_material/part1 - GreekSignLanguage/{}.png'.format(i))

        # Cut previous and next frames according to the new bounding box
        im1 = I_prev[bound_box[1]:bound_box[1]+bound_box[3], bound_box[0]:bound_box[0]+bound_box[2]]
        im2 = I_cur[bound_box[1]:bound_box[1]+bound_box[3], bound_box[0]:bound_box[0]+bound_box[2]]

        if single_scale:
            dx_r, dy_r = lk(im1, im2, features, rho, epsilon, dx, dy)
        else:
            dx_r, dy_r = lk_multi(im1, im2, features, rho, epsilon, dx, dy, scale)

        # For face
        fig, ax = plt.subplots(1, 3, figsize=(20, 8))
        plt.title('{} Features Optical Flow, from frame {} to frame {}'.format(param, i-1, i), fontsize=12)
        ax[1].invert_yaxis()
        scale_factor = 10 if param == 'Head' else 30 if param == 'Left hand' else 50
        ax[1].quiver(-dx_r, -dy_r, angles="xy", scale=scale_factor)
        ax[1].set_title('Optical Flow')
        
        if features is not None:
            for h in features:
                x, y = h.ravel()
                cv2.circle(im_copy, (x, y), 1, 255, -1)
        
        ax[0].set_title('Feature Extraction in frame {}'.format(i), fontsize=12)
        ax[0].imshow(im_copy)

        # Create a rectangle patch for the left hand
        x, y,hight, width = bound_box
        left = patches.Rectangle((x, y), height, width, linewidth=1, edgecolor='g', facecolor='none')
        ax[2].add_patch(left)
        ax[2].imshow(I_features)
        
        if i < 10:
            fig.savefig("optical_flow/{}/{}/0{}.png".format(scaling, param, i))
        else:
            fig.savefig("optical_flow/{}/{}/{}.png".format(scaling, param, i))
        
        # Calculate bounding box transform
        dx, dy = displ(dx_r, dy_r)

        # Update bounding box
        if not math.isnan(dx) and not math.isnan(dy):
            bound_box = [bound_box[0] - int(round(dx)), bound_box[1] - int(round(dy)), bound_box[2], bound_box[3]]                         


# In[39]:


track('Head', False, 4, 0.05, 0, 0)


# In[40]:


track('Left hand', False, 4, 0.05, 0, 0)


# In[41]:


track('Right hand', False, 4, 0.05, 0, 0)


# In[42]:


track('Left hand', True, 4, 0.05, 0, 0)


# In[45]:


track('Left hand', True, 1, 0.05, 0, 0)


# In[48]:


track('Left hand', True, 2, 0.05, 0, 0)


# In[50]:


track('Left hand', True, 4, 1, 0, 0)


# In[51]:


track('Left hand', True, 4, 0.075, 0, 0)


# In[52]:


track(param, single_scale, rho=20, epsilon=0.02, dx0=0, dy0=0, scale=3)


# In[53]:


track('Left hand', False, 4, 0.05, 0, 0, 5)


# In[55]:


track('Left hand', False, 4, 0.05, 0, 0, 6)


# In[ ]:




