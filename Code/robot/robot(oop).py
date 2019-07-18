#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def readMax(matrix):
    indices = np.where(matrix == matrix.max())
    return int(indices[0]),int(indices[1]),matrix[int(indices[0]),int(indices[1])]


# In[3]:


class robot():
    '''
    Define a robot class
    grid_size --- The size of the grid on which we place our robot to search for the leak
    '''
    grid_size = 15   #class variable
    def __init__(self, xpos, ypos, reading=0, search_size=1):
        self.x = xpos
        self.y = ypos
        self.reading = reading
        self.search_size = search_size
        self.leak_loc_x = xpos
        self.leak_loc_y = ypos
    def move_left(self):
        if self.x >= 1:
            self.x -= 1
        #else:
            #print("hit the left boundary")

    def move_right(self):
        if self.x < robot.grid_size-1:
            self.x += 1
        #else:
            #print("hit the right boundary")
            
    def move_down(self):
        if self.y < robot.grid_size-1:
            self.y += 1
        #else:
            #print("hit the lower boundary")
    
    def move_up(self):
        if self.y >= 1:
            self.y -= 1
        #else:
            #print("hit the upper boundary")
            
    def search(self,matrix):
        '''
        search the robot's neighbouring girds, record the max value as the current reading and move
        the robot to the position of the max value.
        '''
        max_x,max_y, max_value = readMax(matrix)
        if max_value >= self.reading:
            self.reading = max_value
            #should use move_up/down/left/right
            self.x =self.x - 1 + max_x
            self.y =self.y - 1 + max_y
            
            self.leak_loc_x = self.x
            self.leak_loc_y = self.y


# In[18]:


readMax(matrix[0:3,0:2])


# In[21]:


assemble_submatrix(matrix, 0,0,1)


# In[4]:


def assemble_submatrix(matrix, cx, cy, window):
    '''
    form a 3 by 3 submatrix of the given matrix with center at (cx,cy)
    if at the boundary, the size is changed accordingly.
    '''
    m=matrix.shape[0]
    n=matrix.shape[1]
    
    if cx-window<0:
        sub_x_low = cx
    else:
        sub_x_low = cx-window
        
    if cx+window>=m+1:
        sub_x_up = cx
    else:
        sub_x_up = cx+window+1
        
        
    if cy-window<0:
        sub_y_low = cy
    else:
        sub_y_low = cy-window
        
    if cy+window>=n+1:
        sub_y_up = cy
    else:
        sub_y_up = cy+window+1
    
    submatrix = matrix[sub_x_low:sub_x_up, sub_y_low:sub_y_up]
    return submatrix


# In[31]:


assemble_submatrix(matrix, 2,2,1)


# In[33]:


matrix.shape


# In[36]:


robo = robot(0,0)


# 

# In[34]:


def robot_search(matrix, robot):
    m = matrix.shape[0]
    n = matrix.shape[1]
    padded_matrix = np.pad(matrix, [(1, 1), (1, 1)], mode='constant',constant_values=-1)
    for j in range(n):
        #print("y coord", robot.y)
        if j%2 == 0:
            for i in range(m):
                #print("x coord", robot.x)
                robot.move_right()
                submatrix = assemble_submatrix(padded_matrix, cx=robot.x,cy=robot.y,window=robot.search_size)
                print(robot.x, robot.y, "robo position")
                robot.search(submatrix)
        else:
            for i in range(m):
                #print("x coord", robot.x)
                robot.move_left()
                submatrix = assemble_submatrix(padded_matrix, cx=robot.x,cy=robot.y,window=robot.search_size)
                print(robot.x, robot.y, "robo position")
                robot.search(submatrix)

        robot.move_down()
    return robot.reading


# In[38]:


robo.x


# In[39]:


robot_search(matrix,robo)


# In[16]:


robo.y


# In[25]:


matrix[14,14]


# In[9]:


matrix.max()


# In[7]:


np.random.seed(1);
matrix=np.random.rand(15,15)


# In[8]:


matrix.shape


# In[10]:


readMax(matrix)


# In[40]:


# #import matplotlib.ticker as mticker
# %matplotlib notebook
# import matplotlib.animation as animation


# In[ ]:





# In[26]:


a = np.array([[ 1.,  1.,  1.,  1.,  1.], [ 1.,  1.,  1.,  1.,  1.],[ 1.,  1.,  1.,  1.,  1.]])


# In[27]:


a


# In[30]:


np.pad(a, [(1, 1), (1, 1)], mode='constant',constant_values=-1)


# In[ ]:




