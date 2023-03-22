
# # MOwNiT – arytmetyka komputerowa




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


k = 45
p = 3


# In[3]:


def visualize(points1, points2):
    plt.figure(figsize=(12,6))
    plt.xlabel("X[k]")
    plt.ylabel("values")
    plt.plot(points1, marker="o", markersize=5, label="Double")
    plt.plot(points2, marker="o", markersize=5, label="Longdouble")
    plt.legend()
    plt.show()


# In[4]:


def visualize2(points1, points2):
    df = pd.DataFrame({'Forward': points1, 'Backward': points2})
    print(df)


# In[5]:


def visualize3(float_points1, float_points2, double_points1, dobule_points2, longdouble_points1, longdouble_points2):
    df = pd.DataFrame(
        {'Forward float': float_points1, 'Backward float': float_points2, 'Forward double': double_points1,
         'Backward double': dobule_points2, 'Forward long double': longdouble_points1,
         'Backward long double': longdouble_points2})
    print(df)


# In[6]:


def visualize4(points1,points2,points3):
    plt.figure(figsize=(12,6))
    plt.xlabel("X[k]")
    plt.ylabel("values")
    plt.plot(points1, marker="o", markersize=5, label="Float32")
    plt.plot(points2, marker="o", markersize=5, label="Double")
    plt.plot(points3, marker="o", markersize=5, label="Longdouble")
    plt.legend()
    plt.show()


# # Inforamcje o typach danych:

# In[7]:


print(np.finfo(np.float32))


# In[8]:


print(np.finfo(np.double))


# In[9]:


print(np.finfo(np.longdouble))


# # Typ liczb float:

# In[10]:


def float_sequence(k, p):
    arr = [0 for _ in range(k + 2)]
    arr[0],arr[1] = np.float32(1),np.float32(1 / p)
    for i in range(2, k + 2):
        arr[i] = np.float32((1 / p) * (10 * arr[i - 1] - p * arr[i - 2]))

    arrBack = [0 for _ in range(k + 2)]
    arrBack[k],arrBack[k + 1] = arr[k],arr[k + 1]
    for i in range(k - 1, -1, -1):
        arrBack[i] = np.float32((1 / p) * (10 * arrBack[i + 1] - p * arrBack[i + 2]))
    return arr,arrBack


# In[11]:


float32_arr,float32_back = float_sequence(k,p)


# In[12]:


visualize2(float32_arr,float32_back)


# # Typ liczb double:

# In[13]:


def double_sequence(k, p):
    arr = [0 for _ in range(k + 2)]
    arr[0],arr[1] = np.double(1),np.double(1 / p)
    for i in range(2, k + 2):
        arr[i] = np.double((1 / p) * (10 * arr[i - 1] - p * arr[i - 2]))

    arrBack = [0 for _ in range(k + 2)]
    arrBack[k],arrBack[k + 1] = arr[k],arr[k + 1]
    for i in range(k - 1, -1, -1):
        arrBack[i] = np.double((1 / p) * (10 * arrBack[i + 1] - p * arrBack[i + 2]))
    return arr,arrBack


# In[14]:


double_arr,double_back = double_sequence(k,p)


# In[15]:


visualize2(double_arr,double_back)


# #  Typ liczb longdouble:

# In[16]:


def longdouble_sequence(k, p):
    arr = [0 for _ in range(k + 2)]
    arr[0],arr[1] = np.longdouble(1),np.longdouble(1 / p)
    for i in range(2, k + 2):
        arr[i] = np.longdouble((1 / p) * (10 * arr[i - 1] - p * arr[i - 2]))

    arrBack = [0 for _ in range(k + 2)]
    arrBack[k],arrBack[k + 1] = arr[k],arr[k + 1]
    for i in range(k - 1, -1, -1):
        arrBack[i] = np.longdouble((1 / p) * (10 * arrBack[i + 1] - p * arrBack[i + 2]))

    return arr,arrBack


# In[17]:


longdouble_arr,longdouble_back = longdouble_sequence(k,p)


# In[18]:


visualize2(longdouble_arr,longdouble_back)


# # Porównanie wyników wszystkich obliczeń: 

# In[19]:


visualize3(float32_arr,float32_back,double_arr,double_back,longdouble_arr,longdouble_back)


# In[20]:


visualize4(float32_back,double_back,longdouble_back)


# In[21]:


visualize(double_back,longdouble_back)


# In[22]:


print(float32_back)


# In[23]:


print(double_back)


# In[24]:


print(longdouble_back)

