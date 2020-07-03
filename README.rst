.. code:: ipython3

    import sys, os
    import numpy as np
    import scipy 
    import matplotlib.pyplot as plt
    from sklearn import decomposition
    from PIL import Image
    import warnings
    warnings.filterwarnings("ignore")
    from ipywidgets import interact,interactive,fixed,interact_manual,IntSlider
    import ipywidgets as widgets

.. code:: ipython3

    def plt_images(img,x):
        print(img.shape)
        x.imshow(img)
    
    #def rgb2gray(rgb):
    #    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    #from matplotlib import image
    #img=image.imread('test1.png')
    #plt_images(img)
    #img=rgb2gray(img)
    #plt_images(img)

.. code:: ipython3

    @interact
    def svd(img=os.listdir('images'),k=IntSlider(value=50,min=1,max=250,step=1,description='NO OF COMPONENTS:',\
                                                 style={'description_width': 'initial', 'width': '2000px'},continuous_update=False,orientation='horizontal',layout=dict(width='95%'))):
            
        fig,a =  plt.subplots(1,2,figsize=(14,14))
        a[0].set_title("COMPRESSED IMAGE")
        a[1].set_title("ORIGNAL IMAGE")
        dims=(600,600)
        im =  np.array(Image.open(os.path.join('images',img)).convert('L').resize(dims))
        im=im/255.0
        plt_images(im,a[1])
        
        u, s, v = decomposition.randomized_svd(im, k)
        low_rank = u @ np.diag(s) @ v
        plt_images(low_rank,a[0])
        plt.show()
        data_needed=k*dims[0]+dims[1]*k+k#size of v,d +k
        print("\t\tUSING {} COMPONENTs and {} DATA POINTS OUT OF ORIGNAL {} POINTS.".format(k,data_needed,dims[0]*dims[1]))




.. parsed-literal::

    interactive(children=(Dropdown(description='img', options=('1.jpg', '3.jpg'), value='1.jpg'), IntSlider(value=…


.. code:: ipython3

    @interact
    def svd(img=os.listdir('images'),k=IntSlider(value=1,min=1,max=250,step=1,description='NO OF COMPONENTS:',\
                                                 style={'description_width': 'initial', 'width': '2000px'},continuous_update=False,orientation='horizontal',layout=dict(width='95%'))):
            
        fig,a =  plt.subplots(1,2,figsize=(14,14))
        a[0].set_title("COMPRESSED IMAGE")
        a[1].set_title("ORIGNAL IMAGE")
        dims=(600,600)
        im =  np.array(Image.open(os.path.join('images',img)).resize(dims))
        im=im/255.0
        plt_images(im,a[1])
        flatim=im.flatten()
        flatim=im.flatten().reshape(-1,3)
        u, s, v = decomposition.randomized_svd(flatim, k)
        low_rank = u @ np.diag(s) @ v
        print(u.shape)
        low_rank=low_rank.reshape(dims+(3,))
        plt_images(low_rank,a[0])
        plt.show()
        data_needed=3*(k*(dims[0]+dims[1])+k)#size of v,d +k
        print("\t\tUSING {} COMPONENTs and {} DATA POINTS OUT OF ORIGNAL {} POINTS.".format(k,data_needed,3*(dims[0]*dims[1])))
    #     b.imshow(np.reshape(flatim[:], dims), cmap='gray')



.. parsed-literal::

    interactive(children=(Dropdown(description='img', options=('1.jpg', '3.jpg'), value='1.jpg'), IntSlider(value=…

