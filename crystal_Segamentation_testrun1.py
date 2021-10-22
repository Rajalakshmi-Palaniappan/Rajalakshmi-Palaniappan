#!/usr/bin/env python
# coding: utf-8

# In[34]:


import subprocess
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.formats import FORMAT_IMPLEMENTATIONS
import tifffile
from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
import pims
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import label, regionprops, regionprops_table
import math
import matplotlib.colors
import random
import os
import pandas as pd
from ipywidgets import interact, widgets
from IPython.display import display
import matplotlib.pyplot as plt
import warnings

plt.rcParams['figure.dpi'] = 300
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# In[35]:


input_directory = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/With3Channels'

output = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/Channel_of_interest'

os.listdir(input_directory)


# In[36]:


for filename in os.listdir(input_directory):
    filepath = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/With3Channels/' + filename 
    if filename.endswith(".czi"):
        img = AICSImage(filepath)
        img.data
        img.dims
        img.shape
        
        third_channel_data = img.get_image_data("ZYX", C=3, S=0, T=0, colormap="yellow")
        
        tif_filename = filename.replace('.czi' , '.tiff')
        
        tifffile.imsave('/Users/palaniappan/Desktop/Image_Analysis/Input_images/Channel_of_interest/' + filename.replace('.czi' , '.tiff'), third_channel_data)


# In[39]:


input_pp = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/Channel_of_interest'
os.listdir(input_pp)


# In[5]:


os.remove("/Users/palaniappan/Desktop/Image_Analysis/Input_images/Channel_of_interest/.DS_Store")


# In[40]:


for filename in os.listdir(input_pp):
    filepath = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/Channel_of_interest/' + filename
    if filename.endswith('.tiff'):
        multi_tiff_img = Image.open(filepath)

        img_list = []

        for i, page in enumerate(ImageSequence.Iterator(multi_tiff_img)):
            npImage = np.array(page)
             # Get brightness range - i.e. darkest and lightest pixels
            min=np.min(npImage)        # result=144
            max=np.max(npImage)        # result=216

    # Make a LUT (Look-Up Table) to translate image values
            LUT=np.zeros(256,dtype=np.uint8)
            LUT[min:max+1]=np.linspace(start=0,stop=255,num=(max-min)+1,endpoint=True,dtype=np.uint8)

    # Apply LUT and save resulting image
            img_list.append(Image.fromarray(LUT[npImage]))
            img_list[0].save("/Users/palaniappan/Desktop/Image_Analysis/Input_images/Preprocessed_images/" + filename, save_all=True, append_images=img_list[1:])
    


# In[41]:


ilastik_location = '/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/'

os.chdir(ilastik_location)

ilastik_project = '/Users/palaniappan/Desktop/ilastik projects/56hpf_Lysosensor.ilp'

indir = '/Users/palaniappan/Desktop/Image_Analysis/Input_images/Preprocessed_images/'

infiles = os.listdir(indir)


# In[42]:


os.chdir(ilastik_location)
for infile in infiles:

    if '.DS_Store' in infile:
        print ("skipping %s".format(infile))
        continue

#     if infile[-4:] != '.tif':
#         print ("skipping %s".format(infile))
#         continue
    # probabilities, simple segmentation

    export_source_type = "probabilities"
    print (ilastik_project)
    print (indir)
    print (infile)
    command = './run_ilastik.sh --headless --project="%s" --export_source="%s" --output_format="multipage tiff" --output_filename_format="/Users/palaniappan/Desktop/Image_Analysis/Probabilities/{nickname}_results.tif" --raw_data="%s%s"' % (
        ilastik_project,
        export_source_type,
        indir,
        infile)
    print (command)
    os.system(command)


# In[43]:


i = 0
tiffs_path = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities'

for filename in os.listdir(tiffs_path):
    filepath = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities/' + filename


    if filename.endswith(".tiff"):
        img = Image.open(filepath)
        print(filename)
        print(img.n_frames)


# In[44]:


for filename in os.listdir(tiffs_path):
    if ".tiff" in filename:

        print (filename)

        newpath = ('/Users/palaniappan/Desktop/Image_Analysis/Probabilities/' + filename.replace(".tiff", ""))
        if not os.path.exists(newpath):
            print (newpath)
            os.mkdir(newpath)


# In[45]:


for filename in os.listdir(tiffs_path):

    filepath = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities/' + filename

    folder_path = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities/' + filename.replace(".tiff", "")

    if filename.endswith(".tiff"):

        img = Image.open(filepath)
        frame_count = 0
        for i in range(img.n_frames):
            if i%2 == 0:
                try:
                    img.seek(i)
                    img.save('%s/frame_%s.tif'%(folder_path, frame_count,))
                    frame_count += 1
                except EOFError:
                    break


# In[46]:


base_path = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities'

dir_list = [os.path.join(base_path, item) for item in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, item))]

dir_list


# In[57]:


Max = 100

def scaler(arr):
    unique = np.unique(arr)
    Max = np.max(unique)
    scaled = (arr/Max)*100

    return scaled


# In[58]:


def scaling_images(frames):
    scaled_images_max = []

    for j in range(len(frames)):
        image = frames[j]
        #image = image[:,:,0]  # in case foreground and background
        scaled = scaler(image)

        scaled_images_max.append(scaled)
    return scaled_images_max


# In[59]:


def object_detection(tresholds, output_path, scaled_images_max):
    save = 'yes'

    for tresh in tresholds:
        temp = 'tresh_' + str(tresh)

        out_path_cls_tresh = output_path + '/' + temp

        if os.path.exists(out_path_cls_tresh):
            pass
        else:  
            os.makedirs(out_path_cls_tresh, exist_ok=True)

        list_prop = []    
        for frame in range(len(scaled_images_max)) :                   

            image = scaled_images_max[frame]                                           #list_arr[frame]
            image = (image > tresh).astype(np.uint8)                         # below 72 h 50 - 3000  , above 72h 200-3000
            label_img = label(image)
            regions = regionprops(label_img)

            props = regionprops_table(label_img, properties=('centroid',
                                                         'orientation',
                                                         'major_axis_length',
                                                         'minor_axis_length','area','perimeter'))
            df_prop = pd.DataFrame(props)
            suffix = "frame_" + str(frame) + "_"

            df_save = df_prop.set_index( suffix + df_prop.index.astype(str)  )
            #df_final.to_excel(output_path + '/'+"output.xlsx")
            list_prop.append(df_save)




            fig, ax = plt.subplots()
            ax.imshow(image, cmap=plt.cm.gray)


            counter = 0
            for props in regions:


                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

                ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
                ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
                ax.plot(x0, y0, '.g', markersize=2)

                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot(bx, by, '-b', linewidth=0.7)
                ax.text(bx[0]-5, by[0]-5, counter, color='Green', size = 2)
                counter += 1

            ax.axis((0, 1000, 1000, 0))
            if save == 'yes':
                plt.savefig( os.path.join( out_path_cls_tresh, temp + "_"+ image_names[frame] + "_.jpg" ),  dpi=200 ,bbox_inches='tight')
            else:
                pass


        df_final = pd.concat(list_prop)                                        # Create unique dataframe
        df_final.to_excel(out_path_cls_tresh + '/'+ temp+ "_output.xlsx")      # Save it 


# In[60]:


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# In[61]:



for input_path in dir_list:
#     input_path = '/Users/palaniappan/Desktop/Image_Analysis/Probabilities/2021-7-22_wt_56hpf_lysotracker1hSyto9-8h_10_results/'  # Input folder where images are selected

#     frames = pims.ImageSequence('/Users/palaniappan/Desktop/Image_Analysis/Probabilities/2021-7-22_wt_56hpf_lysotracker1hSyto9-8h_10_results/*.tif') 
    print ('%s/*.tif' % input_path)
    frames = pims.ImageSequence('%s/*.tif' % input_path) 



    save_folder = 'output_real_images' # folder that will be created inside 'input_path' folder
    output_path = input_path + '/' + save_folder + '/'



    image_names = os.listdir(str(input_path)) # Get images names for saving later
    image_names = [sub.replace('.tif', '') for sub in image_names] # If you use tiff images don't change the '.tiff'
    image_names.sort()




    if os.path.exists(output_path) is False:
            os.makedirs(output_path , exist_ok=True)
            print('output directory created')

    print("{} Frames loaded".format(len(frames)))
    scaled_images_max = []

    scaled_images_max = scaling_images(frames)
    
    try:
        tresholds = []
     
        while True:
            tresholds.append(int(input()))
         
            # if the input is not-integer, just print the list
    except:
        print(tresholds)
        
    object_detection(tresholds, output_path, scaled_images_max)


# In[ ]:




