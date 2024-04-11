#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Note, must use Python 3.7
# Uses bash commands through os! Only tested on Linux
# This program uses Pyphase as outlined in example on https://pyphase.readthedocs.io/en/master/examples.html
# but reworks the formatting of the input to more easily use other types of datasets

import pyphase
from pyphase import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from PIL import Image
import h5py
import tables as tb
import matplotlib as mpl
import copy
import matplotlib.font_manager as fm # Fixing font
fe = fm.FontEntry(fname='/usr/share/fonts/cm-unicode/cmunrm.otf', name='cmunrm-manual')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
plt.rcParams['font.family'] = fe.name
plt.rcParams['font.size'] = 24
plt.rcParams["mathtext.fontset"] = "cm"

mpl.rc('image', cmap='afmhot')

renderHdf5 = True
saveFormat = 'png'

# 10 mm for 2048 pixels, measured in m
screen_width = 10e-3 # 10 mm
nx = ny = 2048 # simulations
pixelsize=[screen_width/nx,screen_width/ny] # simulations
#pixelsize=[0.0016/100,0.0016/100] # from experiments

distance=np.array([0.1])*1e0
energy = float(0.0023308270676692) # Converted from lambda = 532 nm to eV

data_filename = "circle-reversal"

save_dir = f"./renders/{data_filename}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if renderHdf5:
    os.system("rm pixelsizes.txt")

    pix_file = open("pixelsizes.txt", "a")
    pix_file.writelines([f"basez\tx (um)\ty (um)\n"])
    for d in distance:
        # convert lengths to um instead of m and append to file
        pix_file.writelines([f"{d*1e6}\t{pixelsize[0]*1e6}\t{pixelsize[1]*1e6}\n"])
    pix_file.close()

    os.system("rm *_master.h5")
    h5name = data_filename + ".h5"
    print("Creates h5 based on images in ./patterns. Remember to update values in pixelsizes.txt (how many um between sought after reverse plane and final plane and the size of each pixel in um in final plane")
    os.system('rm ' + data_filename + "*")
    f = tb.open_file(h5name, "w")
    root = f.root
    data_dir = f.create_group(root,"data")
    cpr_dir = f.create_group(data_dir,"cpr")
    image_index = 0 # won't work for more than 10 images
    for file in np.sort(os.listdir("patterns/")):
        if "jpg" in file.lower() or "png" in file.lower():
            imArr = np.array(Image.open("patterns/" + file).convert("L").resize([nx,ny]))/255
            f.create_array(cpr_dir, "00000" + str(image_index), imArr, file)
            image_index += 1
    f.close()

# Phase retrieval from a dataset preset
data = dataset.NanomaxPreprocessed2D(data_filename, version='master')

data.padding = 1
data.energy=data.nx = nx
data.ny = ny
nxp = data.padding * nx
nyp = data.padding * ny

# Align images using default registrator (elastix), values from Pyphase tutorial
pyphase.registrator.parameters.NumberOfResolutions = 8
pyphase.registrator.parameters.MaximumNumberOfIterations = 3000

# Requieres proper install of elastix-4.9.0, newer versions do not work, takes a long time!
# Will only do something if more than one image
if renderHdf5:
    data.align_projection()

for N in range(len(distance)):
    plt.imshow(data.get_projection(projection=0, position=N, pad=True),vmax=1,vmin=0)
    plt.colorbar()
    plt.axis('off')
    plt.title("Measured Intensity " + str(N))
    plt.savefig(save_dir + data_filename + '_initial_' + str(N) + '.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
    plt.clf()
    mpl.image.imsave(save_dir + data_filename +f'_initial_{str(N)}_pure.'+saveFormat, (data.get_projection(projection=0, position=N, pad=True)), cmap = 'gray')


# Phase retrieval from a dataset using HIO_ER
algorithm="HIO-ER"
retriever = phaseretrieval.HIO_ER(shape=[nx,ny],pixel_size=pixelsize,distance=distance,energy=energy)

# Modify some parameters
retriever.alpha = [1e-4, 1e-8] # Regularisation parameter used for initialisation
retriever.iterations_hio = 3# Original 3, iterations of HIO
retriever.iterations_er = 2# Original 2, iterations of ER
retriever.iterations = 2# Original 2, global iterations

# Reconstruct
data_copy =copy.deepcopy(data)
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
attenuation = data_copy.get_image(projection=0, image_type='attenuation')
logged_attenuation = np.log(attenuation-np.min(attenuation)+0.000001)
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(attenuation,vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

mpl.image.imsave(save_dir + data_filename +'_'+algorithm+'_attenuation_pure.'+saveFormat, attenuation, cmap = 'gray')
mpl.image.imsave(save_dir + data_filename +'_'+algorithm+'_logged_attenuation_pure.'+saveFormat, logged_attenuation, cmap = 'gray')


# BELOW ARE REPLICATES OF CODE ABOVE FOR DIFFERENT ALGORITHMS
'''
algorithm="CTF"
retriever = phaseretrieval.CTF(shape=[nx,ny],pixel_size=pixelsize,distance=distance,energy=energy)
retriever.alpha = [1e-4, 1e-8] 
retriever.iterations_hio = 3
retriever.iterations_er = 2
retriever.iterations = 2 
data_copy =copy.deepcopy(data)
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(data_copy.get_image(projection=0, image_type='attenuation'))#,vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

###

algorithm="TIEHOM"
retriever = phaseretrieval.TIEHOM(shape=[nx,ny],pixel_size=pixelsize,distance=distance,energy=energy)
retriever.alpha = [1e-4, 1e-8] 
retriever.iterations_hio = 3
retriever.iterations_er = 2
retriever.iterations = 2 
data_copy =copy.deepcopy(data)
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(data_copy.get_image(projection=0, image_type='attenuation'),vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

###

algorithm="CTFPurePhase"
retriever = phaseretrieval.CTFPurePhase(shape=[nx,ny],pixel_size=pixelsize,distance=distance,energy=energy)
retriever.alpha = [1e-4, 1e-8] 
retriever.iterations_hio = 3
retriever.iterations_er = 2
retriever.iterations = 2 
data_copy =copy.deepcopy(data)
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(data_copy.get_image(projection=0, image_type='attenuation'))#,vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

###

algorithm="Mixed"
data_copy =copy.deepcopy(data)
retriever = phaseretrieval.Mixed(dataset=data_copy)
retriever.alpha = [1e-4, 1e-8] 
retriever.iterations_hio = 3
retriever.iterations_er = 2
retriever.iterations = 2 
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(data_copy.get_image(projection=0, image_type='attenuation'))#,vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

###

algorithm="GradientDescent"
data_copy =copy.deepcopy(data)
retriever = phaseretrieval.GradientDescent(dataset=data_copy)
retriever.alpha = [1e-4, 1e-8] 
retriever.iterations_hio = 3
retriever.iterations_er = 2
retriever.iterations = 2 
retriever.reconstruct_projection(dataset=data_copy, projection=0) 
plt.imshow(data_copy.get_image(projection=0))
plt.colorbar()
plt.axis('off')
plt.title("Phase $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_phase.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()
plt.imshow(data_copy.get_image(projection=0, image_type='attenuation'),vmax=1,vmin=0)
plt.colorbar()
plt.axis('off')
plt.title("Attenuation $\mathtt{" + algorithm + "}$")
plt.savefig(save_dir + data_filename +'_'+algorithm+'_attenuation.'+saveFormat, format=saveFormat,dpi=700,bbox_inches='tight',pad_inches=0)
plt.clf()

###
'''

os.system(f"cp pixelsizes.txt *_master.h5 {save_dir}")
