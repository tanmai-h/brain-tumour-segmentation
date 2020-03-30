import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import random as r
import math
import glob

from constants import *

def create_data(src, mask, label=False, label_num=5, resize=(155,img_size,img_size), smooth=1e-7):
	print(src+mask)
	files = glob.glob(src + mask, recursive=True)
	print(files)
	r.seed(9)
	# r.shuffle(files)	# shuffle patients
	imgs = []
	print('Processing---', mask)
	for file in files:
		img = io.imread(file, plugin='simpleitk')
		#img = trans.resize(img, resize, mode='constant')
		if label:
			if label_num == 5:
				img[img != 0] = 1	   #Region 1 => 1+2+3+4 complete tumor
			if label_num == 1:
				img[img != 1] = 0	   #only left necrosis and NET
			if label_num == 2:
				img[img == 2] = 0	   #turn edema to 0
				img[img != 0] = 1	   #only keep necrosis, ET, NET = Tumor core
			if label_num == 4:
				img[img != 4] = 0	   #only left ET
				img[img == 4] = 1
			if label_num == 3:
				img[img == 3] = 1	   # remain GT, design for 2015 data
				
				
			img = img.astype('float32')
		else:
			img = (img-img.mean()) / (smooth+img.std())	  #normalization => zero mean   !!!care for the std=0 problem
			img = img.astype('float32')
			
		for slice in range(60,130):	 #choose the slice range
			img_t = img[slice,:,:]
			img_t =img_t.reshape((1,)+img_t.shape)
			img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
			
			for n in range(img_t.shape[0]):
				imgs.append(img_t[n,:,:,:])
	
	return np.array(imgs)

def create_data_onesubject_val(src='', mask='', file_path=None, count=1, label=False, label_num=1, smooth=1e-7):
	file = None
	if file_path is None:
		files = glob.glob(src + mask, recursive=True)
		r.seed(9)
		r.shuffle(files)	# shuffle patients
		k = len(files) - count -1
		imgs = []
		file = files[k]
		print('Processing---', mask,'--',file)
	else:
		file = file_path

	img = io.imread(file, plugin='simpleitk')

	if label:
		if label_num == 5:
			img[img != 0] = 1	   #Region 1 => 1+2+3+4 complete tumor
		if label_num == 1:
			img[img != 1] = 0	   #only left necrosis
		if label_num == 2:
			img[img == 2] = 0	   #turn edema to 0
			img[img != 0] = 1	   #only keep necrosis, ET, NET = Tumor core
		if label_num == 4:
			img[img != 4] = 0	   #only left ET
			img[img == 4] = 1
		img = img.astype('float32')
	else:
		img = (img-img.mean()) / (img.std() + smooth)	  #normalization => zero mean   !!!care for the std=0 problem
		img = img.astype('float32')
	
	for slice in range(155):	 #choose the s
		img_t = img[slice,:,:]
		img_t =img_t.reshape((1,)+img_t.shape)
		img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
		
		for n in range(img_t.shape[0]):
			imgs.append(img_t[n,:,:,:])
	
	return np.array(imgs)

def read_subject(path=None, count=106, pull_seq=['flair', 't1ce', 't1', 't2'], 
					mask_seq=[('Label_Full',5),('Label_Core', 2),('Label_ET',4),('Label_All',3)]):
	if path is None:
		raise ValueError

	images, masks = {}, {}
	for seq in pull_seq:
		mask = '**/*{}.nii.gz'.format(seq)
		tmp = create_data_onesubject_val(path, mask=mask, count=count)
		
		images[seq] = tmp
	
	for l in mask_seq:
		name, num = l[0],l[1]
		s_mask =  '**/*seg.nii.gz'
		tmp = create_data_onesubject_val(path, mask=s_mask, count=count, label=True, label_num=num)

		masks[name] = tmp

	return images, masks

def crop_tumor_tissue(x, pred, size):   
	#   input: x:T1c image , pred:prediction of full tumor ,size default  64x64
    crop_x = []
    list_xy = []
    p_tmp = pred[0,:,:]
    p_tmp[p_tmp>0.2] = 1
    p_tmp[p_tmp !=1] = 0
    #get middle point from prediction of full tumor
    index_xy = np.where(p_tmp==1)   # get all the axial of pixel which value is 1

    if index_xy[0].shape[0] == 0:   #skip when no tumor
        return [],[]
        
    center_x = (max(index_xy[0]) + min(index_xy[0])) / 2 
    center_y = (max(index_xy[1]) + min(index_xy[1])) / 2 
    
    if center_x >= 176:
            center_x = center_x-8
        
    length = max(index_xy[0]) - min(index_xy[0])
    width = max(index_xy[1]) - min(index_xy[1])
        
    if width <= 64 and length <= 64:  #64x64
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size/2),int(center_y - size/2)))
            
    if width > 64 and length <= 64:  #64x128
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size/2),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y + 1) : int(center_y + size + 1)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size/2),int(center_y)))
            
    if width <= 64 and length > 64:  #128x64       
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size),int(center_y - size/2)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x),int(center_y - size/2)))
            
    if width > 64 and length > 64:  #128x128
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y + 1) : int(center_y + size + 1)]
        crop_x.append(img_x)
        
        list_xy.append((int(center_x - size),int(center_y)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y + 1) : int(center_y + size + 1)]
        
        crop_x.append(img_x)
        list_xy.append((int(center_x),int(center_y)))
            
        
    return np.array(crop_x) , list_xy   #(y,x)

def paint_color_algo(pred_full, pred_core , pred_ET , li):
	#input image is [n,1, y, x]
    # first put the pred_full on T1c
    pred_full[pred_full > 0.2] = 2      #240x240
    pred_full[pred_full != 2] = 0
    pred_core[pred_core > 0.2] = 1      #64x64
    pred_core[pred_core != 1] = 0
    pred_ET[pred_ET > 0.2] = 4          #64x64
    pred_ET[pred_ET != 4] = 0

    total = np.zeros((1,240,240),np.float32)  
    total[:,:,:] = pred_full[:,:,:]
    for i in range(pred_core.shape[0]):
        for j in range(64):
            for k in range(64):
                if pred_core[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                    total[0,li[i][0]+j,li[i][1]+k] = pred_core[i,0,j,k]
                if pred_ET[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                    total[0,li[i][0]+j,li[i][1]+k] = pred_ET[i,0,j,k]

    return total

if __name__ == '__main__':
		
	plt.figure(figsize=(15,10))

	plt.subplot(241)
	plt.title('T1')
	plt.axis('off')
	plt.imshow(images['T1'][90, 0, :, :],cmap='gray')

	plt.subplot(242)
	plt.title('T2')	
	plt.axis('off')
	plt.imshow(images['T2'][90, 0, :, :],cmap='gray')
		
	plt.subplot(243)
	plt.title('Flair')
	plt.axis('off')
	plt.imshow(images['Flair'][90, 0, :, :],cmap='gray')

	plt.subplot(244)
	plt.title('T1c')
	plt.axis('off')
	plt.imshow(images['T1c'][90, 0, :, :],cmap='gray')

	plt.subplot(245)
	plt.title('Ground Truth(Full)')
	plt.axis('off')
	plt.imshow(masks['Label_Full'][90, 0, :, :],cmap='gray')

	plt.subplot(246)
	plt.title('Ground Truth(Core)')
	plt.axis('off')
	plt.imshow(masks['Label_Core'][90, 0, :, :],cmap='gray')

	plt.subplot(247)
	plt.title('Ground Truth(ET)')
	plt.axis('off')
	plt.imshow(masks['Label_ET'][90, 0, :, :],cmap='gray')

	plt.subplot(248)
	plt.title('Ground Truth(All)')
	plt.axis('off')
	plt.imshow(masks['Label_all'][90, 0, :, :],cmap='gray')

	plt.show()