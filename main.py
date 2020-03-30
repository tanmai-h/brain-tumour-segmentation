# import tensorflow as tf
import utils
import numpy as np
# from keras.models import load_model
from utils import io, plt

def pre(img, smooth=1e-7):
    img = img.astype('float32')
    img = (img-img.mean()) / (smooth+img.std())
    img = img.astype('float32')

    return img

def get_scan(n=1, path='./samples/'):
    images = {}
    images['flair'] = pre(io.imread(path+str(n)+'_flair.nii.gz',plugin='simpleitk'))[90]
    images['t1'] = pre(io.imread(path+str(n)+'_t1.nii.gz',plugin='simpleitk'))[90]
    images['t2'] = pre(io.imread(path+str(n)+'_t2.nii.gz',plugin='simpleitk'))[90]
    images['t1ce'] = pre(io.imread(path+str(n)+'_t1ce.nii.gz',plugin='simpleitk'))[90]
    
    seg = io.imread(path+str(n)+'_seg.nii.gz',plugin='simpleitk')
    
    img = seg.copy()
    images['full'] = img[90] 
    
    img = seg.copy()
    img[img != 0] = 1
    images['init'] = img[90] 
    
    img = seg.copy()
    img[img == 2] = 0	   
    img[img != 0] = 1
    images['core'] = img[90] 
        
    img = seg.copy()
    img[img != 4] = 0
    img[img == 4] = 1
    images['et'] = img[90] 
    
    return images

def presenter(scan):
    plt.figure(figsize=(15,10))

    plt.subplot(241)
    plt.title('T1')
    plt.axis('off')
    plt.imshow(scan['t1'],cmap='gray')

    plt.subplot(242)
    plt.title('T2')	
    plt.axis('off')
    plt.imshow(scan['t2'],cmap='gray')

    plt.subplot(244)
    plt.title('T1ce')
    plt.axis('off')
    plt.imshow(scan['t1ce'],cmap='gray')

    plt.subplot(243)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(scan['flair'],cmap='gray')  
    
    plt.subplot(245)
    plt.title('Ground Truth (Initial)')
    plt.axis('off')
    plt.imshow(scan['init'],cmap='gray')

    plt.subplot(246)
    plt.title('Ground Truth (Core)')
    plt.axis('off')
    plt.imshow(scan['core'],cmap='gray')

    plt.subplot(247)
    plt.title('Ground Truth (Enhanced)')
    plt.axis('off')
    plt.imshow(scan['et'],cmap='gray')

    plt.subplot(248)
    plt.title('Ground Truth (Full)')
    plt.axis('off')
    plt.imshow(scan['full'],cmap='gray')

    plt.show()

def predictor(model, batch_size=1, model_core=None, model_et=None, scan=None):
    x = np.zeros((1,2,240,240),np.float32)
    x[:,0,:,:] = np.expand_dims(scan['flair'], axis=0)
    x[:,1:,:,:] = np.expand_dims(scan['t2'], axis=0)

    pred_full = model.predict(x,batch_size=batch_size)
    
    if model_core is None or model_et is None:
        return pred_full[0,0,:,:]
    
    crop , li = utils.crop_tumor_tissue(np.expand_dims(scan['t1ce'],axis=0),pred_full[0,:,:,:],64)
    pred_core = model_core.predict(crop)
    pred_ET = model_et.predict(crop)

    tmp = utils.paint_color_algo(pred_full[0,:,:,:], pred_core, pred_ET, li)

    core = np.zeros((1,240,240),np.float32)
    ET = np.zeros((1,240,240),np.float32)
    core[:,:,:] = tmp[:,:,:]
    ET[:,:,:] = tmp[:,:,:]
    core[core == 4] = 1
    core[core != 1] = 0
    ET[ET != 4] = 0
    
    pfull = pred_full[0,0,:,:]
    pcore = core[0, :, :]
    pet = ET[0, :, :]
    pc = tmp[0, :, :]

    plt.figure(figsize=(15,10))
    plt.subplot(141)
    plt.title('Prediction (Initial)')
    plt.axis('off')
    plt.imshow(pred_full[0, 0, :, :],cmap='gray')

    plt.subplot(142)
    plt.title('Prediction (Core)')
    plt.axis('off')
    plt.imshow(core[0, :, :],cmap='gray')

    plt.subplot(143)
    plt.title('Prediction (Enhanced)')
    plt.axis('off')
    plt.imshow(ET[0, :, :],cmap='gray')

    plt.subplot(144)
    plt.title('Prediction (Complete)')
    plt.axis('off')
    plt.imshow(tmp[0, :, :],cmap='gray')

    plt.show()

    p = {}
    p['full'] = pfull
    p['core'] = pcore
    p['et'] = pet
    p['Complete'] = pc

    return p