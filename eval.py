import main, utils, models
from utils import io, plt
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

n = 88

imgs, masks = [], []
print('Loading Data')
# for i in range(1,n+1):
#     scan = main.get_scan(n=i)
#     x = np.zeros((2,240,240))
#     x[0,:,:] = np.expand_dims(scan['flair'], axis=0)
#     x[1:,:,:] = np.expand_dims(scan['t2'], axis=0)
#     masks.append(scan['seg'])
#     imgs.append(x)

for i in range(1, n+1):
    npy = np.load('./npy/'+str(i)+'.npy')
    x = np.zeros((2,240,240))
    x[0,:,:] = npy[0]
    x[1,:,:] = npy[1]
    masks.append(npy[2])
    imgs.append(x)
imgs = np.array(imgs)
masks = np.array(masks)

print('Loading Model')
model = models.UNET()
model.load_weights('weights-full-best.h5')

print('Predicting')

predictions = main.predictor(model=model, batch_size=8, scan=imgs)
# predictions = model.predict(imgs, batch_size=8, verbose=1)

y_scores = predictions.reshape(predictions.shape[0]*predictions.shape[1]*predictions.shape[2]*predictions.shape[3], 1)
print(y_scores.shape)

y_true = masks.reshape(masks.shape[0]*masks.shape[1]*masks.shape[2]*masks.shape[3], 1)

y_scores = np.where(y_scores>0.5, 1, 0)
y_true   = np.where(y_true>0.5, 1, 0)

import os
os.mkdir('./output')
output_folder = 'output/'

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0] 
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(output_folder+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print ("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(output_folder+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()

# Save 10 results with error rate lower than threshold
threshold = 300
predictions = np.where(predictions>0.5, 1, 0)
masks     = np.where(masks>0.5, 1, 0)
good_prediction = np.zeros([predictions.shape[0],1], np.uint8)
id_m = 0
for idx in range(predictions.shape[0]):
    esti_sample = predictions[idx]
    true_sample = masks[idx]
    esti_sample = esti_sample.reshape(esti_sample.shape[0]*esti_sample.shape[1]*esti_sample.shape[2], 1)
    true_sample = true_sample.reshape(true_sample.shape[0]*true_sample.shape[1]*true_sample.shape[2], 1)
    er = 0
    for idy in range(true_sample.shape[0]):
        if esti_sample[idy] != true_sample[idy]:
           er = er +1
    if er <threshold:
       good_prediction[id_m] = idx    
       id_m += 1   

fig,ax = plt.subplots(10,3,figsize=[15,15])

for idx in range(10):
    ax[idx, 0].imshow(np.uint8(imgs[good_prediction[idx,0]]))
    ax[idx, 1].imshow(np.squeeze(masks[good_prediction[idx,0]]), cmap='gray')
    ax[idx, 2].imshow(np.squeeze(predictions[good_prediction[idx,0]]), cmap='gray')

plt.savefig(output_folder+'sample_results.png')
