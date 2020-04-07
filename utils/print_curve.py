import json
import matplotlib.pyplot as plt 
import numpy as np 

PATH = "results/ParisLille-3D/trainval_best/crossentropy_train.json"
CM = "results/ParisLille-3D/trainval_best/crossentropy.npy"
val_loss = []
tr_loss = []
tr_IOU = []
val_IOU = []


with open(PATH) as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        val_IOU.append(data[i]["avg_iou_val"])
        val_loss.append(data[i]["loss_val"])
        tr_loss.append(data[i]["loss"])
        tr_IOU.append(data[i]["avg_iou"])

plt.plot([i for i in range(len(val_IOU))], tr_loss, color = 'r', label='training loss')  
plt.plot([i for i in range(len(val_IOU))], val_loss, color = 'b', label='val loss')  
plt.legend()
plt.title('Loss across epochs')
plt.show()

plt.plot([i for i in range(len(val_IOU))], tr_IOU, color = 'r', label='training IoU')  
plt.plot([i for i in range(len(val_IOU))], val_IOU, color = 'b', label='val IoU')  
plt.legend()
plt.title('average IoU across epochs')
plt.show()

cm = np.load(CM)
cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+1e-7)

target_names = ["ground","building","pole","bollard","trash can","barrier","pedestrian","car","vegetation"]

import seaborn as sn
sn.heatmap(cm[:9,:9], xticklabels=target_names, yticklabels = target_names)
plt.show()
