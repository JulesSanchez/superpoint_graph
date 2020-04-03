import json
import matplotlib.pyplot as plt 
import numpy as np 

PATH = "results/ParisLille-3D/trainval_best/trainlog.json"
CM = "results/ParisLille-3D/trainval_best/pointwise_cm.npy"
val_loss = []
tr_loss = []
tr_IOU = []
val_IOU = []


# with open(PATH) as json_file:
#     data = json.load(json_file)
#     for i in range(len(data)):
#         #val_IOU.append(data[i]["avg_iou_val"])
#         val_loss.append(data[i]["loss_val"])
#         tr_loss.append(data[i]["loss"])
#         tr_loss.append(data[i]["avg_iou"])

cm = np.load(CM)
cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+1e-7)

target_names = ["unclassified","ground","building","pole","bollard","trash can","barrier","pedestrian","car","vegetation"]

import seaborn as sn
sn.heatmap(cm)
plt.show()
