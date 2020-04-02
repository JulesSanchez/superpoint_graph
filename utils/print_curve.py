import json
import matplotlib.pyplot as plt 

PATH = "results/ParisLille-3D/trainval_best/trainlog.json"
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
        tr_loss.append(data[i]["avg_iou"])
