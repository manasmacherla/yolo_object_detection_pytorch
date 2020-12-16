from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride # getting the downsampled grid size 
    bbox_attrs = 5+num_classes #bx, by, bw, by, objectness value, class confidences
    num_anchors = len(anchors) #number of anchor boxes

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous() # call to contiguous maps the indices with the correct memory pointer
    #calling transpose only changes the map b/w indices and locations but it still needs to be changed 
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors] #dividing by the stride to match the downsampled grid size

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # capping the offset values of bbox for each grid cell to 1 by using sigmoid
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid) #a-x coords, b-y coords

    x_offset = torch.FloatTensor(a).view(-1, 1) # making the x_offset values a 1D column tensor
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    #for 13x13 gridcell - concat opn 169x2 size, repeat opn - 169x6, view opn - 507x2, unsqueeze opn - 1x507x2
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset #adding the offsets for each gridcell

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda() #changes the type into torch.cuda.FloatTensor for cuda operations

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes]) # sigmoid opn on the class confidence scores 
    prediction[:,:,:4] *= stride #upsampling the bbox size back to the original input dim

    return prediction

def write_results(prediction, confidence, num_classes, nms_conf):

    # inequality opn - batchsize x10647, float opn converts into 0 or 1 bx10647, unsqueeze opn adds column(2) dim bx10647x1
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        
        #gives a tuple of indices and values in the axis specified https://pytorch.org/docs/stable/generated/torch.max.html
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1) #adding a dimension to concatenate later
        max_conf_score = max_conf_score.float().unsqueeze(1) 
        seq = (image_pred[:,:5], max_conf, max_conf_score) #(10647, 5), (10647, 1), (10647, 1) 
        image_pred = torch.cat(seq, 1) 