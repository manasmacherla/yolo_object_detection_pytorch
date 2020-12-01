from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    Returns a list of blocks. Each block describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0 ]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) > 0:
                blocks.append(block)
                block = {}
                block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()   

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        if (x["type"] == "convolutional"):
            activation = ["activation"]
            try:
                batch_normalize = x["batch_normalize"]
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            padding = int(x["pad"])

            if padding:
                pad = kernel_size -1 // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            x["layers"][0] = int(x["layers"][0])

            if (len(x["layers"]) == 1):
                x["layers"][0] = int(index + x["layers"][0])
                filters = output_filters[x["layers"][0]]

            elif (len(x["layers"]) > 1):
                x["layers"][0] = int(index + x["layers"][0])
                x["layers"][1] = int(x["layers"][0])
                filters = output_filters[x["layers"][0]] + output_filters[x["layers"][1]]

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(',')
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))






