# %matplotlib inline
import torch
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os  
import pickle
from PIL import Image
import numpy as np
from torch.nn.functional import softmax
# from scipy.special import softmax

import sys
sys.path.append("..") 
# import d2lzh_pytorch as d2l

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

resnet18 = torch.load('./model_resnet18-925.pkl').to(device).eval()
resnet50 = torch.load('./resnet50(923).pkl').to(device).eval()
resnet152 = torch.load('./resnet152.pkl').to(device).eval()

# 获取本次训练的人名和索引的对应关系
label = {}
with open('label.pkl','rb') as file:
    label = pickle.loads(file.read())
# print(label)
# 测试集label对应关系
import pickle
label_answer = {}
with open('label_answer.pkl','rb') as file:
    label_answer = pickle.loads(file.read())
label_answer = {value:key for key, value in label_answer.items()}

# 加载测试数据（在test目录下）
# from PIL import Image
# import numpy as np

transform18 = torchvision.transforms.Compose([
        #torchvision.transforms.Grayscale(num_output_channels=3), # 彩色图像转灰度图像num_output_channels默认1
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

        ])

transform50 = torchvision.transforms.Compose([
       # torchvision.transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize([330,330]),
        torchvision.transforms.CenterCrop([224, 224]),
        
        torchvision.transforms.ToTensor()
    ])

transform152 = torchvision.transforms.Compose([
       # torchvision.transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize([330,330]),
        torchvision.transforms.CenterCrop([224, 224]),
        
        torchvision.transforms.ToTensor()
    ])

# 生成测试结果文件
path = os.listdir('test')
r_d = {}
for f in path:
#     print('test/' + f)
    with torch.no_grad():    
        
        img = Image.open('test/' + f)
        test_imgs18 = transform18(img).unsqueeze(0)
        test_imgs50 = transform50(img).unsqueeze(0)
        test_imgs152 = transform152(img).unsqueeze(0)

        test_imgs18 = test_imgs18.to(device)
        test_imgs50 = test_imgs50.to(device)
        test_imgs152 = test_imgs152.to(device)

        y18 = resnet18(test_imgs18)
        y50 = resnet50(test_imgs50)
        y152 = resnet152(test_imgs152)


        pred18 = softmax(y18,dim=1)
        pred50 = softmax(y50,dim=1)
        pred152 = softmax(y152,dim=1)
     
        pred = 2/5 * pred18 + 3/8 * pred50 +3/8 * pred152
        index = pred.argmax(dim=1)
#     pred = torch.argmax(y, dim = 1)
        r = label_answer[label[int(index)]]
        r_d[int(f.strip('.jpg'))] = r
#     print(1)
# 写入结果文件
r_d = sorted(r_d.items(), key=lambda a:a[0])
r_d = dict(r_d)
ret = open("result.csv","w")
for key, value in r_d.items():
    print("%d,%s"%(key, value), file=ret)
ret.close()