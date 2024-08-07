import os

import numpy as np
import matplotlib.pyplot as plt
#import pydicom
from os import path
import torch

# dcm=pydicom.dcmread("E:\zxm\AAPM-Mayo\L067_4L_100kv_fulldose.1.00001.dcm")
# print(dcm)
#
# image = np.load(rb"E:\zxm\ME\SAED_CNN\data\L067_0001_input.npy")
# print(type(image))
# plt.imshow(image)
# plt.show()
#
# print(image.shape)
#

loss_data = np.load("/home/lhl/ct/ME/KD-online/save_mutual1/SAED/loss_1000_iter.npy")
print(loss_data)

#def scaner_file(url):
#    file = os.listdir(url)
#    i = 0
#    for f in file:
#        i += 1
#        real_url = path.join(url, f)
#        print(real_url)
#        real_url_npy = np.load(real_url)
#        print(real_url_npy.shape)
#        print(i)

#scaner_file('/home/lhl/ct/ME/KD-offline/data')
'''
SAED_TL = np.load('/home/wjy/zxm/SAED_CNN/cocatenate-best/net2/textureloss4-best/MSE_loss_1000_iter.npy')#('/home/wjy/zxm/SAED_CNN/save/loss_1000_iter.npy')
SAED_MSE = np.load('/home/wjy/zxm/SAED_CNN/cocatenate-best/net1/coca_nosa1-better/loss_1000_iter.npy')
RED_MSE = np.load('/home/wjy/zxm/RED_CNN/net1/save_lr1e-4_adam-default/loss_1000_iter.npy')

x1 = SAED_TL
x2 = SAED_MSE
x3 = RED_MSE
y = range(0,1000)
plt.plot(y,x2,color='blue', label='RED_MSE',ls='--')
plt.plot(y,x1,color='red', label='SAED_TL',ls='-')
plt.plot(y,x2,color='green', label='SAED_MSE',ls='-.')

plt.legend()

plt.title('Train Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()
plt.savefig('train_loss.jpg')

'''
