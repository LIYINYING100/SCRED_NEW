import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ct_dataset(Dataset):  # 继承dataset类
    def __init__(self, mode, load_mode, saved_path,saved_path1, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))  # 数据集路径为saved_path
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        input_path1 = sorted(glob(os.path.join(saved_path1, '*_input.npy')))  # 数据集路径为saved_path
        target_path1 = sorted(glob(os.path.join(saved_path1, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            input_1 = [f for f in input_path1 if test_patient not in f]
            target_1 = [f for f in target_path1 if test_patient not in f]
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
                self.input_1 = input_1
                self.target_1 = target_1
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else: # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img1, target_img1 = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)
            input_img1, target_img1 = np.load(input_img1), np.load(target_img1)
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            input_img1 = self.transform(input_img)
            target_img1 = self.transform(target_img)
        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            input_patches1, target_patches1 = get_patch(input_img1,
                                                      target_img1,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches,input_patches1, target_patches1)
        else:
            return (input_img, target_img)


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode=None, load_mode=0,
               saved_path=None,saved_path1=None, test_patient=None,
               patch_n=None, patch_size=None,shuffle=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, saved_path,saved_path1,test_patient, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    return data_loader

