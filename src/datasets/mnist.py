from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
import glob,os,torchvision
import numpy as np
from .preprocessing import get_target_label_idx, global_contrast_normalization
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
root_ = ['/home/mhkim/oc_datatsets/classifier_OC/train_oc',
         '/home/mhkim/oc_datatsets/classifier_OC/test_oc_',
         '/home/mhkim/oc_datatsets/classifier_OC/clear_person_ssim']

def Check_32pixel_images(path_full):
    list_path = []
    for _path in path_full:
        img = cv2.imread(_path)
        if min(img.shape[0], img.shape[1]) < 32:
            pass
        else:
            list_path.append(_path)
    return list_path

class MNIST_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=1):
        super().__init__(root)
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        self.train_set = CustumDataset(root_dir=root_[2],target_transform = target_transform,normal_class = normal_class,
                                   input_size=128,
                                   train=True,
                                   padding=False, inbalance=False, aug_falldown=False)
        print("self.normal_classes : ",self.normal_classes)
        train_idx_normal = get_target_label_idx(self.train_set.labels, self.normal_classes)
        #self.train_set = Subset(train_set, train_idx_normal) # 사람 데이터만 train_set으로 지정
        print("train_set.__getitem__(train_idx_normal[0])" , self.train_set.__getitem__(train_idx_normal[0]))
        self.test_set = CustumDataset(root_dir=root_[1],target_transform = target_transform,normal_class = normal_class,
                                 input_size=128,
                                 train=False,
                                 padding=False, inbalance=True, aug_falldown=False)


class CustumDataset(Dataset):
    def __init__(self, root_dir, input_size, target_transform = None,normal_class= 0,train = True, padding = True, inbalance = False, aug_falldown = False,
                 bright_ness = 0.2, hue = 0.15, contrast = 0.15, random_Hflip = 0.3, rotate_deg = 20,temp=False):
        min_max = [(-1.4646, 1.6818),(-3.8436, 3.3011)]
                   #(-34.924463588638204, 14.419298165027628)]
        self.train = train
        self.target_transform = target_transform
        self.orig_normal_paths, self.orig_back_paths = None,None
        if train :
           self.orig_normal_paths = glob.glob(root_dir + "/*.jpg")
        else :
           self.orig_back_paths = glob.glob(os.path.join(root_dir, "background") + "/*.jpg") # 0
           self.orig_normal_paths = glob.glob(os.path.join(root_dir, "normal") + "/*.jpg") # 1
        if self.orig_back_paths:
            self.back_paths = []
            self.back_paths = Check_32pixel_images(self.orig_back_paths)
        self.normal_paths = []
        self.normal_paths = Check_32pixel_images(self.orig_normal_paths)

        self.labels = []
        self.total_paths=[]
        if not train :
            self.labels = [-1] * len(self.back_paths) + [1] *len(self.normal_paths)
            for p in self.back_paths:self.total_paths.append(p)
        else :
            self.labels = [1] * len(self.normal_paths)
        for p in self.normal_paths:self.total_paths.append(p)
        np.ravel(self.total_paths)
        transform = []
        transform.append(torchvision.transforms.ToTensor())
        transform.append(transforms.Normalize([min_max[normal_class][0]] * 3,
                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3))
        transform.append(torchvision.transforms.Resize((input_size, input_size)))
        self.transform = torchvision.transforms.Compose(transform)
        print(self.transform)

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, index):
        img = Image.open(self.total_paths[index])
        img = self.transform(img)
        img, target = img, self.labels[index]
        return img, target, index  # only line changed
