from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
import glob,os,torchvision
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
root_ = ['/home/mhkim/oc_datatsets/classifier_OC/train_oc',
         '/home/mhkim/oc_datatsets/classifier_OC/test_oc']
def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()
def global_contrast_normalization(x, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
      #  print(x_scale)

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale
    print(x.max() , x.min())
   # print('x : {}'.format(x))
    return x

root_ = ['/home/mhkim/oc_datatsets/classifier_OC/train_oc',
         '/home/mhkim/oc_datatsets/classifier_OC/test_oc']

def Check_32pixel_images(path_full):
    list_path = []
    for _path in path_full:
        img = cv2.imread(_path)
        if min(img.shape[0], img.shape[1]) < 32:
            pass
        else:
            list_path.append(_path)
    return list_path
def AddFirstPriorityData(root,path_normal,path_back):
    _path_normal = glob.glob(os.path.join(root_[0],path_normal) + "/*.jpg") # 0
    _path_back = glob.glob(os.path.join(root_[0],path_back) + "/*.jpg") # 0
    normal=Check_32pixel_images(_path_normal)
    back=Check_32pixel_images(_path_normal)
    return normal,back



class MNIST_Dataset():
    def __init__(self, root: str, normal_class=0):
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = CustumDataset(root_dir=root_[0],target_transform = target_transform,normal_class = normal_class,
                                   input_size=128,
                                   train=True,
                                   padding=False, inbalance=False, aug_falldown=False)
        self.normal_classes = tuple([normal_class])
        train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)
        self.test_set = CustumDataset(root_dir=root_[1],target_transform = target_transform,normal_class = normal_class,
                                 input_size=128,
                                 train=False,
                                 padding=False, inbalance=True, aug_falldown=False)
        print(train_set.getGCN())
        print(self.test_set.getGCN())

class CustumDataset(Dataset):
    def __init__(self, root_dir, input_size, target_transform = None,normal_class= 0,train = True, padding = True, inbalance = False, aug_falldown = False,
                 bright_ness = 0.2, hue = 0.15, contrast = 0.15, random_Hflip = 0.3, rotate_deg = 20,temp=False):
        self.train = train
        self.target_transform = target_transform
        self.orig_back_paths = glob.glob(os.path.join(root_dir, "background") + "/*.jpg") # 0
        #self.orig_fall_paths = glob.glob(os.path.join(root_dir, "falldown") + "/*.jpg")  # 1
        self.orig_normal_paths = glob.glob(os.path.join(root_dir, "normal") + "/*.jpg") # 2
        self.back_paths = []
        self.fall_paths = []
        self.normal_paths = []
        self.back_paths = Check_32pixel_images(self.orig_back_paths)
        self.normal_paths = Check_32pixel_images(self.orig_normal_paths)
        self.labels = [0] * len(self.back_paths) + [1] *len(self.normal_paths)
        self.total_paths=[]
        for p in self.back_paths:self.total_paths.append(p)
        for p in self.normal_paths:self.total_paths.append(p)
        np.ravel(self.total_paths)
        transform = []
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Resize((input_size, input_size)))
        self.transform = torchvision.transforms.Compose(transform)
        print(self.transform)

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, index):
        img = Image.open(self.total_paths[index])
        img = self.transform(img)
        img, target = img, self.labels[index]
        #print(target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index  # only line changed
    def getGCN(self):
        x_ten = None
        for i in range(len(self.total_paths)):
            x_ten = self.__getitem__(i)[0]
            #x_ten = x_ten.numpy()
            break
            # if x_ten is None:
            #     x_ten = self.__getitem__(i)[0]
            # else:
            #     x_ten = torch.cat(x_ten[0])
            #     # x_ten.cat(self.__getitem__(i))

        #x_ten = torch.stack(x_ten)
        return global_contrast_normalization(x_ten, scale = 'l1')

#train_dataset = MNIST_Dataset(root=root_[0], normal_class=1)

#val_dataset = MNIST_Dataset(root=root_[1], normal_class=1)
_path_normal = glob.glob(os.path.join(root_[0], 'normal') + "/*.jpg")  # 0
img1 = Image.open(_path_normal[0])
transform = []
transform.append(torchvision.transforms.ToTensor())
transform.append(torchvision.transforms.Resize((128, 128)))
transform = torchvision.transforms.Compose(transform)
t_img1 = transform(img1)
global_contrast_normalization(t_img1,'l1')


_path_back = glob.glob(os.path.join(root_[0], 'background') + "/*.jpg")  # 0
img2 = Image.open(_path_back[0])
t_img2 = transform(img2)
global_contrast_normalization(t_img2,'l1')