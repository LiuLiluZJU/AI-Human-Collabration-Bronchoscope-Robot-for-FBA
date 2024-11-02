import os
import sys
import h5py
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image


class AlignDataSet(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSet, self).__init__()
        self.dataset_root = dataset_dir
        self.file_name = os.path.join(self.dataset_root, "actions.txt")
        self.rgb_root = os.path.join(self.dataset_root, "rgb_images")
        with open(self.file_name, 'r') as f:
            data_list = f.readlines()
        self.data_list = data_list
        print(self.data_list)
        self.dataset_size = len(self.data_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, rgb_root, data):
        data_split = data.rstrip('\n').split(' ')
        image_name = data_split[0]
        action = np.array([float(data_split[1]), float(data_split[2]), float(data_split[3])]) * 1000
        command = np.array([float(data_split[4]), float(data_split[5]), float(data_split[6]), float(data_split[7]), float(data_split[8])])
        image = cv2.imread(os.path.join(rgb_root, image_name))
        return image, command, action
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        image, command, action = self.load_file(self.rgb_root, self.data_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_list[item]


class AlignDataSetSplit(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSetSplit, self).__init__()
        self.trajectory_list = os.listdir(dataset_dir)
        self.data_full_list = []
        for trajectory_name in self.trajectory_list:
            file_name = os.path.join(dataset_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(dataset_dir, trajectory_name, "rgb_images", data[0])
                    self.data_full_list.append(data)
        print(self.data_full_list)
        self.dataset_size = len(self.data_full_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        return image, command, action
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_full_list[item][0]


class AlignDataSetDagger(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSetDagger, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        return image, command, action

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_full_list[item][0]


class AlignDataSetDaggerAug(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerAug, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        return image, command, action

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        return image_tensor, command, action, self.data_full_list[item][0]


class AlignDataSetDaggerWithDepthAug(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerWithDepthAug, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, command, action, depth

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, command, action, depth = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, command, action, depth_tensor, self.data_full_list[item][0]


class AlignDataSetDaggerHighLevel(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerHighLevel, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        ref_rgb_path = rgb_path.replace("rgb_images", "ref_rgb_images")
        condition_path = rgb_path.replace("rgb_images", "condition_images")
        targets_path = rgb_path.replace("rgb_images", "targets_images")
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        ref_image = cv2.imread(ref_rgb_path)
        ref_image = ref_image[:, :, ::-1]  # BGR to RGB
        condition = cv2.imread(condition_path)
        condition = condition[:, :, ::-1]  # BGR to RGB
        # condition = cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)
        targets = cv2.imread(targets_path)
        targets = cv2.cvtColor(targets, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, ref_image, condition, targets, depth, command

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, ref_image, condition, targets, depth, command = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        ref_image = cv2.resize(ref_image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        # cv2.imshow("image", image)
        # cv2.imshow("ref_image", ref_image)
        # cv2.imshow("condition", condition)
        # cv2.imshow("targets", targets)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(1)
        image_PIL = Image.fromarray(image)
        ref_image_PIL = Image.fromarray(ref_image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            ref_image_tensor = self.transforms_train(ref_image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
            ref_image_tensor = self.transforms_eval(ref_image_PIL)
        condition_PIL = Image.fromarray(condition)
        condition_tensor = self.transforms_eval(condition_PIL)
        targets_PIL = Image.fromarray(targets)
        targets_tensor = self.transforms_eval(targets_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, ref_image_tensor, condition_tensor, targets_tensor, depth_tensor, command, self.data_full_list[item][0]
    

class AlignDataSetDaggerSingleLevel(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerSingleLevel, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        ref_rgb_path = rgb_path.replace("rgb_images", "ref_rgb_images")
        condition_path = rgb_path.replace("rgb_images", "condition_images")
        targets_path = rgb_path.replace("rgb_images", "targets_images")
        action = np.array([float(data[13]), float(data[14])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12])])
        stop = float(data[12])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        ref_image = cv2.imread(ref_rgb_path)
        ref_image = ref_image[:, :, ::-1]  # BGR to RGB
        condition = cv2.imread(condition_path)
        condition = condition[:, :, ::-1]  # BGR to RGB
        # condition = cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)
        targets = cv2.imread(targets_path)
        targets = cv2.cvtColor(targets, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, ref_image, condition, targets, depth, action, stop

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, ref_image, condition, targets, depth, action, stop = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        ref_image = cv2.resize(ref_image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        # cv2.imshow("image", image)
        # cv2.imshow("ref_image", ref_image)
        # cv2.imshow("condition", condition)
        # cv2.imshow("targets", targets)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(1)
        image_PIL = Image.fromarray(image)
        ref_image_PIL = Image.fromarray(ref_image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            ref_image_tensor = self.transforms_train(ref_image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
            ref_image_tensor = self.transforms_eval(ref_image_PIL)
        condition_PIL = Image.fromarray(condition)
        condition_tensor = self.transforms_eval(condition_PIL)
        targets_PIL = Image.fromarray(targets)
        targets_tensor = self.transforms_eval(targets_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, ref_image_tensor, condition_tensor, targets_tensor, depth_tensor, action, stop, self.data_full_list[item][0]
    

class AlignDataSetDaggerSingleLevelBalanced(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerSingleLevelBalanced, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        fh_cls = open(os.path.join(root_list, 'train_cls_list.txt'), 'r')
        label_cls_list = []
        for line in self.data_full_list:
            line = line.rstrip()
            words = line.split()
            label_cls_list.append((words[0], int(words[1])))
        
        weights_cls = self.get_weights_for_balanced_classes(label_cls_list, 2)
        self.prob_cls = np.array(weights_cls) / sum(weights_cls)
    
    def __len__(self):
        return self.dataset_size
    
    def get_weights_for_balanced_classes(self, labels, nclasses):
        count = [0] * nclasses
        for item in labels:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            if count[i] != 0:
                weight_per_class[i] = N / float(count[i])
        weight = [0] * len(labels)
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val[1]]
        return weight

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)

        fh_cls = open(os.path.join(root_list, 'train_cls_list.txt'), 'r')
        label_cls_list = []
        for line in fh_cls:
            line = line.rstrip()
            words = line.split()
            label_cls_list.append((words[0], int(words[1])))
        
        weights_cls = self.get_weights_for_balanced_classes(label_cls_list, 2)
        self.prob_cls = np.array(weights_cls) / sum(weights_cls)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        ref_rgb_path = rgb_path.replace("rgb_images", "ref_rgb_images")
        condition_path = rgb_path.replace("rgb_images", "condition_images")
        targets_path = rgb_path.replace("rgb_images", "targets_images")
        action = np.array([float(data[13]), float(data[14])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12])])
        stop = float(data[12])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        ref_image = cv2.imread(ref_rgb_path)
        ref_image = ref_image[:, :, ::-1]  # BGR to RGB
        condition = cv2.imread(condition_path)
        condition = condition[:, :, ::-1]  # BGR to RGB
        # condition = cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)
        targets = cv2.imread(targets_path)
        targets = cv2.cvtColor(targets, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, ref_image, condition, targets, depth, action, stop

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        fn_num_cls = len(self.data_full_list)
        fn_index_cls = np.random.choice(np.arange(self.dataset_size), 1, replace=True, p=self.prob_cls)

        image, ref_image, condition, targets, depth, action, stop = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        ref_image = cv2.resize(ref_image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        # cv2.imshow("image", image)
        # cv2.imshow("ref_image", ref_image)
        # cv2.imshow("condition", condition)
        # cv2.imshow("targets", targets)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(1)

        fn_num_cls = len(self.data_full_list)
        fn_index_cls = np.random.choice(np.arange(fn_num_cls), 1, replace=True, p=self.prob_cls)
        fn_cls, label_cls = self.label_cls_list[fn_index_cls.item()]
        img_path_cls = os.path.join(self.root_data, 'data', fn_cls)
        frame_cls = Image.open(img_path_cls)
        frame_cls = frame_cls.convert('RGB')

        image_PIL = Image.fromarray(image)
        ref_image_PIL = Image.fromarray(ref_image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            ref_image_tensor = self.transforms_train(ref_image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
            ref_image_tensor = self.transforms_eval(ref_image_PIL)
        condition_PIL = Image.fromarray(condition)
        condition_tensor = self.transforms_eval(condition_PIL)
        targets_PIL = Image.fromarray(targets)
        targets_tensor = self.transforms_eval(targets_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, ref_image_tensor, condition_tensor, targets_tensor, depth_tensor, action, stop, self.data_full_list[item][0]


class AlignDataSetDaggerTargetsDetection(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerTargetsDetection, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        ref_rgb_path = rgb_path.replace("rgb_images", "ref_rgb_images")
        condition_path = rgb_path.replace("rgb_images", "condition_images")
        targets_path = rgb_path.replace("rgb_images", "targets_images")
        action = np.array([float(data[13]), float(data[14])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12])])
        stop = float(data[12])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        ref_image = cv2.imread(ref_rgb_path)
        ref_image = ref_image[:, :, ::-1]  # BGR to RGB
        condition = cv2.imread(condition_path)
        condition = condition[:, :, ::-1]  # BGR to RGB
        # condition = cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)
        targets = cv2.imread(targets_path)
        targets = cv2.cvtColor(targets, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, ref_image, condition, targets, depth, action, stop

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, _, _, targets, depth, _, _ = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        ref_image = cv2.resize(ref_image, (200, 200))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        targets_PIL = Image.fromarray(targets)
        targets_tensor = self.transforms_eval(targets_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, targets_tensor, depth_tensor, self.data_full_list[item][0]
    

class AlignDataSetDaggerWithDepthAugAngleMultiFrame(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerWithDepthAugAngleMultiFrame, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        action = np.array([float(data[12]), float(data[13])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        rgb_path = data[0]
        rgb_root_dir = rgb_path.rstrip(rgb_path.split("\\")[-1])
        rgb_suffix = "." + rgb_path.split("\\")[-1].split(".")[-1]
        rgb_index = int(rgb_path.split("\\")[-1].rstrip(rgb_suffix))
        image_tenosr_list = []
        depth_tensor_list = []
        for offset in range(5):
            rgb_index_cur = rgb_index - offset
            rgb_path_cur = os.path.join(rgb_root_dir, str(rgb_index_cur) + rgb_suffix)
            if os.path.exists(rgb_path_cur):
                rgb_path = rgb_path_cur
            depth_path = rgb_path.replace("rgb_images", "depth_images")
            image = cv2.imread(rgb_path)
            image = image[:, :, ::-1]  # BGR to RGB
            image = cv2.resize(image, (200, 200))
            image_PIL = Image.fromarray(image)
            if self.train_flag:
                image_tensor = self.transforms_train(image_PIL)
                # print(image_tensor.max(), image_tensor.min())
                # self.show_image_tensor(image_tensor)
            else:
                image_tensor = self.transforms_eval(image_PIL)
            image_tenosr_list.append(image_tensor)
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth_PIL = Image.fromarray(depth)
            depth_tensor = self.transforms_eval(depth_PIL)
            depth_tensor_list.append(depth_tensor)
        image_tensor = torch.cat(image_tenosr_list, dim=0)
        depth_tensor = torch.cat(depth_tensor_list, dim=0)
        return image_tensor, command, action, depth_tensor

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image_tensor, command, action, depth_tensor = self.load_file(self.data_full_list[item])
        return image_tensor, command, action, depth_tensor, self.data_full_list[item][0]


class AlignDataSetForeignBody(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetForeignBody, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        rgb_cls = float(data[1])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, depth, rgb_cls

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, depth, image_cls = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        depth = cv2.resize(depth, (200, 200))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, depth_tensor, image_cls, self.data_full_list[item][0]
