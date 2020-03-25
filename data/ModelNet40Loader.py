import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _get_data_files(list_filename):
    with open(list_filename) as f:
        # return [line.rstrip()[5:] for line in f]
        return np.array([line.rstrip() for line in f])


def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def _get_point_file(point_filename):
    with open(point_filename) as f:
        return np.array([line.rstrip().split() for line in f])


def _split_data(data, label, val=False):
    # num_example = data.shape[0]
    num_example = len(data)
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data, label = (data[arr], label[arr])
    if val:
        ratio0, ratio1 =0.8, 0.9
        s0 = np.int(num_example*ratio0)
        s1 = np.int(num_example*ratio1)
        # samples splited
        x_train = data[:s0]
        y_train = label[:s0]
        x_val = data[s0:s1]
        y_val = label[s0:s1]
        x_test = data[s1:]
        y_test = label[s1:]
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        ratio = 0.9
        s = np.int(num_example*ratio)
        x_train = data[:s]
        y_train = label[:s]
        x_test = data[s:]
        y_test = label[s:]
        return x_train, y_train, x_test, y_test

def getDataFiles(list_filename):
    # return [line.rstrip() for line in open(list_filename)]
    files = [line.rstrip() for line in open(list_filename)]
    trian_files = files[:-2]
    test_file = []
    test_f = files[-1]
    test_file.append(test_f)
    return trian_files, test_file


class ModelNet40Cls(data.Dataset):

    def __init__(
            self, num_points, root, transforms=None, train=True):
        super().__init__()

        self.transforms = transforms

        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)

        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        
        return current_points, label

    def __len__(self):
        return self.points.shape[0]


class myData(data.Dataset):

    def __init__(self, num_points, root, transforms=None, train=True):
        super().__init__()

        self.transforms = transforms
        self.root = os.path.abspath(root)
        self.data_dir = os.path.join(self.root, 'bos_uniformed')

        self.train, self.num_points = train, num_points

        if train:
            self.data = _get_data_files(os.path.join(self.root, 'bosphorus_id_train.txt'))
            self.label = _get_data_files(os.path.join(self.root, 'bosphorus_label_train.txt')).astype(float)
        else:
            self.data = _get_data_files(os.path.join(self.root, 'bosphorus_id_test.txt'))
            self.label = _get_data_files(os.path.join(self.root, 'bosphorus_label_test.txt')).astype(float)

        # self.train_data, self.train_label, self.test_data, self.test_label = _split_data(self.data, self.label)


    def __getitem__(self, idx):
        points_path = os.path.join(self.data_dir, self.data[idx])
        points = _get_point_file(points_path).astype(float)

        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = points[pt_idxs].copy()
        label = torch.from_numpy(np.array(self.label[idx])).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1,0,0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
