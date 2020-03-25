import numpy as np
import os

root = '/home/user4/database/BosphorusDB/'
train_data = os.path.join(root, 'bosphorus_id_train.txt')
train_label = os.path.join(root, 'bosphorus_label_train.txt')
test_data = os.path.join(root, 'bosphorus_id_test.txt')
test_label = os.path.join(root, 'bosphorus_label_test.txt')

def _get_data_files(list_filename):
    with open(list_filename) as f:
        # return [line.rstrip()[5:] for line in f]
        return np.array([line for line in f])


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
        return  x_train, y_train, x_test, y_test


data_path = os.path.join(root, 'bosphorus_id.txt')
label_path = os.path.join(root, 'bosphorus_id_label.txt')

data = _get_data_files(data_path)
label = _get_data_files(label_path)

x_train, y_train, x_test, y_test = _split_data(data, label)

with open(train_data, 'w') as f_trd:
    f_trd.writelines(x_train)
with open(train_label, 'w') as f_trl:
    f_trl.writelines(y_train)
with open(test_data, 'w') as f_ted:
    f_ted.writelines(x_test)
with open(test_label, 'w') as f_tel:
    f_tel.writelines(y_test)


