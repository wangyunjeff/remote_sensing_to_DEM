import os
import random

seg_file_path = 'RemoteSensingImage'
save_path = 'ImageSets'

trainval_percent = 0.9
train_percent = 0.9

files = os.listdir(seg_file_path)
total_seg = []

for file in files:
    if file.endswith('.png'):
        total_seg.append(file)

num = len(total_seg)

tv_num = int(num * train_percent)
tr_num = int(tv_num * train_percent)
tv = random.sample(range(num), tv_num)
tr = random.sample(range(num), tr_num)

print("train and val size", tv_num)
print("train size", tr_num)

ftrainval = open(os.path.join(save_path, 'trainval.txt'), 'w')
ftest = open(os.path.join(save_path, 'test.txt'), 'w')
ftrain = open(os.path.join(save_path, 'train.txt'), 'w')
fval = open(os.path.join(save_path, 'val.txt'), 'w')

for i in range(num):
    name = total_seg[i][:-4] + '\n'
    if i in tv:
        ftrainval.write(name)
        if i in tr:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
