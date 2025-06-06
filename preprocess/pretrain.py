import os
import pandas as pd

dataset   = "AV-Speech"
data_path = os.path.expanduser(f'/home/kh31/wyf/3AI25/AVF-MAE/pretrained-datasets/{dataset}')

split_dir = os.path.join(data_path, 'labels') #存放标签的地址
video_dir = os.path.join(data_path, 'videos') #存放视频的地址

audio_dir = os.path.join(data_path, 'audio_16k') #音频的地址

num_splits = 1 # 我们只分一折
splits     = range(1, num_splits + 1)

#------------------------------------#
# 进行标签与映射文件的转换
#------------------------------------#
labels    = ['dummy']
label2idx = {l:idx for idx, l in enumerate(labels)}


#-----------------------#
# read split file
# only once
#-----------------------#
for split in splits:
    #----------------------------#
    # 确定文件的存放地址并进行创建
    #----------------------------#
    save_dir = f'/home/kh31/wyf/3AI25/AVF-MAE/saved/data/AV-Speech-Final/audio_visual'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #------------------------------------#
    # 将原始文件给读取进来
    # 并存到字典里面
    # 同时定义一个空列表去存数据
    #------------------------------------#
    train_split_file = os.path.join(split_dir, f'set_{split}/all_samples.txt')
    df               = pd.read_csv(train_split_file, header=None, delimiter=' ')
    train_label_dict = dict(zip(df[0], df[1]))
    train_label_list = []


    for v, l in train_label_dict.items(): # example: 00025.mp4 anger
        sample_name  = v.split('.')[0]
        video_file   = os.path.join(video_dir, v) # 这里是因为没有分帧 直接写文件名称即可

        label_idx    = label2idx[l]
        train_label_list.append([video_file, label_idx])

    #---------------#
    # 确定总样本数
    #---------------#
    total_samples = len(train_label_list)
    print(f'Total samples in split {split}: {total_samples}, train={len(train_label_list)}')

    #----------------------#
    # 将结果写入train.csv
    #----------------------#
    new_train_split_file = os.path.join(save_dir, f'AV-Speech-Final-av-pretrain.csv')
    df                   = pd.DataFrame(train_label_list)
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')


