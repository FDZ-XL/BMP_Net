# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/29/029 14:41
@Author  : NDWX
@File    : train.py
@Software: PyCharm
"""

import glob
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import warnings
import numpy as np
import torch

from BMP_Net.code.model.arch import UNext
import BMP_Net.code.segmentation_models_pytorch as smp
from BMP_Net.code.model.biformerT4unetBT.as3mlplknojw import Unet

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context      #下载权重文件
from BMP_Net.code.utils.dataset import build_dataloader
from BMP_Net.code.utils.tools import PolynomialLRDecay

# from paperunetpp import NestedUNet
# from code.nets.r2unet import R2U_Net
# from code.nets.segnet import SegNet
# from code.nets.attention_unet import AttU_Net

warnings.filterwarnings('ignore')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 固定随机种子
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 加载模型
def load_model(DEVICE,Class):
    # model = Unet(encoder="Swin-B", num_classes=classes,
    #              pretrained_model_path="model_data/swin_base_patch4_window7_224.pth")
    model = Unet(num_classes=2)
    #model=NestedUNet()
    #model=Unet(in_ch=3,out_ch=2)
    #model=PSPNet(num_classes=2, backbone="mobilenet", downsample_factor=8, pretrained=True, aux_branch=False)
    #model=DeepLab(num_classes=2, backbone="xception", downsample_factor=8, pretrained=True)
    # model=smp.Unet(
    #         encoder_name="vgg16",        # 选择解码器, 例如 mobilenet_v2 或 efficientnet-b7
    #         encoder_weights="imagenet",     # 使用预先训练的权重imagenet进行解码器初始化
    #         in_channels=3,                  # 模型输入通道（1个用于灰度图像，3个用于RGB等）
    #         classes=2,                      # 模型输出通道（数据集所分的类别总数）
    #     )
    model.to(DEVICE)
    return model


if __name__ == '__main__':
    for j in range(5):
        random_seed = 1
        num_epochs = 50
        batch_size = 4
        channels = 3
        lr = 1e-4
        setup_seed(random_seed)

        train_dataset = [sorted(glob.glob("../data/CHN6-CUG512/train/image/*.jpg")),
                         sorted(glob.glob("../data/CHN6-CUG512/train/label/*.png"))]

        val_dataset = [sorted(glob.glob("../data/CHN6-CUG512/val/image/*.jpg")),
                       sorted(glob.glob("../data/CHN6-CUG512/val/label/*.png"))]

        # train_dataset, val_dataset = split_dataset(dataset, random_seed)
        train_loader, valid_loader = build_dataloader(train_dataset, val_dataset, int(batch_size))

        # model_save_path = "../user_data/model_data/swinbunetb4lktsa0.pth"
        model = load_model(DEVICE,Class=2)

        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])
        scheduler = PolynomialLRDecay(optimizer, num_epochs, 1e-5)

        loss = torch.nn.CrossEntropyLoss()
        loss.__name__ = "CrossEntropyLoss"

        metrics = [
            smp.utils.metrics.IoU(activation='argmax2d'),
            smp.utils.metrics.Fscore(activation='argmax2d')
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        max_score = 0
        for i in range(num_epochs):
            print('\nEpoch: {}'.format(i + 1))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            scheduler.step()
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                print('max_score', max_score)
                torch.save(model, '../user_data/duibishiyan/chn6/unext/unextls'+str(j+1)+'.pth')
                print('Model saved!')
        if i == 49:
            torch.save(model, '../user_data/duibishiyan/chn6/unext/unextls'+str(j+1)+'zh'+'.pth')
            print("savezuihouyige")
        print("****************************************第"+str(j+1)+"个循环***************************")
        time.sleep(30 * 60)



