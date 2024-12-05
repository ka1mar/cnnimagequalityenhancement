import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import os
from YUV_RGB import yuv2rgb
from MODEL_TORCH import EnhancerModel, ImageDataset, psnr

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2



def yuv2rgb(Y,U,V,fw,fh):
    U_new = cv2.resize(U, (fw, fh),cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), cv2.INTER_CUBIC)
    U = U_new
    V = V_new
    Y = Y
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    for m in range(fh):
        for n in range(fw):
            if (rf[m, n] > 255):
                rf[m, n] = 255
            if (gf[m, n] > 255):
                gf[m, n] = 255
            if (bf[m, n] > 255):
                bf[m, n] = 255
            if (rf[m, n] < 0):
                rf[m, n] = 0
            if (gf[m, n] < 0):
                gf[m, n] = 0
            if (bf[m, n] < 0):
                bf[m, n] = 0
    r = rf
    g = gf
    b = bf
    return r, g, b


def FromFolderYuvToFolderPNG(folderyuv,folderpng,fw,fh):
    dir_list = os.listdir(folderpng)
    for name in dir_list:
        os.remove(folderpng+name)
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')
    #list of patch left-top coordinates
    numdx = (fw-patchsize)//patchstep
    dx = np.zeros(numdx)
    numdy = (fh - patchsize) // patchstep
    dy = np.zeros(numdy)
    for i in range(numdx):
        dx[i]=i*patchstep
    for i in range(numdy):
        dy[i]=i*patchstep
    dx = dx.astype(int)
    dy = dy.astype(int)
    Im = np.zeros((patchsize, patchsize,3))
    dir_list = os.listdir(folderyuv)
    pngframenum = 0
    for name in dir_list:
        fullname = folderyuv + name
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size= fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2*size)//(fw*fh*3)
            frames=100
            print(fullname,frames)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                r,g,b = yuv2rgb (Y,U,V,fw,fh)
                for i in range(numdx):
                    for j in range(numdy):
                        Im[:, :, 0] = b[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 1] = g[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 2] = r[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        pngfilename = "%s/%i.png" % (folderpng,pngframenum)
                        cv2.imwrite(pngfilename, Im)
                        pngframenum = pngframenum + 1
            fp.close()
    return (pngframenum-1)


def TrainImageEnhancementModel(folderRaw, folderComp, folderRawVal, folderCompVal):
    print('Loading datasets...')
    transforms_train = A.Compose([
        A.D4(p=0.5),
        ToTensorV2()
    ],
    additional_targets={'comp_image': 'image'})

    augment_train = A.Compose([
        A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
    ])

    transforms_test = A.Compose([
        ToTensorV2()
    ],
    additional_targets={'comp_image': 'image'})

    train_dataset = ImageDataset(folderRaw, folderComp, transform=transforms_train, augment=augment_train)
    val_dataset = ImageDataset(folderRawVal, folderCompVal, transform=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enhancer = EnhancerModel().to(device)

    optimizer = optim.AdamW(enhancer.parameters(), lr=0.0003)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.MSELoss()

    num_epochs = 200
    best_psnr = 0
    prev_name = "./prev_name.pt"
    for epoch in range(num_epochs):
        # Training
        enhancer.train()
        train_loss = 0.0
        for batch in train_loader:
            comp_images, raw_images = batch
            comp_images, raw_images = comp_images.to(device), raw_images.to(device)
            
            optimizer.zero_grad()
            outputs = enhancer(comp_images)
            loss = criterion(outputs, raw_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        enhancer.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for batch in val_loader:
                comp_images, raw_images = batch
                comp_images, raw_images = comp_images.to(device), raw_images.to(device)
                outputs = enhancer(comp_images)

                loss = criterion(outputs, raw_images)
                val_loss += loss.item()

                val_psnr += psnr(outputs, raw_images).item()

            val_loss /= len(val_loader)
            val_psnr /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss*100:.6f}, Validation Loss: {val_loss*100:.6f}, Validation PSNR: {val_psnr:.4f}')

        # Обновление learning rate
        scheduler.step()

        if val_psnr > best_psnr and epoch > 50:
            name = f"./enhancer_{epoch}_{val_psnr:.4f}.pt"
            torch.save(enhancer.state_dict(), name)
            best_psnr = val_psnr

            if os.path.exists(prev_name):
                os.remove(prev_name)

            prev_name = name
    
    return enhancer


#Frame size of training data
w=480
h=320
#patch size and patch step for training
patchsize = 40
patchstep = 20

#test folders for raw and compressed in yuv and png formats
testfolderRawYuv = './testrawyuv/'
testfolderRawPng = './testrawpng/'
testfolderCompYuv = './testcompyuv/'
testfolderCompPng = './testcomppng/'

#train folders for raw and compressed in yuv and png formats
trainfolderRawYuv = './trainrawyuv/'
trainfolderRawPng = './trainrawpng/'
trainfolderCompYuv = './traincompyuv/'
trainfolderCompPng = './traincomppng/'


def train(PrepareDataSetFromYUV):
    if PrepareDataSetFromYUV:
        FromFolderYuvToFolderPNG(testfolderRawYuv,testfolderRawPng,w,h)
        FromFolderYuvToFolderPNG(testfolderCompYuv,testfolderCompPng,w,h)
        FromFolderYuvToFolderPNG(trainfolderRawYuv,trainfolderRawPng,w,h)
        FromFolderYuvToFolderPNG(trainfolderCompYuv,trainfolderCompPng,w,h)
    enhancer = TrainImageEnhancementModel(trainfolderRawPng,trainfolderCompPng,testfolderRawPng,testfolderCompPng)
    return enhancer

train(PrepareDataSetFromYUV=False)