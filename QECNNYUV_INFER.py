import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


from MODEL_TORCH import EnhancerModel as EnhancerModelTORCH
from cnnimagequalityenhancement_final.MODEL_TF import EnhancerModelTF


def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def yuv2rgb (Y,U,V,fw,fh):
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


def LoadImageEnhancementModelTF(fw,fh):
    enhancer = EnhancerModelTF(fw,fh)
    #enhancer.compile(loss='mean_squared_error',optimizer='Adam',metrics=[psnr])
    enhancer.load_weights('enhancer.weights.h5')

    return enhancer


def LoadImageEnhancementModelTORCH(fw,fh):
    enhancer = EnhancerModelTORCH()
    enhancer.load_state_dict(torch.load("enhancer_140_23.7589.pt", weights_only=True))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enhancer.to(device)
    enhancer.eval()

    return enhancer


def GetRGBFrame(folderyuv,VideoNumber,FrameNumber,fw,fh):
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    dir_list = os.listdir(folderyuv)
    v=0
    for name in dir_list:
        fullname = folderyuv + name
        if v!=VideoNumber:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
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
                if f==FrameNumber:
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)
                    return r,g,b


def GetEngancedRGBTF(enhancer,RGBin,fw,fh):
    RGBin = np.expand_dims(RGBin, axis=0)
    EnhancedPatches = enhancer.predict(RGBin)
    EnhancedPatches=np.squeeze(EnhancedPatches, axis=0)
    return EnhancedPatches


def GetEngancedRGBTORCH(enhancer,RGBin,fw,fh):
    RGBin = np.expand_dims(RGBin, axis=0).astype(np.float32)
    RGBin = torch.from_numpy(RGBin).permute(0, 3, 1, 2).to("cuda:0") / 255.0
    EnhancedPatches = enhancer(RGBin).detach().cpu().permute(0, 2, 3, 1).numpy()
    EnhancedPatches = np.clip(EnhancedPatches, a_min = 0, a_max = 1) 
    EnhancedPatches=np.squeeze(EnhancedPatches, axis=0)
    return EnhancedPatches


def ShowFramePSNRPerformance(enhancerTF, enhancerTORCH, folderyuv,foldercomp,VideoIndex,framesmax,fw,fh):
    RGBRAW = np.zeros((h, w, 3))
    RGBCOMP = np.zeros((h, w, 3))

    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = folderyuv + name
        print(name)
        if v != VideoIndex:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            frames = (2 * size) // (fw * fh * 3)
            if frames>framesmax:
                frames = framesmax

            global PSNRCOMP, PSNRENHTF, PSNRENHTORCH
            PSNRCOMP = np.zeros((frames))
            PSNRENHTF = np.zeros((frames))
            PSNRENHTORCH = np.zeros((frames))

            foldervis = "./visualizations"

            for f in range(frames):
                if f % 10 == 0:
                    print(f)

                r, g, b = GetRGBFrame(folderyuv, VideoIndex, f, w, h)
                RGBRAW[:, :, 0] = r
                RGBRAW[:, :, 1] = g
                RGBRAW[:, :, 2] = b
                r, g, b = GetRGBFrame(foldercomp, VideoIndex, f, w, h)
                RGBCOMP[:, :, 0] = r
                RGBCOMP[:, :, 1] = g
                RGBCOMP[:, :, 2] = b
                PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)

                RGBENHTF = GetEngancedRGBTF(enhancerTF, RGBCOMP, w, h)
                PSNRENHTF[f] = cal_psnr(RGBRAW / 255.0, RGBENHTF / 255.0)

                RGBENHTORCH = GetEngancedRGBTORCH(enhancerTORCH, RGBCOMP, w, h)
                PSNRENHTORCH[f] = cal_psnr(RGBRAW / 255.0, RGBENHTORCH)

                RGBRAW = cv2.cvtColor(RGBRAW.astype(np.uint8), cv2.COLOR_BGR2RGB) 
                RGBCOMP = cv2.cvtColor(RGBCOMP.astype(np.uint8), cv2.COLOR_BGR2RGB) 
                RGBENHTF = cv2.cvtColor(RGBENHTF, cv2.COLOR_BGR2RGB) 
                RGBENHTORCH = cv2.cvtColor((RGBENHTORCH*255).astype(np.uint8), cv2.COLOR_BGR2RGB)

                cv2.imwrite(f"{foldervis}/{VideoIndex}_{f}_raw.png", RGBRAW)
                cv2.imwrite(f"{foldervis}/{VideoIndex}_{f}_comp.png", RGBCOMP)
                cv2.imwrite(f"{foldervis}/{VideoIndex}_{f}_enh_old.png", RGBENHTF)
                cv2.imwrite(f"{foldervis}/{VideoIndex}_{f}_enh_new.png", RGBENHTORCH)

        break
    
    ind = np.argsort(PSNRCOMP)

    plt.plot(PSNRCOMP[ind], label='Compressed')
    plt.plot(PSNRENHTF[ind], label='Enhanced old')
    plt.plot(PSNRENHTORCH[ind], label='Enhanced new')
    plt.xlabel('Frame index')
    plt.ylabel('PSNR, dB')
    plt.grid()
    plt.legend()
    tit = "%s PSNR = [%.2f, %.2f, %.2f] dB" % (name,np.mean(PSNRCOMP), np.mean(PSNRENHTF), np.mean(PSNRENHTORCH))
    plt.title(tit)

    plt.savefig(f'models_comparison_on_{name.split(".")[0]}_{VideoIndex}.png', bbox_inches='tight')



#Frame size of training data
w=480
h=320
#patch size and petch step for training
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


enhancerTF = LoadImageEnhancementModelTF(w,h)
enhancerTORCH = LoadImageEnhancementModelTORCH(w,h)

ShowFramePSNRPerformance(enhancerTF, enhancerTORCH, testfolderRawYuv,testfolderCompYuv,0,100,w,h)

ShowFramePSNRPerformance(enhancerTF, enhancerTORCH, testfolderRawYuv,testfolderCompYuv,100,200,w,h)

ShowFramePSNRPerformance(enhancerTF, enhancerTORCH, testfolderRawYuv,testfolderCompYuv,200,300,w,h)
