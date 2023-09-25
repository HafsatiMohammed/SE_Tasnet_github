# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:32:55 2022

@author: hafsa
"""
from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses import PITLossWrapper
import torch
from pit_criterion import cal_loss
#from Model_TasNet import *
import torchaudio
#from TorchAudioConvTasnet import *

import soundfile as sf
#import stft
import os 
import fnmatch 
import random 
import numpy as np
import glob 
from tqdm import tqdm
import pickle
#import pandas as pd
from numpy.random import RandomState
import matplotlib.pyplot as plt
import scipy.io
from random import randint
import scipy
import librosa
from multiprocessing import Process, Value, Array, Lock
import heapq
from Losses import *
"""
from pystoi import stoi
from pesq import pesq
from itertools import permutations
"""






# Model params 

N = 256
L = 20 
B = 256
H = 512
P = 3
X = 8
R = 4
C = 1
norm_type = 'gLN'
causal = 0
mask_nonlinear = 'relu'







def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)




# Data handeling 


def olafilt(b, x, zi=None):
    """
    Filter a one-dimensional array with an FIR filter
    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.
    Parameters
    ----------
    b : one-dimensional numpy array
        The impulse response of the filter
    x : one-dimensional numpy array
        Signal to be filtered
    zi : one-dimensional numpy array, optional
        Initial condition of the filter, but in reality just the
        runout of the previous computation.  If `zi` is None or not
        given, then zero initial state is assumed.
    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """

    L_I = b.shape[0]
    # Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = 2<<(L_I-1).bit_length()
    L_S = L_F - L_I + 1
    L_sig = x.shape[0]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(L_sig+L_F, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(L_sig+L_F)

    FDir = fft_func(b, n=L_F)

    # overlap and add
    for n in offsets:
        res[n:n+L_F] += ifft_func(fft_func(x[n:n+L_S], n=L_F)*FDir)

    if zi is not None:
        res[:zi.shape[0]] = res[:zi.shape[0]] + zi
        return res[:L_sig], res[L_sig:]
    else:
        return res[:L_sig]
    

def Add_noise(s,n,SNR):
    #SNR = 10**(SNR/20)
    Es = np.sqrt(np.sum(s[:]**2)+1e-6)
    En = np.sqrt(np.sum(n[:]**2)+1e-6)
    iSNR = 10*np.log10(Es**2/(En**2+1e-6)) 
    alpha = 10**((iSNR-SNR)/20)
    
    #alpha = Es/(SNR*(En+1e-8))
    #•Mix = s+alpha*n[0:160000]
    
    return  alpha


def Load_SRIR():  
    SRIR_Dir = r'RI_Train_NewMic'
    SRIR = np.zeros((6,5,18,32682,3))
    #print(os.listdir(SRIR_Dir))
    for eachfile in tqdm(os.listdir(SRIR_Dir)):   
        Name = eachfile.split('_')
        file = os.path.join(SRIR_Dir, eachfile)
        matlabfile = scipy.io.loadmat(file)
        ImpResp_Rev = matlabfile['ImpResp_Rev'];
        ImpResp_Rev = ImpResp_Rev[:,0]
        secs = len(ImpResp_Rev[0])/48000 # Number of seconds in signal X
        samps = secs*16000    # Number of samples to downsample
        ImpResp_Rev_ = np.zeros((32682, 3))
        ImpResp_Rev_[:,0] = np.squeeze(scipy.signal.resample(ImpResp_Rev[0],int(samps))[:], 1)
        ImpResp_Rev_[:,1] = np.squeeze(scipy.signal.resample(ImpResp_Rev[1],int(samps))[:],1)
        ImpResp_Rev_[:,2] = np.squeeze(scipy.signal.resample(ImpResp_Rev[2],int(samps))[:],1)
        SRIR[int(Name[1])-1, int(Name[2])-1 , int(Name[3].split('.')[0])-1,:,: ] = ImpResp_Rev_# scipy.signal.resample(ImpResp_Rev_[:,0], int(samps))
    return SRIR


def Preprocessing(S, N, SRIR, DoA, nuul):
    wlen = 512
    sequence_length = 16000 
    batch_numOfSequence = 10
    
    Input = np.zeros((batch_numOfSequence, 16000,1)) 
    Output = np.zeros((batch_numOfSequence,1,16000))
    Mix_complex = np.zeros((batch_numOfSequence,1,20,257), 'complex')
    fs = 16000
    SRIR_Anec= np.zeros((32682,3))
    Contrib_Anec = np.zeros((len(S),160000,1))
    Contrib_Rev = np.zeros((len(S)+1,160000,1))
    
    for iR in range(len(S)):
        SRIR_iR = SRIR[iR]#/np.max(np.abs(SRIR[iR][:,0]))
        for il in range(1):
            I = np.argmax(SRIR_iR[:,il])
            SRIR_Anec[I:I+8,il] =  SRIR_iR[I:I+8,il]
            SRIR_Anec[I:I+8,il] =  SRIR_iR[I:I+8,il]

            #print(20*np.log10(np.sqrt(np.sum(S[0]**2))/np.sqrt(np.sum(S[iR]**2))))
        for il in range(1):
            Contrib_Anec[iR,:,il] = olafilt(SRIR_Anec[:,il], S[iR])[:160000]
            Contrib_Rev[iR,:,il] = olafilt(SRIR[iR][:,il], S[iR])[:160000]
            if iR>0:
                if iR==1:
                    DesiredSNR = np.random.uniform(low=-5, high=15, size=(1,))
                    alpha = Add_noise(Contrib_Rev[0,:,0] , Contrib_Rev[il,:,0] ,DesiredSNR)
                    Contrib_Rev[iR,:,il] = alpha * Contrib_Rev[iR,:,il]
                else:
                    DesiredSNR = np.random.uniform(low=-5, high=15, size=(1,))
                    alpha = Add_noise(Contrib_Rev[0,:,0] , Contrib_Rev[il,:,0] ,DesiredSNR)
                    Contrib_Rev[iR,:,il] = alpha * Contrib_Rev[iR,:,il]
            
            
            
    SRIR_Noise = np.mean(np.array([SRIR[random.randrange(0, len(S), 1)][32682-30000:,:], SRIR[random.randrange(0, len(S), 1)][32682-30000:,:]]), axis=0) 
    SRIR_Noise = SRIR_Noise
    Desired_SIR = np.random.uniform(low=-5, high=0, size=(1,))
    #alpha = Add_noise(S[0],N,Desired_SIR)
    #♣N = alpha*N       
    for il in range(1):
        Contrib_Rev[-1,:,il] = olafilt(SRIR[iR][:,il], N)[:160000]
    if not nuul:    
        DesiredSNR = np.random.uniform(low=-6, high=15, size=(1,))
        alpha = Add_noise(Contrib_Rev[0,:,0] , Contrib_Rev[-1,:,0] ,DesiredSNR)
        Contrib_Rev[-1,:,0] = alpha * Contrib_Rev[-1,:,0]    
    
    
    Mix = np.sum(Contrib_Rev,0)
    Mix = Mix/np.max(np.abs(Mix[:,0])) 
    Contrib_Anec[0,:,0] = Contrib_Anec[0,:,0]#/np.max(np.abs(Contrib_Anec[0,:,0]))
    
    
    ibatch =0
    t=0
    Contrib_Anec = Contrib_Anec/np.max(np.abs(Contrib_Anec)+1e-6) 

    for iSequence in range(batch_numOfSequence):
        #print(Mix.shape)
        Input[ibatch,:,0] = Mix[t:t+sequence_length,0]
        Output[ibatch,0,:] = Contrib_Anec[0,t:t+sequence_length,0]

        t=t+sequence_length
               
        ibatch=ibatch+1 
        

    return Input, Output    




def DataGeneration(Dir_Speech, Dir_Noise, List_RT60, SRIR):
    List_Speech = os.listdir(Dir_Speech)
    List_Noise  = os.listdir(Dir_Noise)
    Input = np.zeros((10, 16000,1)) 
    Output = np.zeros((10, 1 , 16000))
    
    batch = 0
    for RT60 in range(len(List_RT60)):
            NumberOfSources = random.choice([2,3])
            Chosen_Speech = random.choices(List_Speech,k=NumberOfSources)
            Chosen_Noise = random.choices(List_Noise,k=10)
            S = []
            alpha = random.randint(1, 1)
            for il in range(len(Chosen_Speech)):
                if il==0:
                    y,fs = librosa.load(os.path.join(Dir_Speech, Chosen_Speech[il]),sr=16000)
                    y = alpha*y/(np.max(np.abs(y))+1e-6)
                    if len(y)<160000:
                        y = np.append(np.zeros(160000-len(y)), y)
                    else:
                        y = y[0:160000]
                    
                    nuul = random.randint(0, 10) <0
                    if nuul:
                        y = np.zeros(160000)
                        #k = random.randint(1,160000-160000)
                        #y = y[k:k+160000]
                    for il in range(random.randint(0,8)):
                        A = random.randint(2*16000, 8*16000)
                        B = A + random.randint(500, 2*16000)
                        y[A:B] = np.zeros((B-A))
                elif il==1:
                    if not nuul:
                        y,fs = librosa.load(os.path.join(Dir_Noise, Chosen_Noise[il]),sr=16000)
                        y = alpha*y/(np.max(np.abs(y))+1e-6)
                        if len(y)<160000:
                            y = np.append(np.zeros(160000-len(y)), y)
                        else:
                            y = y[0:160000]
                    else: 
                        y = np.zeros(160000)

                    for il in range(random.randint(0,4)):
                        A = random.randint(2*16000, 8*16000)
                        B = A + random.randint(500, 2*16000)
                        y[A:B] = np.zeros((B-A))
                
                else:
                    y,fs = librosa.load(os.path.join(Dir_Noise, Chosen_Noise[il]),sr=16000)
                    y = alpha*y/(np.max(np.abs(y))+1e-6)
                    if len(y)<160000:
                        y = np.append(np.zeros(160000-len(y)), y)
                    else:
                        y = y[0:160000]
                        #►k = random.randint(1,160000-16000)
                        #y = y[k:k+16000]
                    for il in range(random.randint(0,5)):
                        A = random.randint(2*16000, 8*16000)
                        B = A + random.randint(500, 2*16000)
                        y[A:B] = np.zeros((B-A))

                        
                S.append(y)
            y,fs = librosa.load(os.path.join(Dir_Noise, Chosen_Noise[6]),sr=16000)
            y = y/(np.max(np.abs(y))+1e-6)
            if len(y)<160000:
                y = np.append(np.zeros(160000-len(y)), y)
            else:
                y = y[0:160000]
                #k = random.randint(1,160000-16000)
                #y = y[k:k+16000]
            y = alpha*y/(np.max(np.abs(y))+1e-8)
            N = y
            DoA = []
            srir = [] 
            Dir =  random.choices(range(18),k=NumberOfSources)
            DoA.append(Dir)
            for il in range(NumberOfSources):
                InCam = random.randint(0, 4)
                srir.append(SRIR[List_RT60[RT60],InCam, Dir[il],:,:]) 
            Input[:,:,:], Output[:,:,:] = Preprocessing(S, N, srir, DoA, nuul)
            #batch = batch+4
    return Input, Output
 


    
    
if __name__ == '__main__':

    torch.cuda.set_device(1)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    Dir_Speech = r'../Speech_Sepration_Car/BMW/LibriSpeech/train-clean-360/clean_speech'     
    Dir_Noise =  r'Noise_Background'
    SRIR =  Load_SRIR()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
# define model
    """
    model = ConvTasNet(N, L, B, H, P, X, R,
                       C, norm_type=norm_type, causal=causal,
                       mask_nonlinear=mask_nonlinear)
    """
    model = torchaudio.models.ConvTasNet(num_sources=1, enc_kernel_size = 256, enc_num_feats = 256, msk_kernel_size = 3, msk_num_feats= 128, msk_num_hidden_feats = 256, msk_num_layers = 8, msk_num_stacks = 2)
    
    
    model.cuda()
    checkpoint = torch.load('BestModel_SE_SDR.pth.tar')
    model.load_state_dict(checkpoint)
    
    
    
    
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    """
    
    
    #model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    #model = torch.nn.DataParallel(model)
    #model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.0001,
                                    weight_decay=0.0)
    #model = load_model("best_model_custom_IRM_mobile.h5")
    List_RT60 = []
    NumberOfIteration = 700

    save_dir = "ValidpklFiles_TASNet/"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = 10000
    n_epochs = 5000
    loss_train = np.zeros(shape = (n_epochs,), dtype=np.float32)
    loss_valid = np.zeros(shape = (n_epochs,), dtype=np.float32)
    loss_val = np.zeros(shape = (n_epochs,))
    Batch_size = 10
    training = True
    if training== True: 
        for epoch in range(5000):
             
            List_RT60 = []
            for il in range(5):
                List_RT60.extend([il]*NumberOfIteration)
            random.shuffle(List_RT60)
            step_batch = int(1)
            cpt = 0
            loss_glob = 0
            model.train() 
            with tqdm(range(0,len(List_RT60),step_batch), unit="batch") as pbar:
                for ibatch in pbar:
                    
            #for ibatch in tqdm(range(0,len(List_RT60),step_batch)):
                    Input, Output  = DataGeneration(Dir_Speech, Dir_Noise, List_RT60[ibatch:ibatch+step_batch], SRIR)
                    Output = torch.from_numpy(Output).float().to(device)
                    #Input = np.transpose(Input,[0,2,3,1])#X_train.shape[0],X_train.shape[2], X_train.shape[1], X_train.shape[3]))
                    #Output = np.transpose(Output,[0,2,3,1])
                    #Mix_complex = np.transpose(Mix_complex,[0,2,3,1]) +10**-6                   
                    if epoch ==80000:
                        with open(save_dir+str(cpt)+'.pkl','wb') as f: 
                            pickle.dump([Input,Output],f)
                        cpt = cpt+1
                    else:
                        estimate_source = model( torch.unsqueeze(torch.from_numpy(Input[:,:,0]), 1).float().to(device))
                        mixture_lengths = torch.from_numpy(np.array([16000]*Batch_size))
                        loss_SNR, max_snr, estimate_source, reorder_estimate_source = \
                            cal_loss(Output, estimate_source, mixture_lengths.to(device))
                        loss_SISDR =  pit_neg_si_sdr(estimate_source, Output, zero_mean=False, epsilon=1e-10)                           
                        loss = (0.7*loss_SISDR + 0.3*loss_SNR)
                       
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   5)
                        optimizer.step()
                        
                        loss_glob =loss_glob+loss 
                        pbar.set_postfix(loss = loss_glob/cpt)
                        cpt = cpt+1
            loss_train[epoch] = loss_glob/cpt 
            model.eval()
            DirVal = os.listdir(save_dir)
            cpt = 0
            loss_glob = 0
            with torch.no_grad():
                if epoch ==80000:
                    pass
                else:
                    with tqdm(range(0,len(DirVal)), unit="batch") as pbar:
                        for il in pbar:
                            file = DirVal[il]
                            
                            Input, Output =  load_pickle(os.path.join(save_dir, file))
                            #Output = torch.from_numpy(Output).float().to(device)
                            estimate_source = model( torch.unsqueeze(torch.from_numpy(Input[:,:,0]), 1).float().to(device))
                            mixture_lengths = torch.from_numpy(np.array([16000]*Batch_size))
                            Output = Output.to(device)                            
                            loss_SNR, max_snr, estimate_source, reorder_estimate_source = \
                                cal_loss(Output, estimate_source, mixture_lengths.to(device))
                            loss_SISDR =  pit_neg_si_sdr(estimate_source, Output, zero_mean=False, epsilon=1e-10)
                            loss = (0.3*loss_SISDR + 0.7*loss_SNR)

                            
                            loss_glob =loss_glob+loss 
                            cpt = cpt+1
                            pbar.set_postfix( loss_valid = loss_glob/cpt)
                    loss_valid[epoch] = loss_glob/cpt
     
                    
                    if loss_valid[epoch] < best_loss:
                         #model.save("best_model_custom_IRM_mobile_SDR.h5")
                        #Wmodel.compute_output_shape(input_shape=(1,16000,1))
                        file_path = os.path.join('./BestModel_SE_SDR.pth.tar' )
                        torch.save(model.state_dict(), file_path)
                        print(f"\n[INFO]\tNew best loss (from {best_loss:0.2f} to {loss_valid[epoch]:0.2f}).. Model saved.")
                        best_loss = loss_valid[epoch]
                print(f"\nEpoch {epoch+1}/{n_epochs}:\n\tTrain:\t\tloss:\t\t{loss_train[epoch]:0.3f}.\n")
    



