#coding=utf-8
import os
import librosa
import math
import numpy as np
import argparse
import cv2
import torch
from torch_stft import STFT
import PIL.Image as Image
import pickle as p
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy import misc

# get number of frames
def GetFrameNumber(frame_path):
    command3 = 'ls -l ' + frame_path + ' | grep "^-" | wc -l'
    FRAME_NUMBER = int(os.popen(command3).read())
    return FRAME_NUMBER

# transfer image data into matrix
def CreateVisualMatrix(frame_path, FRAME_NUMBER):
    # get size of the image
    image_file_path = os.path.join(frame_path, "000001.jpg")
    if not os.path.exists(os.path.dirname(image_file_path)):
        print("error: some frames are lost. Visual Matrix initializing failed.")
        return 
    command1 = 'identify -format "%w" ' + image_file_path 
    command2 = 'identify -format "%h" ' + image_file_path
    width = int(os.popen(command1).read()) 
    height = int(os.popen(command2).read())
    size = width*height
    
    # initialize an empty row in Visual Matrix
    resultV = np.array([])
    for frame_name in os.listdir(frame_path):
        # load image
        frame_file_path = os.path.join(frame_path, frame_name)
        image = Image.open(frame_file_path)
        # separate r,g,b channel
        r, g, b = image.split()
        r_arr = np.array(r).reshape(size)
        g_arr = np.array(g).reshape(size)
        b_arr = np.array(b).reshape(size)
        # turn the image into an array
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        resultV = np.concatenate((resultV, image_arr))
    # turn the array into two dimensions matrix [IMAGE_INFO, FRAME_NUMBER]
    resultV = resultV.reshape((FRAME_NUMBER, size*3)).T
    print("image -> matrix  done")
    #print(resultV[0:5,1:3])
    #file_path = "V.txt"
    #with open(file_path, mode='wb') as f:
      #  p.dump(resultV, f)
      #  print("保存文件成功")
    return resultV
        
"""
def CreateAudioMatrix(signal, FRAME_NUMBER, sample_rate=11025, frame_size=0.025, frame_stride=0.01, winfunc=np.hamming, NFFT=512):
    # STFT to original signal
    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    #print("signal length is", signal_length)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    
    plt.imshow(pow_frames)
    plt.axis('auto')
    plt.rcParams['axes.facecolor'] = 'pink'
    plt.tight_layout()
    if os.path.exists(os.path.dirname("A.jpg")):
        os.removedirs(os.path.dirname("A.jpg"))
    plt.savefig("A.jpg")
    plt.show()
    
    # Turn stft spectrogram into Matrix A
    resultA = np.pad(pow_frames, FRAME_NUMBER*ceil(len(pow_frames)/FRAME_NUMBER),'constant')
    resultA = resultA.reshape((FRAME_NUMBER, -1)).T
    print("audio -> matrix  done")
    print(pow_frames, "\n\n", resultA)


def CreateAudioMatrix(sig, FRAME_NUMBER, sample_rate=11025):
    # STFT to original signal
    audio = librosa.load(sig, sr=sample_rate)[0]
    #audio, sample_rate = sf.read(sig)
    device = 'cpu'
    filter_length = 1024
    hop_length = 256
    win_length = 1024 # doesn't need to be specified. if not specified, it's the same as filter_length
    window = 'hann'
    librosa_stft = librosa.stft(audio, n_fft=filter_length, hop_length=hop_length, window=window)
    _magnitude = np.abs(librosa_stft)
    print(FRAME_NUMBER)
    print(len(audio))
    print(_magnitude.shape)
    #print(20*np.log10(1+_magnitude)[0])
    
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)

    stft = STFT(
        filter_length=filter_length, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window
    ).to(device)

    magnitude, phase = stft.transform(audio)
    
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.title('PyTorch STFT magnitude')
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')
    plt.imshow(20*np.log10(1+magnitude[0].cpu().data.numpy()), aspect='auto', origin='lower')
    
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.title('STFT magnitude')
    plt.xlabel('Frames')
    plt.ylabel('magnitude')
    plt.imshow(_magnitude, aspect='auto', origin='lower')

    plt.subplot(212)
    plt.title('Librosa STFT magnitude')
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')
    plt.imshow(20*np.log10(1+_magnitude), aspect='auto', origin='lower')
    plt.tight_layout()
    if os.path.exists(os.path.dirname("A.jpg")):
        os.removedirs(os.path.dirname("A.jpg"))
    plt.savefig('A.jpg')
    plt.clf()
    
    output = stft.inverse(magnitude, phase)
    output = output.cpu().data.numpy()[..., :]
    audio = audio.cpu().data.numpy()[..., :]
    print(np.mean((output - audio) ** 2)) # on order of 1e-17
    
"""
def CreateAudioMatrix(sig, FRAME_NUMBER, sample_rate=11025):
    audio = librosa.load(sig, sr=sample_rate)[0]
    f, t, zxx = signal.stft(audio, fs=11025)
    """
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if os.path.exists(os.path.dirname("B.jpg")):
        os.removedirs(os.path.dirname("B.jpg"))
    plt.savefig('B.jpg')
    print("the f array is\n", f.shape,"\n\n\n the t array is\n", t.shape, "\n\n\n the zxx array is\n", zxx.shape, "\n\n\n")
    """
    resultA = zxx.T.reshape(1,-1)
    #print(resultA.shape)
    para = FRAME_NUMBER-(len(t)*len(f))%FRAME_NUMBER
    #print(para)
    resultA = np.pad(resultA, ((0, 0), (0, int(para))), 'constant', constant_values=0)
    #print(resultA.shape)
    resultA = resultA.reshape(FRAME_NUMBER, -1)
    resultA = resultA.T
    
    #print(resultA.shape)
    print("audio -> matrix  done")
    return resultA

def ProposedADMM(Matrix_V, Matrix_A, FRAME_NUMBER):
    # construct a Laplacian Matrix L(shape(FRAME_NUMBER, FRAME_NUMBER))
    w = np.zeros(FRAME_NUMBER*FRAME_NUMBER).reshape(FRAME_NUMBER, FRAME_NUMBER)
    d = np.zeros(FRAME_NUMBER)
    L = np.zeros(FRAME_NUMBER*FRAME_NUMBER).reshape(FRAME_NUMBER, FRAME_NUMBER)
    for i in range(0,FRAME_NUMBER):
        for j in range(0, FRAME_NUMBER):
            if(abs(i-j)==1):
                #print(i,j,i-j)
                w[i,j]=1
                d[i] +=1
    for i in range(0,FRAME_NUMBER):
        for j in range(0, FRAME_NUMBER):
            if(i==j):
                L[i,j]=d[i]-w[i,j]
            else:
                L[i,j]=0-w[i,j]
    print(w)
    print(d)
    print(L)
    




if __name__ == '__main__':
    #CreateVisualMatrix("test")
    # Read wav file
    # sample_rate, audio = wavfile.read("1.wav")
    #print(len(audio))
    #CreateAudioMatrix(audio, sample_rate)
    # Get speech data in the first 2 seconds
    #signal = signal[0:int(2. * sample_rate)]

    # Calculate the short time fourier transform

    #pow_spec = CreateAudioMatrix(audio, sample_rate)
    audio = "1.wav"
    frame_number = GetFrameNumber('test')
    CreateAudioMatrix(audio, frame_number)
# ADMM method 


# output