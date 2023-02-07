from cProfile import label
from posixpath import isabs
from turtle import color
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

###colors####
corg = '#3A2A78'
cnse = '#5CC671' #'#8FC7DB'
cflt = '#BD5B42' #'#DB88C3'
#############

#noise generating function
#alpha defines kind of noise
#alpha=0 white noise (default)
#alpha=-1 pink noise
#alpha=-2 red noise
def generateNoise(N, alpha = 0.0):
    assert(alpha <= 0.0)
    x = np.random.randn(N)
    if alpha == 0:
        return x
    else:
        numUniquePts = int(np.ceil((N+1)/2))
        #x_ = np.fft.rfft(x, N)
        x_ = np.fft.fft(x, N)[0:numUniquePts]
        s_ = np.power(np.arange(1, numUniquePts + 1, dtype='int'), alpha/2)
        x_ = x_ * s_
        #x_ = np.fft.irfft(x_, N).real
        if N % 2 == 0:
            x_ = np.concatenate([x_, np.flip(np.conj(x_))[1:numUniquePts-1]])
        else:
            x_ = np.concatenate([x_, np.flip(np.conj(x_))[0:numUniquePts-1]])
        x_ = np.fft.ifft(x_).real
        v_ = np.sqrt(np.var(x_))
        x_ = (x_ - np.mean(x_)) / v_
        return x_

######
#1D
######

def denoise1D(x, N, thr = 50.0):
    fhat = np.fft.fft(x, N)
    psd = np.real_if_close(fhat * np.conj(fhat)) / N #or abs(fhat)*abs(fhat)
    indices = psd > thr
    fhat = fhat * indices
    return np.fft.ifft(fhat).real

#create the signal
dt = 0.002
t = np.arange(0, 1, dt)
n = len(t)
f_clean = np.sin(2*np.pi*13*t) + np.sin(2*np.pi*33*t)

figure(figsize=(11, 3))
plt.plot(t, f_clean, color = corg, linewidth=1.25)
plt.xlim(0, 0.5)
plt.ylim(-5, 5)
plt.xticks([0, 0.5])
plt.yticks([-5, 0, 5])
plt.show()

#add noise
noise = [generateNoise(len(t)), generateNoise(len(t), alpha=-2.0)]
f = [f_clean + 1.5 * noise[0], f_clean + 1.5 * noise[1]]
fmax = np.ceil(max( abs(f[0]).max(), abs(f[1]).max() ))

#filter out noise in frequency domain
ffilt = [denoise1D(f[0], n), denoise1D(f[1], n)]

#plot signal and noise
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

plt.sca(axs[0])
plt.plot(t, f_clean, color = corg, linewidth=1.25, ls='--')
plt.plot(t, f[0], color = cnse, linewidth=1.25)
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

plt.sca(axs[1])
plt.plot(t, f_clean, color = corg, linewidth=1.25, ls='--', label = 'Original')
plt.plot(t, f[1], color = cnse, linewidth=1.25, label = 'verrauschtes Signal')
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()

#plot filtered signal and noise
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

plt.sca(axs[0])
plt.plot(t, f[0], color = cnse, linewidth=1.25, ls='--')
plt.plot(t, ffilt[0], color = cflt, linewidth=1.25)
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

plt.sca(axs[1])
plt.plot(t, f[1], color = cnse, linewidth=1.25, ls='--', label = 'verrauschtes Signal')
plt.plot(t, ffilt[1], color = cflt, linewidth=1.25, label = 'rekonstruiertes Signal')
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()

#plot filtered signal and original
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

plt.sca(axs[0])
plt.plot(t, ffilt[0], color = cflt, linewidth=1.25, ls='--')
plt.plot(t, f_clean, color = corg, linewidth=1.25)
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

plt.sca(axs[1])
plt.plot(t, ffilt[1], color = cflt, linewidth=1.25, ls='--', label = 'rekonstruiertes Signal')
plt.plot(t, f_clean, color = corg, linewidth=1.25, label = 'Original')
plt.xlim(0, 0.5)
plt.ylim(-fmax, fmax)
plt.xticks([0, 0.5])
plt.yticks([-fmax, 0, fmax])

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()

######
#2D
######

#read image
img = mpimg.imread('Ring-artifact.jpg')
assert (img.max() == 255) & (img.min() == 0)
print(img.shape)
ny = img.shape[0]
nx = img.shape[1]

#create noise
norm_ratio_noise = 0.2 #ratio of normF(noise)/normF(clean image)
imgnoise = np.random.randn(*img.shape) #we need an unpacked (*) list
imgnoise = imgnoise / np.linalg.norm(imgnoise, 'fro') * np.linalg.norm(img, 'fro') * norm_ratio_noise
print(np.linalg.norm(img, 'fro'))
print(np.linalg.norm(imgnoise, 'fro'))
imgnoise = img + imgnoise
imin = min(0, imgnoise.min())
imax = max(255, imgnoise.max())

ihat = np.fft.fft2(imgnoise, imgnoise.shape)

filterMatrix = np.ones(ihat.shape)
keep_fraction = 0.1
filterMatrix[int(ny*keep_fraction):int(ny*(1-keep_fraction)), :] = 0
filterMatrix[:, int(nx*keep_fraction):int(nx*(1-keep_fraction))] = 0
#we could also use a circular/elliptic filter instead of a rectangular one

ipsd = np.fft.fftshift(np.real_if_close(ihat * np.conj(ihat)) / nx / ny)
ihat = ihat * filterMatrix
irec = np.fft.ifft2(ihat).real

#plot
figure(figsize=(5.4, 6.5), dpi=100)
plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
plt.axis('off')
plt.show()

figure(figsize=(5.4, 6.5), dpi=100)
plt.imshow(imgnoise, cmap=plt.get_cmap('gray'), vmin=imin, vmax=imax)
plt.axis('off')
plt.show()

figure(figsize=(5.4, 6.5), dpi=100)
plt.imshow(irec, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
plt.axis('off')
plt.show()