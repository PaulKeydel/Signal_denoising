from cProfile import label
from posixpath import isabs
from turtle import color
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

#noise generating function
#alpha defines kind of noise
#alpha=0 white noise
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

#create the signal
dt = 0.002
t = np.arange(0, 1, dt)
n = len(t)
f = np.sin(2*np.pi*13*t) + np.sin(2*np.pi*33*t)
f_clean = f

#add noise
noise = generateNoise(len(t), alpha=-1.0)
f = f + 1.5 * noise
fmax = np.ceil(abs(f).max())

fhat = np.fft.fft(f, n)
psd = np.real_if_close(fhat * np.conj(fhat)) / n #or abs(fhat)*abs(fhat)
freq = [1/(dt * n)] * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

#filter out noise in frequency domain
thr = 50
indices = psd > thr
psdClean = psd * indices
fhat = fhat * indices
ffilt = np.fft.ifft(fhat).real

#plot
fig, axs = plt.subplots(3, 1)

plt.sca(axs[0])
plt.plot(t, f, color = 'c', label = 'noisy')
plt.plot(t, f_clean, color = 'k', label = 'clean')
plt.xlim(t[0], t[-1])
plt.ylim(-fmax, fmax)
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], psd[L], color = 'c', label = 'power spectrum density')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.sca(axs[2])
plt.plot(t, ffilt, color = 'k', label = 'filtered')
plt.xlim(t[0], t[-1])
plt.ylim(-fmax, fmax)
plt.legend()

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
imgnoise = np.random.randn(*img.shape) #using unpacking operator
imgnoise = imgnoise / np.linalg.norm(imgnoise, 'fro') * np.linalg.norm(img, 'fro') * norm_ratio_noise
print(np.linalg.norm(img, 'fro'))
print(np.linalg.norm(imgnoise, 'fro'))
imgnoise = img + imgnoise
imin = min(0, imgnoise.min())
imax = max(255, imgnoise.max())

ihat = np.fft.fft2(imgnoise, imgnoise.shape)

filterMatrix = np.ones(ihat.shape)
keep_fraction = 0.1
filterMatrix[int(ny*keep_fraction):int(ny*(1-keep_fraction))] = 0
filterMatrix[:, int(nx*keep_fraction):int(nx*(1-keep_fraction))] = 0
#we could also use a circular/elliptic filter instead of a rectangular one

ipsd = np.fft.fftshift(np.real_if_close(ihat * np.conj(ihat)) / nx / ny)
ihat = ihat * filterMatrix
irec = np.fft.ifft2(ihat).real

#plot
fig, axs = plt.subplots(2, 2)

plt.sca(axs[0][0])
plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

plt.sca(axs[0][1])
plt.imshow(imgnoise, cmap=plt.get_cmap('gray'), vmin=imin, vmax=imax)

plt.sca(axs[1][0])
plt.imshow(irec, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

plt.sca(axs[1][1])
plt.imshow(ipsd, norm = LogNorm(vmin=5))
plt.colorbar()
rect = patches.Rectangle((int(nx*(0.5-keep_fraction)), int(ny*(0.5-keep_fraction))), int(2*nx*keep_fraction), int(2*ny*keep_fraction), linewidth=1, edgecolor='r', facecolor='none')
axs[1][1].add_patch(rect)

plt.show()