import numpy as np
import matplotlib.pyplot as plt
import wave

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

#denoising with Fourier transform
def filter_fourier(f, N, thr):
    fhat = np.fft.fft(f, N)
    psd = np.real_if_close(fhat * np.conj(fhat)) / N
    indices = psd > thr
    fhat = fhat * indices
    return psd, np.fft.ifft(fhat).real

#function to save the signal as audio file
def to_wave_file(signal_vector, sample_rate, filename):
    left_channel =  signal_vector
    right_channel = signal_vector

    #put the channels together and convert to little-endian 16 bit integers
    audio = np.array([left_channel, right_channel]).T
    audio = (audio * (2 ** 15 - 1)).astype("<h")

    with wave.open(filename, "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)  # 2 bytes per sample
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())

#compute short-term Fourier transform
def stft1D(f, fft_size, overlap_fac, window):
    assert(len(window) == fft_size)
    segment_offset = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    total_segments = np.int32(np.ceil(len(f) / segment_offset))
    unique_pts = int(np.ceil((fft_size + 1) / 2))
    #zero-padding at the end of signal f
    f_pad = np.concatenate((f, np.zeros(fft_size)))
    #space to hold the result
    result = np.empty((total_segments, unique_pts), dtype=np.float32)

    #iterate over all segments and take the Fourier Transform
    for i in range(total_segments):
        current_pos = segment_offset * i
        segment = f_pad[current_pos:(current_pos + fft_size)]
        spectrum = np.fft.rfft(segment * window)
        result[i, :] = (abs(spectrum) * abs(spectrum)).real
    return result, unique_pts

#set sampling frequency fs and temporal length of the signal in seconds
samplerate = 44100
t_max = 2

#generate the signal from sine waves and create data vector
t = np.linspace(0, t_max, t_max * samplerate)
n = len(t)
f = np.sin(2 * np.pi * 533 * t) + np.concatenate(([0] * samplerate, np.sin(2 * np.pi * 2117 * t[samplerate:])))

#add noise and save both signals as wav-file
f_noise = f + 0.5 * generateNoise(len(t), alpha=-1.0)
to_wave_file(f, samplerate, "orig.wav")
to_wave_file(f_noise, samplerate, "orig_noisy.wav")

#filter out noise in frequency domain and save it to wav-file
L = int(np.ceil((n + 1) / 2))
psd, ffilt1 = filter_fourier(f_noise, n, 2000)
to_wave_file(ffilt1, samplerate, "filt1.wav")
psd, ffilt2 = filter_fourier(f_noise, n, 7000)
to_wave_file(ffilt2, samplerate, "filt2.wav")

#plot
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
plt_time_range = np.arange(samplerate - 500, samplerate + 500, 1).tolist()

plt.sca(axs[0])
plt.plot(t[plt_time_range], f_noise[plt_time_range], color = 'c', label = 'noisy')
plt.plot(t[plt_time_range], f[plt_time_range], color = 'k', label = 'clean')
plt.legend()

plt.sca(axs[1])
plt.plot(psd[:L], color = 'c', label = 'power spectrum density')
plt.plot([2000] * L, color='orange', linestyle='dashed')
plt.plot([7000] * L, color='blue', linestyle='dashed')
plt.xticks(np.arange(0, L, 4000), ((samplerate/n) * np.arange(0, L, 4000)).astype(int))
plt.legend()

plt.sca(axs[2])
plt.plot(t[plt_time_range], ffilt1[plt_time_range], color = 'orange', label = 'filtered, thr = 2000')
plt.plot(t[plt_time_range], ffilt2[plt_time_range], color = 'blue', label = 'filtered, thr = 7000')
plt.legend()

plt.savefig("denoising_demo.svg", format="svg", bbox_inches="tight")
plt.show()

stft, L = stft1D(f_noise, 1000, 0, np.hanning(1000))
#plotting everything
img = plt.imshow(stft, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
plt.xticks(np.arange(0, L, 20),
           np.around((samplerate / 1000) * np.arange(0, L, 20)).astype(int),
           rotation='vertical')
plt.savefig("stft.svg", format="svg", bbox_inches="tight")
plt.show()