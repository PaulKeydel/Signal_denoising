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
        numUniquePts = int(np.ceil((N + 1) / 2))
        #x_ = np.fft.rfft(x, N)
        x_ = np.fft.fft(x, N)[0:numUniquePts]
        s_ = np.power(np.arange(1, numUniquePts + 1, dtype="int"), alpha/2)
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

#denoising a real-valued signal f with Fourier transform
def filter_fourier(f, N, thr):
    sym_part = int(np.ceil((N + 1) / 2))
    fhat = np.fft.fft(f, N)
    psd = np.real_if_close(fhat * np.conj(fhat)) / N
    indices = psd > thr
    fhat = fhat * indices
    return psd[:sym_part], np.fft.ifft(fhat).real

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

def get_nonzero_window(window_type: str, window_size: int) -> np.float32:
    window = np.ones(window_size, dtype=np.float32)
    if window_type == "hann":
        window = np.hanning(window_size + 2)[1:(window_size + 1)]
    assert(len(window) == window_size)
    return window

#compute short-term Fourier transform
def stft1D(f, fft_size, overlap_fac, window_type):
    window = get_nonzero_window(window_type, fft_size)
    segment_offset = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    total_segments = np.int32(np.ceil(len(f) / segment_offset))
    unique_pts = int(np.ceil((fft_size + 1) / 2))
    #zero-padding at the end of signal f
    f_pad = np.concatenate((f, np.zeros(fft_size)))
    #space to hold the result
    coeffs = np.empty((total_segments, unique_pts), dtype=np.complex64)
    psd = np.empty((total_segments, unique_pts), dtype=np.float32)

    #iterate over all segments and take the Fourier Transform
    for i in range(total_segments):
        start = segment_offset * i
        stop = segment_offset * i + fft_size
        segment = f_pad[start:stop]
        spectrum = np.fft.rfft(segment * window)
        coeffs[i, :] = spectrum
        psd[i, :] = (abs(spectrum) * abs(spectrum)).real
    return coeffs, psd

#inverse short-term Fourier transform
def istft1D(coeffs, fft_size, signal_length, overlap_fac, window_type):
    window = get_nonzero_window(window_type, fft_size)
    segment_offset = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    total_segments = np.int32(np.ceil(signal_length / segment_offset))

    #space to hold the result
    reco = np.zeros(signal_length, dtype=np.float64)
    window_overlap_add = np.zeros(signal_length)
    #iterate over all segments and take the Fourier Transform
    for i in range(total_segments):
        start = segment_offset * i
        stop = min(start + fft_size, signal_length)
        segment = coeffs[i, :]
        reco[start:stop] += (np.fft.irfft(segment) * window)[0:(stop-start)]
        window_overlap_add[start:stop] += (window ** 2)[0:(stop-start)]
    window_overlap_add = window_overlap_add ** -1
    return reco * window_overlap_add

def filter_stft(f, window_size, overlap_fac, window_type, thr):
    fcoeffs, psd = stft1D(f, window_size, overlap_fac, window_type)
    fcoeffs = fcoeffs * (psd > thr)
    reco = istft1D(fcoeffs, window_size, len(f), overlap_fac, window_type)
    return psd, reco

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
psd, ffilt1 = filter_fourier(f_noise, n, 2000)
to_wave_file(ffilt1, samplerate, "filt1.wav")
psd, ffilt2 = filter_fourier(f_noise, n, 7000)
to_wave_file(ffilt2, samplerate, "filt2.wav")
L = len(psd)

#plot
fig, axs = plt.subplots(3, 1, figsize=(6, 8))
plt_time_range = np.arange(samplerate - 500, samplerate + 500, 1).tolist()
freqs = np.arange(0, L) / t_max

plt.sca(axs[0])
plt.plot(t[plt_time_range], f_noise[plt_time_range], color = "c", label = "noisy")
plt.plot(t[plt_time_range], f[plt_time_range], color = "k", label = "clean")
plt.xlabel("Moment in time (s)")
plt.legend()

plt.sca(axs[1])
plt.plot(freqs, psd, color = "c", label = "power spectrum density")
plt.plot(freqs, [2000] * L, color="orange", linestyle="dashed")
plt.plot(freqs, [7000] * L, color="blue", linestyle="dashed")
plt.xlabel("Frequency (Hertz)")
plt.legend()

plt.sca(axs[2])
plt.plot(t[plt_time_range], ffilt1[plt_time_range], color = "orange", label = "filtered, thr = 2000")
plt.plot(t[plt_time_range], ffilt2[plt_time_range], color = "blue", label = "filtered, thr = 7000")
plt.xlabel("Moment in time (s)")
plt.legend()

plt.tight_layout()
plt.savefig("denoising_demo.svg", format="svg", bbox_inches="tight")
plt.show()

#do the short-term FT with a window of temporal size <t_window>
t_window = 0.02
n_samples = int(samplerate * t_window)
psd, reco = filter_stft(f_noise, n_samples, 0.5, "hann", 30000)
L = psd.shape[1]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

plt.sca(axs[0, 0])
plt.plot(t[plt_time_range], f_noise[plt_time_range], color = "c", label = "noisy")
plt.xlabel("Moment in time (s)")
plt.legend()

plt.sca(axs[0, 1])
plt.imshow(psd, origin="lower", cmap="jet", interpolation="none", aspect="auto", extent=[0, L/t_window, 0, t_max])
#plt.colorbar()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Time [s]")

plt.sca(axs[1, 0])
plt.plot(t[plt_time_range], f[plt_time_range], color = "c", label = "clean")
plt.xlabel("Moment in time (s)")
plt.legend()

plt.sca(axs[1, 1])
plt.plot(t[plt_time_range], reco[plt_time_range], color = "c", label = "filtered")
plt.xlabel("Moment in time (s)")
plt.legend()

fig.suptitle("Time resolution via a short-term Fourier Transform (window size = " + str(t_window) + "s)", fontsize=14)
plt.savefig("stft.svg", format="svg", bbox_inches="tight")
plt.show()