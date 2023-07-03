import numpy as np
import pylops

data = np.load('../data/data.npz')['arr_0']

dt = 0.004
dx = 10

ns = data.shape[0]
nr = data.shape[1]
nt = data.shape[2]

p = 3  #  divide last axis size by p and pad
data_p = np.concatenate((data, np.zeros((data.shape[0], data.shape[1], data.shape[2]//p))), axis=-1)

del data

smoother = np.ones_like(data_p[0])

M = 200  # number of samples before the interface
n = np.arange(0, M)
theta = n * (np.pi/2) / M
semi_hann = np.cos(theta)  # sinusoidal taper

for i in range(smoother.shape[0]):
    smoother[i, p*smoother.shape[1]//(p+1) - M: p*smoother.shape[1]//(p+1)] = semi_hann
    smoother[i, p*smoother.shape[1]//(p+1): smoother.shape[1]] *= 0

# Apply smooth function to padded data

data_s = np.zeros_like(data_p)

for i in range(data_s.shape[0]):
    data_s[i] = np.multiply(data_p[i], smoother)  # point-wise multiplication

# Delete non-useful arrays to save memory
del data_p, smoother

h_taper = np.ones_like(data_s[0])

M = 100
n = np.arange(0, M)
theta = n * (np.pi/2) / M
semi_hann_l = np.sin(theta)  # left side
semi_hann_r = np.cos(theta)  # right side

for i in range(h_taper.shape[1]):
    h_taper[:M, i] = semi_hann_l
    h_taper[-M:, i] = semi_hann_r
    
# Apply horizontal taper to smoothed data

data_t = np.zeros_like(data_s)

for i in range(data_t.shape[0]):
    data_t[i] = np.multiply(data_s[i], h_taper)  # point-wise multiplication

# Delete non-more useful arrays to save memory
del data_s, h_taper

data_fft = np.fft.rfft(data_t, axis=-1)  # time axis

# Cast the type to complex64 to alocate only half of the array size
data_fft = data_fft.transpose(2, 0, 1).astype(np.complex64)

# Create the multi-dimensional convolution operator
MDCop = pylops.waveeqprocessing.MDC(
    data_fft,
    nt=data_t.shape[2],      # number of time samples
    nv=data_t.shape[0],  # number of sources
    dt=dt,      # dt = 0.004 s
    dr=dx,      # dx = 12.5 m
    twosided = False,
    transpose = False
)

# Delete FFT data to save memory
del data_fft

multiples =  MDCop @ data_t.transpose(2,1,0).ravel()

print('Operator successfully applied!')

multiples = multiples.reshape(data_t.shape[2], ns, nr)
multiples = multiples.transpose(1,2,0)
multiples = multiples[:,:, :nt]

np.savez('../data/srme_multiples_raw.npz', multiples)
