import numpy as np
import pylops

data = np.load('../data/data_cube.npz')['arr_0']

print('Loaded data')

p = 3  # the bigger, the smaller the padding
data_p = np.concatenate((data, np.zeros((data.shape[0], data.shape[1], data.shape[2]//p))), axis=-1)
data_p.shape

ns = data_p.shape[0]
nr = data_p.shape[1]
nt = data_p.shape[2]

dt = 0.004
dx = 12.5

v_taper = np.ones_like(data_p)

M = 200
n = np.arange(0, M)
theta = n * (np.pi/2) / M
semi_hann = np.cos(theta)

for i in range(v_taper.shape[0]):
    for j in range(v_taper.shape[1]):
        v_taper[i, j, p*v_taper.shape[2]//(p+1) - M: p*v_taper.shape[2]//(p+1)] = semi_hann
        v_taper[i, j, p*v_taper.shape[2]//(p+1): v_taper.shape[2]] *= 0
        
data_vt = np.multiply(data_p, v_taper)  

h_taper = np.ones_like(data_p)

M = 100
n = np.arange(0, M)
theta = n * (np.pi/2) / M
semi_hann_l = np.sin(theta)  # left side
semi_hann_r = np.cos(theta)  # right side

for i in range(h_taper.shape[0]):
    for j in range(h_taper.shape[2]):
        h_taper[i, 0: M, j] = semi_hann_l
        h_taper[i, h_taper.shape[1] - M: h_taper.shape[1], j] = semi_hann_r
        
data_t = np.multiply(data_vt, h_taper).astype(np.float32)# data tapered

print('Tapered data')

# Delete previous data to save memory
del data_vt, h_taper, v_taper

data_fft = np.fft.rfft(data_t, axis=-1)

data_fft = data_fft.transpose(2, 0, 1).astype(np.complex64)

print('FFTed data')

# Create the multi-dimensional convolution operator
MDCop = pylops.waveeqprocessing.MDC(
    data_fft,
    nt=nt,      # number of time samples
    nv=ns,  # number of sources
    dt=dt,      # dt = 0.004 s
    dr=dx,      # dx = 12.5 m
    twosided = False,
    transpose = False
)
print('MDCop.shape:', MDCop.shape)

print('data_t.size:', data_t.size)

multiples =  MDCop @ data_t.transpose(2,1,0).ravel()

multiples = multiples.reshape(nt, ns, nr)
multiples = multiples.transpose(1,2,0)
multiples = multiples[:,:, :data.shape[2]]
print('multiples.shape:', multiples.shape)

np.savez('../data/srme_multiples.npz', multiples)