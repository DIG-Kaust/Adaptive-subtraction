import time
import os
import segyio
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from examples.seismic import Model, RickerSource

from waveeqmod import Acoustic2DDevito
from utils import fixed_to_fixed_streamer, fixed_to_continous_streamer


# Dimensions
nz, nx = 650, 320 # vp2d.shape[0], vp2d.shape[1] **truncated model**
dz, dx = 10, 20 # original sampling

# Velocity
vp = segyio.open('../data/SEAM_Vp_Elastic_N23900.sgy',
                 ignore_geometry=True)
vp2d = segyio.collect(vp.trace[:]).T

vp2d = vp2d[2:nz,:nx] # create model from 20m below the next free surface

# Resample on 10 x 10 grid
x = np.arange(nx) * dx
z = np.arange(nz-2) * dz  # take away the 2 samples

xint = np.arange(x[0], x[-1], 10) 
zint = np.arange(z[0], z[-1], 10)  

nxint, nzint = len(xint), len(zint)

X, Z = np.meshgrid(xint, zint, indexing='ij')  # create empty grid
XZ = np.vstack((X.ravel(), Z.ravel())).T

# Interpolate original spaced model
interpolator = RegularGridInterpolator((x,z), vp2d.T, bounds_error=False, fill_value=0)
# Reshape it 
vp2dinterp = interpolator(XZ).reshape(nxint, nzint).T

# Water column velocity = 1490 m/s
water_vel = vp2dinterp[0,0]
vp2dinterp[vp2dinterp<=water_vel] = 0
vp2dinterp = vp2dinterp * 1.4  # scaling 
vp2dinterp[vp2dinterp==0] = water_vel

# Geometry arrange
nsrc = nxint # number of sources
nrec = nxint # number of receivers
shape = (nxint, nzint)
spacing = (10, 10)
origin = (0, 0) 

# Modelling parameters
nbl = 300 # number of boundary layers around the domain
space_order = 6 # space order of the simulation
t0 = 0 # initial time
tn = 8000 # total simulation time (ms)
dt = 4 # time sampling (ms)
f0 = 10 # source peak freq (1/ms)

# Free surface to generate data with multiples
fs = True

# Create finite-diference acoustic modelling class with Devito
awe = Acoustic2DDevito()

# Create model
awe.create_model(shape, origin, spacing, vp2dinterp.T, space_order, nbl=nbl, fs=fs)
awe.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=5,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=5,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# Build water velocity model

vp_dw = water_vel*np.ones_like(vp2dinterp)

# Force engine to have the same time sampling as the previous vel. model
vp_dw[-1, -1] = np.max(vp2dinterp)

# Create finite-diference acoustic modelling class with Devito
awe_dw = Acoustic2DDevito()

# Create model
awe_dw.create_model(shape, origin, spacing, vp_dw.T, space_order, nbl=nbl, fs=fs)
awe_dw.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=5,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=5,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# Dimensions
nz, nx = 652, 320 # vp2d.shape[0], vp2d.shape[1] truncated
dz, dx = 10, 20

# Velocity
vp = segyio.open('../data/SEAM_Vp_Elastic_N23900.sgy',
                 ignore_geometry=True)
vp2d = segyio.collect(vp.trace[:]).T

vp2d = vp2d[:nz,:nx] 

# Resample on 10 x 10 grid
x = np.arange(nx) * dx
z = np.arange(nz) * dz
xint = np.arange(x[0], x[-1], 10)
zint = np.arange(z[0], z[-1], 10)
nxint, nzint = len(xint), len(zint)
X, Z = np.meshgrid(xint, zint, indexing='ij')
XZ = np.vstack((X.ravel(), Z.ravel())).T

nsrc = nxint # number of sources
nrec = nxint # number of receivers

vp2dinterp = np.zeros(nxint*nzint, dtype=np.float32)
interpolator = RegularGridInterpolator((x,z), vp2d.T, bounds_error=False, fill_value=0)
vp2dinterp = interpolator(XZ).reshape(nxint, nzint).T

# Reescale velocity again

vp2dinterp[vp2dinterp<=1490] = 0
vp2dinterp = vp2dinterp * 1.4
vp2dinterp[vp2dinterp==0] = 1490

# Model parameter change
shape = (nxint, nzint)

# Take away fs to avoid surface related multiples
fs = False

# Compute p1 - primary reflection

awe_1 = Acoustic2DDevito()

awe_1.create_model(shape, origin, spacing, vp2dinterp.T, space_order, nbl=nbl, fs=fs)
awe_1.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=25,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=25,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)


# Compute p2 - receiver ghost

awe_2 = Acoustic2DDevito()

awe_2.create_model(shape, origin, spacing, vp2dinterp.T, space_order, nbl=nbl, fs=fs)
awe_2.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=25,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=15,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)



# Compute p3 -> source ghost

awe_3 = Acoustic2DDevito()

awe_3.create_model(shape, origin, spacing, vp2dinterp.T, space_order, nbl=nbl, fs=fs)
awe_3.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=15,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=25,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)



# Compute p4 -> source-receiver ghost

awe_4 = Acoustic2DDevito()

awe_4.create_model(shape, origin, spacing, vp2dinterp.T, space_order, nbl=nbl, fs=fs)
awe_4.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=15,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=15,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

vp_dw = vp2dinterp[0,0]*np.ones_like(vp2dinterp)

# Force engine to have the same time sampling as the previous vel. model
vp_dw[-1, -1] = np.max(vp2dinterp)

# Compute p1 / primary reflection
awe_dw1 = Acoustic2DDevito()

awe_dw1.create_model(shape, origin, spacing, vp_dw.T, space_order, nbl=nbl, fs=fs)
awe_dw1.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=25,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=25,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# Compute p2 / receiver ghost 
awe_dw2 = Acoustic2DDevito()

awe_dw2.create_model(shape, origin, spacing, vp_dw.T, space_order, nbl=nbl, fs=fs)
awe_dw2.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=25,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=15,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# # Compute p3 / source ghost
awe_dw3 = Acoustic2DDevito()

awe_dw3.create_model(shape, origin, spacing, vp_dw.T, space_order, nbl=nbl, fs=fs)
awe_dw3.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=15,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=25,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# Compute p4 / source-receiver ghost
awe_dw4 = Acoustic2DDevito()

awe_dw4.create_model(shape, origin, spacing, vp_dw.T, space_order, nbl=nbl, fs=fs)
awe_dw4.create_geometry(src_x=np.arange(0, nsrc) * spacing[0],
                    src_z=15,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=15,
                    t0=t0, tn=tn, src_type='Ricker', f0=f0)

# Total data

start_time = time.time()/60

dtot, _ = awe.solve_all_shots(dt=dt, savedtot=True) 

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# Total data direct wave

start_time = time.time()/60

dw_tot, _ = awe_dw.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

d_nodw_tot = dtot - dw_tot
np.savez('../data/data.npz', d_nodw_tot)

print('Data saved successfully in: ../data/data.npz')

# p1

start_time = time.time()/60

p1_tot, _ = awe_1.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p2

start_time = time.time()/60

p2_tot, _ = awe_2.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p3

start_time = time.time()/60

p3_tot, _ = awe_3.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p4

start_time = time.time()/60

p4_tot, _ = awe_4.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p1 DW

start_time = time.time()/60

dw1_tot, _ = awe_dw1.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p2 DW

start_time = time.time()/60

dw2_tot, _ = awe_dw2.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p3 DW

start_time = time.time()/60

dw3_tot, _ = awe_dw3.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

# p4 DW

start_time = time.time()/60

dw4_tot, _ = awe_dw4.solve_all_shots(dt=dt, savedtot=True)  

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

p_nodw_tot = (p1_tot + p4_tot - p2_tot - p3_tot) - (dw1_tot + dw4_tot - dw2_tot - dw3_tot)

np.savez('../data/DEVITO2/primaries.npz', p_nodw_tot)
print('Primaries saved successfully in: ../data/primaries.npz')

