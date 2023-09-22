import os
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Tuple
from typing import Union
from tqdm import tqdm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from examples.seismic import Model, RickerSource
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

mpl.rc('font', size=16)
mpl.rc('figure', figsize=(8, 6))


class Acoustic2DDevito():
    def __init__(self):
        pass

    def create_model(self, shape: Tuple[int], origin: Tuple[float], spacing: Tuple[float],
                     vp: npt.DTypeLike, space_order: int = 6, nbl: int = 20, fs: bool = False):
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries
        fs : :obj:`bool`, optional
            Add free surface
        """
        self.space_order = space_order
        self.fs = fs
        self.model = Model(space_order=space_order, vp=vp * 1e-3, origin=origin, shape=shape,
                           dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp", fs=fs)

    def create_geometry(self, src_x: npt.DTypeLike, src_z:  Union[float, npt.DTypeLike],
                        rec_x: npt.DTypeLike, rec_z: Union[float, npt.DTypeLike],
                        t0: float, tn: int, src_type: str = None, f0: float = 60):
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates=np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        self.geometry = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates,
                                            t0, tn, src_type=src_type,
                                            f0=None if f0 is None else f0 * 1e-3, fs=self.fs)

    def solve_one_shot(self, isrc: int, wav: npt.DTypeLike = None, dt: float = None,
                       saveu: bool = False):
        """Solve wave equation for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        wav : :obj:`float`, optional
            Wavelet (if not provided, use wavelet in geometry)
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        saveu : :obj:`bool`, optional
            Save snapshots

        Returns
        -------
        d : :obj:`np.ndarray`
            Data
        u : :obj:`np.ndarray`
            Wavefield snapshots

        """

        # create geometry for single source
        geometry = AcquisitionGeometry(self.model, self.geometry.rec_positions, self.geometry.src_positions[isrc, :],
                                       self.geometry.t0, self.geometry.tn, f0=self.geometry.f0,
                                       src_type=self.geometry.src_type, fs=self.fs)
        src = None
        if wav is not None:
            # assign wavelet
            src = RickerSource(name='src', grid=self.model.grid, f0=20,
                               npoint=1, time_range=geometry.time_axis)
            src.coordinates.data[:, 0] = geometry.src.coordinates.data[0, 0]
            src.coordinates.data[:, 1] = geometry.src.coordinates.data[0, 1]
            src.data[:] = wav

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d, u, _ = solver.forward(src=src, save=saveu)

        # resample
        taxis = None
        if dt is not None:
            d = d.resample(dt)
            taxis = d.time_values

        return d, u, taxis

    def solve_all_shots(self, wav: npt.DTypeLike = None, tqdm_signature = None, dt : float = None,
                        figdir : str = None, datadir : str = None, savedtot: bool = False):
        """Solve wave equation for all shots in geometry

        Parameters
        ----------
        wav : :obj:`float`, optional
            Wavelet (if not provided, use wavelet in geometry)
        tqdm_signature : :obj:`func`, optional
            tqdm function handle to use in for loop
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        figdir : :obj:`bool`, optional
            Directory where to save figures for each shot
        datadir : :obj:`bool`, optional
            Directory where to save each shot in npz format
        savedtot : :obj:`bool`, optional
            Save total data

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data

        """

        # Create figure directory
        if figdir is not None:
            if not os.path.exists(figdir):
                os.mkdir(figdir)

        # Create data directory
        if datadir is not None:
            if not os.path.exists(datadir):
                os.mkdir(datadir)

        # Model dataset (serial mode)
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []
        taxis = None
        if tqdm_signature is None:
            tqdm_signature = tqdm
        for isrc in tqdm_signature(range(nsrc)):

            d, _, _ = self.solve_one_shot(isrc, wav=wav, dt=dt)
            if isrc == 0:
                taxis = d.time_values
            if savedtot:
                dtot.append(d.data)

            if datadir is not None:
                np.save(os.path.join(datadir, f'Shot{isrc}'), d)

            if figdir is not None:
                self.plot_shotrecord(d.data, clip=1e-3, figpath=os.path.join(figdir, f'Shot{isrc}'))

        # combine all shots in (s,r,t) cube
        if savedtot:
            dtot = np.array(dtot).transpose(0, 2, 1)
        return dtot, taxis


    def solve_blended_shots(self, wav: npt.DTypeLike = None, dt: float = None, saveu: bool = False):
        """Solve wave equation for blended

        Parameters
        ----------
        wav : :obj:`float`
            Wavelet (if not provided, use wavelet in geometry)
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        saveu : :obj:`bool`, optional
            Save snapshots

        Returns
        -------
        d : :obj:`np.ndarray`
            Data
        u : :obj:`np.ndarray`
            Wavefield snapshots

        """

        # create geometry for single source
        geometry = AcquisitionGeometry(self.model, self.geometry.rec_positions, self.geometry.src_positions,
                                       self.geometry.t0, self.geometry.tn, f0=self.geometry.f0,
                                       src_type=self.geometry.src_type, fs=self.fs)
        src = None
        if wav is not None:
            # assign wavelet
            src = RickerSource(name='src', grid=self.model.grid, f0=20,
                               npoint=wav.shape[1], time_range=geometry.time_axis)
            src.coordinates.data[:, 0] = self.geometry.src.coordinates.data[:, 0]
            src.coordinates.data[:, 1] = self.geometry.src.coordinates.data[:, 1]
            src.data[:] = wav

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d, u, _ = solver.forward(src=src, save=saveu)

        # resample
        if dt is not None:
            d = d.resample(dt)
            taxis = d.time_values

        return d, u, taxis

    def plot_velocity(self, source=True, receiver=True, colorbar=True, cmap="jet", figsize=(13, 5), figpath=None):
        """Display velocity model

        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        source : :obj:`bool`, optional
            Display sources
        receiver : :obj:`bool`, optional
            Display receivers
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        cmap : :obj:`str`, optional
            Colormap
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """
        domain_size = 1.e-3 * np.array(self.model.domain_size)
        extent = [self.model.origin[0], self.model.origin[0] + domain_size[0],
                  self.model.origin[1] + domain_size[1], self.model.origin[1]]

        slices = list(slice(self.model.nbl, -self.model.nbl) for _ in range(2))
        if self.model.fs:
            slices[1] = slice(0, -self.model.nbl)
        if getattr(self.model, 'vp', None) is not None:
            field = self.model.vp.data[slices]
        else:
            field = self.model.lam.data[slices]

        plt.figure(figsize=figsize)
        plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                          vmin=np.min(field), vmax=np.max(field),
                          extent=extent)
        plt.xlabel('X position (km)')
        plt.ylabel('Depth (km)')

        # Plot source points, if provided
        if receiver:
            plt.scatter(1e-3 * self.geometry.rec_positions[:, 0], 1e-3 * self.geometry.rec_positions[:, 1],
                        s=25, c='w', marker='D')

        # Plot receiver points, if provided
        if source:
            plt.scatter(1e-3 * self.geometry.src_positions[:, 0], 1e-3 * self.geometry.src_positions[:, 1],
                        s=25, c='red', marker='o')

        # Ensure axis limits
        #plt.xlim(self.model.origin[0], self.model.origin[0] + domain_size[0])
        #plt.ylim(self.model.origin[1] + domain_size[1], self.model.origin[1])

        # Create aligned colorbar on the right
        if colorbar:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.set_label('Velocity (km/s)')

        # Save figure
        if figpath:
            plt.savefig(figpath)


    def plot_shotrecord(self, rec, colorbar=True, clip=1, figsize=(17, 12), figpath=None):
        """Plot a shot record (receiver values over time).


        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        rec : :obj:`np.ndarray`, optional
            Receiver data of shape (time, points).
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        clip : :obj:`str`, optional
            Clipping
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """

        scale = np.max(rec) * clip
        extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.domain_size[0],
                  1e-3 * self.geometry.tn, self.geometry.t0]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot = ax.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
        ax.axis('tight')
        ax.set_xlabel('X position (km)')
        ax.set_ylabel('Time (s)')

        # Create aligned colorbar on the right
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plot, cax=cax)

        # Save figure
        if figpath:
            plt.savefig(figpath)

        return ax
