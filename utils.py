import numpy as np


def fixed_to_fixed_streamer(sx, rx, data, nrec_streamer=200):
    """From fixed receiver geometry to fixed streamer

    Convert a dataset from fixed receiver geometry to one with fixed streamer geometry
    behind a source (to take into account the boat velocity, use ``fixed_to_continous_streamer``
    instead)

    Parameters
    ----------
    sx
    rx
    data
    nrec_streamer

    Returns
    -------
    data_streamer

    """
    nsrc, nrec, nt = len(sx), len(rx), data.shape[2]
    data_streamer = np.zeros((nsrc, nrec_streamer, nt))

    for isrc in range(nsrc):
        # find receiver at location of current source
        irec = np.argmin(np.abs(sx[isrc] - rx))
        data_streamer[isrc] = data[isrc, irec - nrec_streamer:irec]
    data_streamer = np.flip(data_streamer, axis=1)

    return data_streamer


def fixed_to_continous_streamer(rx, dt, sxin, vboat, timings, data, nrec_streamer=200):
    """From fixed receiver geometry to continuos streamer

    Convert a dataset from fixed receiver geometry to one with continously moving streamer geometry
    by taking into account to boat velocity ``vboat``

    Parameters
    ----------
    rx
    dt
    sxin
    vboat
    timings
    data
    nrec_streamer

    Returns
    -------
    data_streamer

    """
    nt, nrec = data.shape
    orec = rx[0]
    drec = rx[1] - rx[0]

    # define source array
    sx = sxin + timings * vboat
    sx_t = sxin + np.arange(nt) * (dt * vboat)

    # extract data with all receivers in streamer configuration
    irec_t = np.round((sx_t - orec) / drec).astype(np.int64)

    data_streamer = np.zeros((nt, nrec_streamer))
    for it in range(nt):
        data_streamer[it] = data[it, (irec_t[it] - nrec_streamer):irec_t[it]]

    return sx, data_streamer, irec_t