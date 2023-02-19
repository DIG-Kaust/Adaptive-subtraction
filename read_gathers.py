import numpy as np
import segyio

def read_gathers(filename):
    """Read seismic gathers from a file and store them into a 3d numpy array.

        Parameters
        ----------
        filename : :obj:`str`
            Path to data file

        Returns
        -------
        cube : :obj:`np.ndarray`
            3d array with all gathers

        """
    with segyio.su.open(filename, ignore_geometry=True, endian='little') as sufile:
        # map all the headers
        sufile.mmap()

        # Extract some headers for all traces
        fldr = sufile.attributes(segyio.su.fldr)[:] # gather number for each trace
        nt_all = sufile.attributes(segyio.su.ns)[:]
        nt = nt_all[0]   # number of samples per trace
        ntr = len(fldr)   # number of traces

        ns = fldr[ntr-1]-fldr[0]+1 # number of shots
        nr = np.unique(fldr, return_counts=True)[1][0] # number of receivers, must be same for all shots
        
        # Create seismic cube to store the data
        cube = np.zeros((ns,nr,nt))  # number of shots, number of receivers, number of samples
        i = 0                        # flag to store complete shots in seismic cube
        
        # Find boundaries for all shots, store in data array
        shot_complete = 0
        nxshot = 0
        
        for itr in range(ntr):
            # determine start of a new shot
            if nxshot == 0:
                # found new shot            
                itr1 = itr
                nxshot = 1 
                fldr_current = fldr[itr]
                data = sufile.trace[itr]
            else:
                #print('add trace %d to current shot' %(itr))
                data = np.vstack((data,sufile.trace[itr]))
                nxshot = nxshot+1
                itr2 = itr

            # if next trace is from new shot or at last trace: mark complete
            if itr < ntr-1:
                if fldr[itr+1] != fldr_current:
                    shot_complete = 1
            else:
                    shot_complete = 1

            # process the shot if it is complete
            if shot_complete > 0:
                shot_complete = 0
                nxshot = 0
                cube[i] = data 
                i += 1
                
    return cube