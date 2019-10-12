import os, os.path, sys

if 'OORB_DATA' not in os.environ:
    os.environ['OORB_DATA'] = '/Users/mjuric/projects/lsst_ssp/oorb-lynne/data'

extra_paths = [
    '/Users/mjuric/projects/lsst_ssp/oorb-lynne/python',
]
for _p in extra_paths:
    if not os.path.isdir(_p):
        print(f"{_p} not present. Skipping.")
        continue

    if _p not in sys.path:
        sys.path += [ _p ]
        print(f"Added {_p}")

from . import api, utils

from pyoorb import pyoorb
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

__all__ = ["Orbits", "Propagator"]

timeScales = dict( UTC=1, UT1=2, TT=3, TAI=4 )
elemType   = dict( CAR=1, COM=2, KEP=3, DEL=4, EQX=5 )

inv_timeScales = dict(zip(timeScales.values(), timeScales.keys()))
inv_elemType   = dict(zip(elemType.values(), elemType.keys()))

def _to_pyoorb_representation(df):
    """Convert orbital elements into the numpy fortran-format array OpenOrb requires.

    The OpenOrb element format is a single array with elemenets:
    0 : orbitId (cannot be a string)
    1-6 : orbital elements, using radians for angles
    7 : element 'type' code (1 = CAR, 2 = COM, 3 = KEP, 4 = DELauny, 5 = EQX (equinoctial))
    8 : epoch
    9 : timescale for epoch (1 = UTC, 2 = UT1, 3 = TT, 4 = TAI : always assumes TT)
    10 : magHv
    11 : g

    Sets self.oorbElem, the orbit parameters in an array formatted for OpenOrb.
    """
    oorbElem = np.empty( (len(df), 12), dtype=np.double, order='F')
    oorbElem[:, 0] = np.arange(len(df))

    if 'objId' in df:
        id = df['objId'].values
    else:
        id = np.arange(len(df))

    # Add the appropriate element and epoch types:
    # Convert other elements INCLUDING converting inclination, node, argperi to RADIANS
    if 'meanAnomaly' in df:
        oorbElem[:, 1] = df['a']
        oorbElem[:, 2] = df['e']
        oorbElem[:, 3] = np.radians(df['inc'])
        oorbElem[:, 4] = np.radians(df['Omega'])
        oorbElem[:, 5] = np.radians(df['argPeri'])
        oorbElem[:, 6] = np.radians(df['meanAnomaly'])
        oorbElem[:, 7] = elemType['KEP']
    elif 'argPeri' in df:
        oorbElem[:, 1] = df['q']
        oorbElem[:, 2] = df['e']
        oorbElem[:, 3] = np.radians(df['inc'])
        oorbElem[:, 4] = np.radians(df['Omega'])
        oorbElem[:, 5] = np.radians(df['argPeri'])
        oorbElem[:, 6] = df['tPeri']
        oorbElem[:, 7] = elemType['COM']
    elif 'x' in df:
        oorbElem[:, 1] = df['x']
        oorbElem[:, 2] = df['y']
        oorbElem[:, 3] = df['z']
        oorbElem[:, 4] = df['xdot']
        oorbElem[:, 5] = df['ydot']
        oorbElem[:, 6] = df['zdot']
        oorbElem[:, 7] = elemType['CAR']
    else:
        raise ValueError(f'Unsupported element type %s: should be one of KEP, COM or CAR.' % elem_type)

    oorbElem[:,8] = df['epoch']
    oorbElem[:,9] = timeScales['TT']

    if 'H' in df or 'G' in df:
        oorbElem[:,10] = df['H']
        oorbElem[:,11] = df['G']
    else:
        oorbElem[:,10] = '0'
        oorbElem[:,11] = 0.15

    return id, oorbElem

def _from_pyoorb_representation(id, oorbElem, element_type):
    et = elemType[element_type]

    # if oorbElem are not of the right element type, convert
    if np.any(oorbElem[:, 7] != et):
        _init_pyoorb()
        oorbElem, err = pyoorb.oorb_element_transformation(in_orbits=oorbElem, in_element_type=et)
        if err != 0:
            raise RuntimeError('Oorb returned error code %s' % (err))

    # convert timescales, if not in TT
    epoch = oorbElem[:, 8]
    if np.any(oorbElem[:, 9] != timeScales['TT']):
        if np.any(oorbElem[:, 9] != oorbElem[0, 9]):
            # this should never happen if only manipulating the states through the public interface
            raise ValueError('Unsupported: mixed timescales in internal pyoorb array')

        scale = inv_timeScales[oorbElem[0, 9]]
        epoch = Time(epoch, format='mjd', scale=scale.lower()).tt

    # convert to dataframe
    df = pd.DataFrame({'objId': id})

    if element_type == 'KEP':
        df['a']           = oorbElem[:, 1]
        df['e']           = oorbElem[:, 2]
        df['inc']         = np.rad2deg(oorbElem[:, 3])
        df['Omega']       = np.rad2deg(oorbElem[:, 4])
        df['argPeri']     = np.rad2deg(oorbElem[:, 5])
        df['meanAnomaly'] = np.rad2deg(oorbElem[:, 6])
    elif element_type == 'COM':
        df['q']       = oorbElem[:, 1]
        df['e']       = oorbElem[:, 2]
        df['inc']     = np.rad2deg(oorbElem[:, 3])
        df['Omega']   = np.rad2deg(oorbElem[:, 4])
        df['argPeri'] = np.rad2deg(oorbElem[:, 5])
        df['tPeri']   = oorbElem[:, 6]
    elif element_type == 'CAR':
        df['x']    = oorbElem[:, 1]
        df['y']    = oorbElem[:, 2]
        df['z']    = oorbElem[:, 3]
        df['xdot'] = oorbElem[:, 4]
        df['ydot'] = oorbElem[:, 5]
        df['zdot'] = oorbElem[:, 6]
    else:
        raise ValueError(f'Unsupported element type %s: should be one of KEP, COM or CAR.' % elem_type)

    df['epoch'] = epoch

    df['H'] = oorbElem[:, 10]
    df['G'] = oorbElem[:, 11]

    return df

class Orbits(api.Orbits):

    def __init__(self, id, elem):
        self._id, self._elem = id, elem

    def to_df(self, element_type='KEP'):
        # convert from internal representation to dataframe
        return _from_pyoorb_representation(self._id, self._elem, element_type)

    @staticmethod
    def from_df(elementsDF):
        return Orbits(*_to_pyoorb_representation(elementsDF))

    def __len__(self):
        return len(self._elem)

    def __getitem__(self, i):
        return Orbits(self._id[i], self._elem[i])

def _astropy_time_to_oorb_time(t):
    # time can be a scalar, or a numpy array (both wrapped into astropy.Time object)
    try:
        nt = len(t)
    except TypeError:
        nt = 1

    oorbT = np.empty([nt, 2], dtype='double', order='F')

    oorbT[:, 0] = t.mjd
    oorbT[:, 1] = timeScales[t.scale.upper()]

    return oorbT

oorb_initialized = False
def _init_pyoorb():
    # Initialize oorb. Very important. You will get '0's as earth positions otherwise.
    # FIXME: the try/except construct ensures oorb_init is called only once. Calling it more than once messes up oorb.
    #        This should be fixed in oorb.
    global oorb_initialized
    if not oorb_initialized:
        ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
        err = pyoorb.oorb_init(ephfile)
        if err != 0:
            raise RuntimeError('error: pyoorb.oorb_init: errcode=%d' % err)
        oorb_initialized = True

def _convert_ephems_basic_to_df(oorbEphs, ids, all):
    """Converts oorb ephemeris array to numpy recarray, with labeled columns.

    Parameters
    ----------
    oorbEphs : numpy.ndarray
        The oorb-formatted ephemeris values

    Returns
    -------
    pandas.DataFrame
        The table of ephemerides.
    """

    df = pd.DataFrame(oorbEphs.reshape(-1, oorbEphs.shape[-1]))
    df.columns = ['time', 'ra', 'dec', 'dradt', 'ddecdt', 'phase', 'solarelon', 'helio_dist', 'geo_dist', 'magV', 'trueAnomaly']
    df['velocity'] = np.sqrt(df['dradt']**2 + df['ddecdt']**2)

    if all:
        ids = np.tile(ids, oorbEphs.shape[0])        
    df.insert(0, 'objId', ids)

    return df

class Propagator(api.Propagator):

    def __init__(self, dynmodel='N'):
        _init_pyoorb()
        self.dynmodel = dynmodel

    def propagate(self, orbits, epoch):
        # Propagates the orbits to the requested epoch.
        # in: orbits -- subclass of Orbits
        # in: epoch -- subclass of astropy.Time
        # returns: Orbits instance at epoch `epoch`
        assert type(orbits) == Orbits

        t = _astropy_time_to_oorb_time(epoch)
        elem, err = pyoorb.oorb_propagation(in_orbits=orbits._elem, in_dynmodel=self.dynmodel, in_epoch=t)
        if err != 0:
            raise RuntimeError('error: pyoorb.oorb_propagation: errcode=%d' % err)
        return Orbits(orbits._id, elem)

    def observerPosition(self, obscode, epochs):
        # in: obscode -- MPC observatory code
        # in: epochs -- subclass of astropy.Time
        # returns: astropy.SkyCoord with the observer's position
        t = _astropy_time_to_oorb_time(epochs)
        earthPos, err = pyoorb.oorb_observer(in_obscode=obscode, in_date_ephems=t)
        if err != 0:
            raise RuntimeError('error: pyoorb.oorb_observer: errcode=%d' % err)

        return SkyCoord(
            representation_type='cartesian',
            x=earthPos[:, 1], y=earthPos[:, 2], z=earthPos[:, 3],
            unit='au',
            obstime=epochs,
            frame='heliocentriceclipticiau76',
            )

    def ephemerides(self, obscode, orbits, epochs, all=True):
        assert type(orbits) == Orbits

        t = _astropy_time_to_oorb_time(epochs)
        if all:
            ephems, err = pyoorb.oorb_ephemeris_basic(orbits._elem, in_obscode=obscode, in_date_ephems=t, in_dynmodel=self.dynmodel)
            if err != 0:
                raise RuntimeError('error: pyoorb.oorb_ephemeris_basic: errcode=%d' % err)
        else:
            assert len(orbits._elem) == len(epochs)
            ephems = np.empty((len(orbits._elem), 11))
            for idx, _ in enumerate(orbits._elem):
                ephems[idx], err = pyoorb.oorb_ephemeris_basic(orbits._elem[idx:idx+1], in_obscode=obscode, in_date_ephems=t[idx:idx+1], in_dynmodel=self.dynmodel)
                if err != 0:
                    raise RuntimeError('error: pyoorb.oorb_ephemeris_basic: errcode=%d' % err)

        return _convert_ephems_basic_to_df(ephems, orbits._id, all)

if __name__ == "__main__":
    # Load the orbits file
    orbits = Orbits.from_des('../../analyses/impactors.des.txt')
    orbits = orbits[:2]
    df = orbits.to_df()

    # Propagation tests
    prop = Propagator()

    t = Time(df['epoch'] - 10, scale='tt', format='mjd')
    neworb = prop.propagate(orbits, t[0])

    dfout = neworb.to_df('COM')
    print(dfout)

    # Observer position tests
    obsPos = prop.observerPosition('I11', t)
    print(obsPos)
    from utils import eclToEqu
    equ = eclToEqu(obsPos.cartesian.xyz.value)
    equScMy = SkyCoord(x=equ[0], y=equ[1], z=equ[2], representation_type='cartesian', frame='hcrs', unit='au', obstime=obsPos.obstime)
    equScAp = obsPos.transform_to('hcrs')
    print(equScMy)
    print(equScAp)
    print(equScMy.cartesian.xyz - equScAp.cartesian.xyz)

    # Ephemerides
    ephems = prop.ephemerides('I11', orbits, t)
    print(ephems)
