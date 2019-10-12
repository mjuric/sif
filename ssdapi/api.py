from abc import ABC, abstractmethod

## Universal integrator API thoughts
## Workflow:
##  -- instantiate the integrator
##  -- upload the orbits to integrate, return a handle
##  -- call propagation and ephemeris calculation methods
## API:
##    init(**kwargs)
##    set_orbits(orbits)

class Orbits(ABC):
    @abstractmethod
    def __init__(self, elementsDF):
        pass

    @abstractmethod
    def to_df(self, element_type='KEP'):
        pass

    @staticmethod
    @abstractmethod
    def from_df(elementsDF):
        pass

    @classmethod
    def from_des(cls, fn, delim=None, skiprows=None):
        df = read_des(fn, delim=delim, skiprows=skiprows)
        return cls.from_df(df)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    def __iter__(self):
        for k in range(len(self)):
            yield self[k, k+1]


class Propagator(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def propagate(self, orbits, epoch):
        # Propagates the orbits to the requested epoch.
        # in: orbits -- subclass of Orbits, orbits to propagate
        # in: epoch -- subclass of astropy.Time
        # returns: Orbits instance at epoch `epoch`
        pass

    @abstractmethod
    def observerPosition(self, obscode, epochs):
        # in: obscode -- MPC observatory code
        # in: epochs -- subclass of astropy.Time
        # returns: astropy.SkyCoord with the observer's position
        pass

    @abstractmethod
    def ephemerides(self, obscode, orbits, epochs, all=True):
        # in: obscode -- MPC observatory code
        # in: orbits -- subclass of Orbits, orbits to compute ephemerides for
        # in: epochs -- subclass of astropy.Time
        # in: all -- if True, compute (norbits x nepochs) ephemerides; otherwise
        #            assume len(epochs) == len(orbits) and compute ephemerides
        #            for orbit orbits[k] at epoch epochs[k].
        # returns: dataframe with the ephemerides (equatorial coordinates)
        pass

###################

dataCols = dict(
    COM = ['objId', 'q', 'e', 'inc', 'Omega', 'argPeri', 'tPeri', 'epoch', 'H', 'G'],
    KEP = ['objId', 'a', 'e', 'inc', 'Omega', 'argPeri', 'meanAnomaly', 'epoch', 'H', 'G'],
    CAR = ['objId', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot', 'epoch', 'H', 'G']
)

def read_des(orbitfile, delim=None, skiprows=None):
    """Read orbits from a file, generating a pandas dataframe containing columns matching dataCols,
    for the appropriate orbital parameter format (currently accepts COM, KEP or CAR formats).

    After reading and standardizing the column names, calls self.setOrbits to validate the
    orbital parameters. Expects angles in orbital element formats to be in degrees.

    Note that readOrbits uses pandas.read_csv to read the data file with the orbital parameters.
    Thus, it should have column headers specifying the column names ..
    unless skiprows = -1 or there is just no header line at all.
    in which case it is assumed to be a standard DES format file, with no header line.

    Parameters
    ----------
    orbitfile : str
        The name of the input file containing orbital parameter information.
    delim : str, optional
        The delimiter for the input orbit file. Default is None, will use delim_whitespace=True.
    skiprows : int, optional
        The number of rows to skip before reading the header information for pandas.
        Default is None, which will trigger a check of the file to look for the header columns.
    """
    names = None
    import pandas as pd

    # If skiprows is set, then we will assume the user has handled this so that the
    # first line read has the header information.
    # But, if skiprows is not set, then we have to do some checking to see if there is
    # header information and which row it might start in.
    if skiprows is None:
        skiprows = -1
        # Figure out whether the header is in the first line, or if there are rows to skip.
        # We need to do a bit of juggling to do this before pandas reads the whole orbit file though.
        with open(orbitfile, 'r') as fp:
            headervalues = None
            for line in fp:
                values = line.split()
                try:
                    # If it is a valid orbit line, we expect column 3 to be a number.
                    float(values[3])
                    # And if it worked, we're done here (it's an orbit) - go on to parsing header values.
                    break
                except (ValueError, IndexError):
                    # This wasn't a valid number or there wasn't anything in the third value.
                    # So this is either the header line or it's a comment line before the header columns.
                    skiprows += 1
                    headervalues = values


        if headervalues is not None:  # (and skiprows > -1)
            # There is a header, but we also need to check if there is a comment key at the start
            # of the proper header line.
            # ... Because this varies as well, and is sometimes separated from header columns.
            linestart = headervalues[0]
            if linestart == '#' or linestart == '!!' or linestart == '##':
                names = headervalues[1:]
            else:
                names = headervalues
            # Add 1 to skiprows, so that we skip the header column line.
            skiprows += 1

    # So now skiprows is a value. If it is -1, then there is no header information.
    if skiprows == -1:
        # No header; assume it's a typical DES file -
        # we'll assign the column names based on the FORMAT.
        names_COM = ('objId', 'FORMAT', 'q', 'e', 'i', 'node', 'argperi', 't_p',
                        'H',  'epoch', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE')
        names_KEP = ('objId', 'FORMAT', 'a', 'e', 'i', 'node', 'argperi', 'meanAnomaly',
                        'H', 'epoch', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE')
        names_CAR = ('objId', 'FORMAT', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot',
                        'H', 'epoch', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE')
        # First use names_COM, and then change if required.
        orbits = pd.read_csv(orbitfile, delim_whitespace=True, header=None, names=names_COM)

        if orbits['FORMAT'][0] == 'KEP':
            orbits.columns = names_KEP
        elif orbits['FORMAT'][0] == 'CAR':
            orbits.columns = names_CAR

    else:
        if delim is None:
            orbits = pd.read_csv(orbitfile, delim_whitespace=True, skiprows=skiprows, names=names)
        else:
            orbits = pd.read_csv(orbitfile, sep=delim, skiprows=skiprows, names=names)

    # Drop some columns that are typically present in DES files but that we don't need.
    if 'INDEX' in orbits:
        del orbits['INDEX']
    if 'N_PAR' in orbits:
        del orbits['N_PAR']
    if 'MOID' in orbits:
        del orbits['MOID']
    if 'COMPCODE' in orbits:
        del orbits['COMPCODE']
    if 'tmp' in orbits:
        del orbits['tmp']

    # Normalize the column names to standard values and identify the orbital element types.
    ssoCols = orbits.columns.values.tolist()

    # These are the alternative possibilities for various column headers
    # (depending on file version, origin, etc.)
    # that might need remapping from the on-file values to our standardized values.
    altNames = {}
    altNames['objId'] = ['objId', 'objid', '!!ObjID', '!!OID', '!!S3MID', 'OID', 'S3MID'
                            'objid(int)', 'full_name', '#name']
    altNames['q'] = ['q']
    altNames['a'] = ['a']
    altNames['e'] = ['e', 'ecc']
    altNames['inc'] = ['inc', 'i', 'i(deg)', 'incl']
    altNames['Omega'] = ['Omega', 'omega', 'node', 'om', 'node(deg)',
                            'BigOmega', 'Omega/node', 'longNode']
    altNames['argPeri'] = ['argPeri', 'argperi', 'omega/argperi', 'w', 'argperi(deg)', 'peri']
    altNames['tPeri'] = ['tPeri', 't_p', 'timeperi', 't_peri', 'T_peri']
    altNames['epoch'] = ['epoch', 't_0', 'Epoch', 'epoch_mjd']
    altNames['H'] = ['H', 'magH', 'magHv', 'Hv', 'H_v']
    altNames['g'] = ['g', 'phaseV', 'phase', 'gV', 'phase_g', 'G']
    altNames['meanAnomaly'] = ['meanAnomaly', 'meanAnom', 'M', 'ma']
    altNames['xdot'] = ['xdot', 'xDot']
    altNames['ydot'] = ['ydot', 'yDot']
    altNames['zdot'] = ['zdot', 'zDot']

    # Update column names that match any of the alternatives above.
    for name, alternatives in altNames.items():
        intersection = list(set(alternatives) & set(ssoCols))
        if len(intersection) > 1:
            raise ValueError('Received too many possible matches to %s in orbit file %s'
                                % (name, orbitfile))
        if len(intersection) == 1:
            idx = ssoCols.index(intersection[0])
            ssoCols[idx] = name
    # Assign the new column names back to the orbits dataframe.
    orbits.columns = ssoCols

    ################# Validation and cleanup
    nSso = len(orbits)

    # Error if orbits is empty (this avoids hard-to-interpret error messages from pyoorb).
    if nSso == 0:
        raise ValueError('Length of the orbits dataframe was 0.')

    # Discover which type of orbital parameters we have on disk.
    orb_format = None
    if 'FORMAT' in orbits:
        orbits.loc[orbits['FORMAT'] == 'CART', 'FORMAT'] = 'CAR'
        if ~(orbits['FORMAT'] == orbits['FORMAT'].iloc[0]).all():
            raise ValueError('All orbital elements in the set should have the same FORMAT.')
        orb_format = orbits['FORMAT'].iloc[0]
        # Backwards compatibility .. a bit. CART is deprecated, so swap it to CAR.
        if orb_format == 'CART':
            orb_format = 'CAR'
        del orbits['FORMAT']
        # Check that the orbit format is approximately right.
        if orb_format == 'COM':
            if 'q' not in orbits:
                raise ValueError('The stated format was COM, but "q" not present in orbital elements?')
        if orb_format == 'KEP':
            if 'a' not in orbits:
                raise ValueError('The stated format was KEP, but "a" not present in orbital elements?')
        if orb_format == 'CAR':
            if 'x' not in orbits:
                raise ValueError('The stated format was CAR but "x" not present in orbital elements?')
    if orb_format is None:
        # Try to figure out the format, if it wasn't provided.
        if 'q' in orbits:
            orb_format = 'COM'
        elif 'a' in orbits:
            orb_format = 'KEP'
        elif 'x' in orbits:
            orb_format = 'CAR'
        else:
            raise ValueError("Can't determine orbital type, as neither q, a or x in input orbital elements.\n"
                                "Was attempting to base orbital element quantities on header row, "
                                "with columns: \n%s" % orbits.columns)

    # Check that the orbit epoch is within a 'reasonable' range, to detect possible column mismatches.
    general_epoch = orbits['epoch'].iloc[0]
    # Look for epochs between 1800 and 2200 - this is primarily to check if people used MJD (and not JD).
    expect_min_epoch = -21503.
    expect_max_epoch = 124594.
    if general_epoch < expect_min_epoch or general_epoch > expect_max_epoch:
        raise ValueError("The epoch detected for this orbit is odd - %f. "
                            "Expecting a value between %.1f and %.1f (MJD!)" % (general_epoch,
                                                                                expect_min_epoch,
                                                                                expect_max_epoch))

    # If these columns are not available in the input data, auto-generate them.
    if 'objId' not in orbits:
        objId = np.arange(0, nSso, 1)
        orbits = orbits.assign(objId = objId)
    if 'H' not in orbits:
        orbits = orbits.assign(H = 20.0)
    if 'G' not in orbits:
        orbits = orbits.assign(G = 0.15)

    # Make sure we gave all the columns we need.
    for col in dataCols[orb_format]:
        if col not in orbits:
            raise ValueError('Missing required orbital element %s for orbital format type %s'
                                % (col, orb_format))

    # Check to see if we have duplicates.
    if len(orbits['objId'].unique()) != nSso:
        warnings.warn('There are duplicates in the orbit objId values' +
                        ' - was this intended? (continuing).')
    # All is well.
    return orbits
