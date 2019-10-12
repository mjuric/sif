import numpy as np

def to_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def eclToEqu(ecl):
    # ecliptic coordinate system defined with IAU1976 obliquity
    # Match OpenOrb and https://lambda.gsfc.nasa.gov/toolbox/tb_coordconv.cfm
    #
    # ecl.shape must be (3, ncoords)
    eps = np.deg2rad(23.43929111111111)
    cos_eps, sin_eps = np.cos(eps), np.sin(eps)
    
    R = np.zeros((3,3))
    R[0, :] = [1, 0, 0]
    R[1, :] = [0, cos_eps, -sin_eps]
    R[2, :] = [0, sin_eps, cos_eps]

    equ = np.matmul(R, ecl)

    return equ