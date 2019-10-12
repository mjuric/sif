import ssdapi.oorb as ssd
from astropy.time import Time
from astropy.coordinates import SkyCoord

if __name__ == "__main__":
    # Load the orbits file
    orbits = ssd.Orbits.from_des('../analyses/impactors.des.txt')
    orbits = orbits[:2]
    df = orbits.to_df()

    # Propagation tests
    prop = ssd.Propagator()

    t = Time(df['epoch'] - 10, scale='tt', format='mjd')
    neworb = prop.propagate(orbits, t[0])

    dfout = neworb.to_df('COM')
    print(dfout)

    # Observer position tests
    obsPos = prop.observerPosition('I11', t)
    print(obsPos)
    from ssdapi.utils import eclToEqu
    equ = eclToEqu(obsPos.cartesian.xyz.value)
    equScMy = SkyCoord(x=equ[0], y=equ[1], z=equ[2], representation_type='cartesian', frame='hcrs', unit='au', obstime=obsPos.obstime)
    equScAp = obsPos.transform_to('hcrs')
    print(equScMy)
    print(equScAp)
    print(equScMy.cartesian.xyz - equScAp.cartesian.xyz)

    # Ephemerides
    ephems = prop.ephemerides('I11', orbits, t)
    print(ephems)
