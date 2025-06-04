from bonded_dem import get_particle_array_bonded_dem
import numpy as np


def test_pair_wise_contacts():
    # create particle array.
    # each particle has a radius of rad
    rad = 0.1
    # create 3 rows and 3 columns of particles
    x = np.array([-rad, rad, 2 * rad, -rad, rad, 2 * rad, -rad, rad, 2 * rad])
    y = np.array([-rad, -rad, -rad, rad, rad, rad, 2 * rad, 2 * rad, 2 * rad])
    max_nbrs = 12
    pa = get_particle_array_bonded_dem(
        x=x, y=y, rad=rad, constants=dict(no_bonds_limits=10,
                                          criterion_dist=3. * rad))

    # get the contacts of particle index 4
    len_nbrs = 9
    expected_nbrs = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_setting_the_contacts():
    # create particle array.
    # each particle has a radius of rad
    rad = 0.1
    # create 3 rows and 3 columns of particles
    x = np.array([-rad, rad, 2 * rad, -rad, rad, 2 * rad, -rad, rad, 2 * rad])
    y = np.array([-rad, -rad, -rad, rad, rad, rad, 2 * rad, 2 * rad, 2 * rad])
    pa = get_particle_array_bonded_dem(
        x=x, y=y, h=3.*rad, rad=rad, constants=dict(no_bonds_limits=10,
                                                     criterion_dist=3. * rad))

    # get the contacts of particle index 4
    len_nbrs = 9
    expected_nbrs = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # the neighbours of the particle index 4 will be from
    nbrs = []

    for i in range(4 * pa.no_bonds_limits[0],
                   4 * pa.no_bonds_limits[0] + pa.tot_cnts[4]):
        nbrs.append(pa.cnt_idxs[i])
    print(pa.cnt_idxs)

    # now sort the nbrs
    nbrs.sort()

    assert (nbrs == expected_nbrs)
    assert (len_nbrs == pa.tot_ctcs[4])


test_setting_the_contacts()
