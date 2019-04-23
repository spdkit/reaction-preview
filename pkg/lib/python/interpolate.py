#! /usr/bin/env python
# core

# [[file:~/Workspace/Paperwork/reaction-preview/codes/reaction-preview.note::*core][core:1]]
import numpy as np

import ase
import ase.io
from ase.neb import NEB

# adopted from: https://wiki.fysik.dtu.dk/ase/tutorials/neb/idpp.html
def create_idpp_images(freactant, fproduct, nimages=11):
    """ create images interpolated using IDPP algorithm

    Parameters
    ----------
    freactant : path to initial atoms
    fproduct  : path to final atoms
    nimages   :   the number of images to be interpolated including two endpoints
    """
    images = _create_images(freactant, fproduct, nimages)

    # Interpolate intermediate images
    neb = NEB(images, remove_rotation_and_translation=True)
    neb.interpolate("idpp")
    return images

# adopted from: ase/neb.py `idpp_interpolate` function
def create_lst_images(freactant, fproduct, nimages=11, mic=False):
    """create images with original LST algorithm without NEB force projection as in IDPP

    Parameters
    ----------
    freactant : path to initial atoms
    fproduct  : path to final atoms
    nimages   : the number of images to be interpolated including two endpoints
    mic       : apply mic or not (PBC)
    """
    from ase.neb import IDPP
    from ase.optimize import BFGS
    from ase.build import minimize_rotation_and_translation

    # create linearly interpolated images
    images = _create_images(freactant, fproduct, nimages)
    neb = NEB(images, remove_rotation_and_translation=True)
    neb.interpolate()

    # refine images with LST algorithm
    d1 = images[0].get_all_distances(mic=mic)
    d2 = images[-1].get_all_distances(mic=mic)
    d = (d2 - d1) / (nimages - 1)
    for i, image in enumerate(images):
        image.set_calculator(IDPP(d1 + i * d, mic=mic))
        qn = BFGS(image)
        qn.run(fmax=0.1)
    # apply optimal translation and rotation
    for i in range(nimages-1):
        minimize_rotation_and_translation(images[i], images[i+1])

    return images

def _create_images(freactant, fproduct, nimages):
    # Read endpoint structures
    image_reactant = ase.io.read(freactant)
    image_product  = ase.io.read(fproduct)

    # Make a band consisting of `nimages` images:
    images = [image_reactant]
    images += [image_reactant.copy() for i in range(nimages-2)]
    images += [image_product]

    return images
# core:1 ends here
