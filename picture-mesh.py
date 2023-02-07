#! /usr/bin/env python3
#
# Author: Clemens Verhoosel
#
# This example demonstrates how to construct an immersed (isogeometric) finite
# element mesh directly from a two-dimensional image file.

from nutils import cli, mesh, export, function, testing
from skimage import color, io
from typing import Tuple, Optional
from matplotlib import collections
import treelog, pathlib, numpy

def picture_mesh(image:pathlib.Path, elems:Tuple[int,int], levelset_refine:Optional[int], degree:int, quadtree_refine:Optional[int]=None):

    '''
    Nutils script generating an immersed mesh based on an image file.

    .. arguments::

        image [./images/TUe-logo.jpg]
            Path to the image file

        elems [20,10]
            Number of elements per direction

        levelset_refine [2]
            Number of refinements for levelset mesh (default: based on image)

        degree [2]
            B-spline smoothing degree

        quadtree_refine []
            Number of quadtree bisections (default: equal to levelset_refine)
    '''

    # Load and plot the original image
    im = io.imread(image)

    with export.mplfigure('original.png') as fig:
        ax = fig.add_subplot(111)
        ax.imshow(im)

    # Convert to a grayscale image
    im = color.rgb2gray(im)

    with export.mplfigure('grayscale.png') as fig:
        ax = fig.add_subplot(111)
        clrs = ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        fig.colorbar(clrs, orientation='horizontal')

    # Construct the numpy grayscale voxel data
    data = numpy.flip(im, axis=0).T

    # Construct the ambient mesh
    lengths = numpy.array([1.,data.shape[1]/data.shape[0]])
    ambient_domain, geom = mesh.rectilinear([numpy.linspace(0,l,e+1) for e, l in zip(elems,lengths)])

    # Construct the levelset domain by refining the ambient domain
    voxel_refine = min(numpy.floor(numpy.log2(sh/ne)).astype(int) for sh, ne in zip(data.shape,ambient_domain.shape))
    if levelset_refine is None or levelset_refine > voxel_refine:
        levelset_refine = voxel_refine

    levelset_domain  = ambient_domain.refine(levelset_refine)
    voxel_spacing    = numpy.array([l/s for l,s in zip(lengths, data.shape)])
    treelog.user(f'levelset shape: {"×".join(str(d) for d in levelset_domain.shape)}')

    # Sample the levelset domain
    levelset_sample = levelset_domain.sample('uniform', 2**(voxel_refine-levelset_refine+1))
    levelset_points = levelset_sample.eval(geom)

    # Find the voxel data values corresponding to the levelset points
    indf = levelset_points/voxel_spacing
    indi = numpy.maximum(numpy.minimum(numpy.floor(indf),numpy.array(data.shape)-1),0).astype(int)
    levelset_data = data[tuple(indi.T)]

    # Construct the voxel intensity function
    intensity = levelset_sample.basis().dot(levelset_data)

    # Smoothen the intensity data using a B-spline basis
    basis    = levelset_domain.basis('spline', degree)
    den, num = levelset_sample.integrate([basis, intensity*basis])
    levelset = basis.dot(num/den)

    bezier = levelset_domain.sample('bezier', degree+1)
    points, vals = bezier.eval([geom, levelset])
    with export.mplfigure('levelset.png') as fig:
        ax = fig.add_subplot(111, aspect='equal')
        ax.autoscale(enable=True, axis='both', tight=True)
        im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap='jet')
        ax.add_collection(collections.LineCollection(points[bezier.hull], colors='k', linewidth=.5, alpha=0.5))
        fig.colorbar(im, orientation='horizontal')

    # Trim the domain
    if quadtree_refine is None or quadtree_refine < levelset_refine:
        quadtree_refine = levelset_refine

    domain = ambient_domain.trim(0.5-levelset, maxrefine=quadtree_refine, leveltopo=levelset_domain)

    sub_bezier = domain.sample(bezier_nodedup, 2)
    sub_points = sub_bezier.eval(geom)
    boundary_bezier = domain.boundary.sample('bezier', 2)
    boundary_points = boundary_bezier.eval(geom)
    ambient_bezier = ambient_domain.sample('bezier', 2)
    ambient_points = ambient_bezier.eval(geom)
    with export.mplfigure('mesh.png') as fig:
        ax = fig.add_subplot(111, aspect='equal', xlim=(0,lengths[0]), ylim=(0,lengths[1]))
        ax.add_collection(collections.LineCollection(boundary_points[boundary_bezier.tri], colors='k'))
        ax.add_collection(collections.LineCollection(ambient_points[ambient_bezier.hull], colors='k', linewidth=.5, alpha=0.5))
        ax.add_collection(collections.LineCollection(sub_points[sub_bezier.hull], colors='k', linewidth=1, alpha=0.5))

    # Post-processing
    area = domain.integrate(function.J(geom), ischeme='gauss1')
    circumference = domain.boundary['trimmed'].integrate(function.J(geom), ischeme='gauss1')

    treelog.user(f'domain area          : {area:5.4f}')
    treelog.user(f'domain circumference : {circumference:5.4f}')

    return area, circumference

def bezier_nodedup(ref, degree):
    '''drop-in replacement for a bezier sample with deduplication of common
    points disabled, so that the resulting hull traces interior interfaces.'''

    from nutils import element, points
    return ref.getpoints('bezier', degree) if not isinstance(ref, element.WithChildrenReference) \
      else points.ConcatPoints(tuple(points.TransformPoints(bezier_nodedup(child, degree//2+1), trans)
        for trans, child in ref.children if child))

if __name__=='__main__':
    cli.run(picture_mesh)

# Unit testing
class test(testing.TestCase):
    def test_tuelogo(self):
        area, circumference = picture_mesh('./images/TUe-logo.jpg', (20,10), 2, 2)
        with self.subTest('area'): self.assertAlmostEqual(area, 0.0896274547240972, places=6)
        with self.subTest('circumference'): self.assertAlmostEqual(circumference, 3.993044249138634, places=6)