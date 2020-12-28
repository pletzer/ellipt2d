#!/usr/bin/env python

"""Tests for `ellipt2d` package."""

import pytest
import numpy


from ellipt2d import Ellipt2d
from triangle import Triangle



def test_one_cell_mesh():
    t = Triangle()
    points = [(0., 0.), (1., 0.), (0., 1.)]
    markers = [1, 1, 1]
    t.set_points(points, markers=markers)
    segments = [(0, 1), (1, 2), (2, 0)]
    t.set_segments(segments)
    t.triangulate(area=0.5)
    cells = t.get_triangles()
    assert len(cells) == 1



# @pytest.fixture
# def simple_problem(simple_mesh):
#     import ellipt2d
#     t = simple_mesh()
#     cells = t.get_triangles()
#     ncells = len(cells)
#     f = numpy.ones(ncells, numpy.float64)
#     g = numpy.zeros(ncells, numpy.float64)
#     s = numpy.zeros(ncells, numpy.float64)
#     s[0] = 1.0
#     e = ellip2d.ellip2d(t, f=f, g=g, s=s)
#     return e


# def test_simple_mesh(simple_mesh, simple_problem):
#     e = simple_problem(simple_mesh)
#     x = e.solve()
#     print(x)
#     # assert here
