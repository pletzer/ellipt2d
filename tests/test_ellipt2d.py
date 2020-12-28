#!/usr/bin/env python

"""Tests for `ellipt2d` package."""

import pytest
import numpy


from ellipt2d import Ellipt2d
from triangle import Triangle
import numpy



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


@pytest.fixture
def simple_one_cell_mesh():
    t = Triangle()
    points = [(0., 0.), (1., 0.), (0., 1.)]
    markers = [1, 1, 1]
    t.set_points(points, markers=markers)
    segments = [(0, 1), (1, 2), (2, 0)]
    t.set_segments(segments)
    t.triangulate(area=0.5)
    return t

def test_one_cell_problem(simple_one_cell_mesh):
    cells = simple_one_cell_mesh.get_triangles()
    ncells = len(cells)
    assert ncells == 1
    f = numpy.ones(ncells, numpy.float64)
    g = numpy.zeros(ncells, numpy.float64)
    s = numpy.zeros(ncells, numpy.float64)
    s[0] = 1.0
    problem = Ellipt2d(simple_one_cell_mesh, f=f, g=g, s=s)
    solution = problem.solve()
    assert len(solution) == 3

