#!/usr/bin/env python

"""Tests for `ellipt2d` package."""

import pytest
import numpy


from ellipt2d import Ellipt2d
from triangle import Triangle
import numpy


@pytest.fixture
def simple_one_cell_mesh():
    # one reference triangle
    t = Triangle()
    points = [(0., 0.), (1., 0.), (0., 1.)]
    markers = [1, 1, 1]
    t.set_points(points, markers=markers)
    segments = [(0, 1), (1, 2), (2, 0)]
    t.set_segments(segments)
    t.triangulate(area=0.5)
    return t

@pytest.fixture
def simple_one_cell_mesh2():
    # one triangle, distorted
    t = Triangle()
    points = [(0.1, 0.2), (0.9, 0.4), (0.3, 1.2)]
    markers = [1, 1, 1]
    t.set_points(points, markers=markers)
    segments = [(0, 1), (1, 2), (2, 0)]
    t.set_segments(segments)
    t.triangulate(area=0.5)
    return t

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


def test_one_cell_problem(simple_one_cell_mesh):
    cells = simple_one_cell_mesh.get_triangles()
    ncells = len(cells)
    assert ncells == 1
    f = numpy.ones(ncells, numpy.float64)
    g = numpy.zeros(ncells, numpy.float64)
    s = numpy.zeros(ncells, numpy.float64)
    s[0] = 1.0
    problem = Ellipt2d(simple_one_cell_mesh, f=f, g=g, s=s)
    EPS = 1.e-10
    # check stiffness matrix
    assert abs(problem.amat[0, 0] - 1) < EPS
    assert abs(problem.amat[0, 1] - (-0.5)) < EPS
    assert abs(problem.amat[0, 2] - (-0.5)) < EPS
    assert abs(problem.amat[1, 1] - 0.5) < EPS
    assert abs(problem.amat[1, 2] - 0.) < EPS
    assert abs(problem.amat[2, 2] - 0.5) < EPS
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(problem.amat[i, j] - problem.amat[j, i]) < EPS
    # check loading term
    assert abs(problem.b[0] - 1/6.) < EPS
    assert abs(problem.b[1] - 0.) < EPS
    assert abs(problem.b[2] - 0.) < EPS
    # matrix is singular
    #solution = problem.solve()
    #assert len(solution) == 3


def test_one_cell_problem2(simple_one_cell_mesh2):
    cells = simple_one_cell_mesh2.get_triangles()
    ncells = len(cells)
    assert ncells == 1
    f = numpy.ones(ncells, numpy.float64)
    g = numpy.zeros(ncells, numpy.float64)
    s = numpy.zeros(ncells, numpy.float64)
    s[0] = 1.0
    problem = Ellipt2d(simple_one_cell_mesh2, f=f, g=g, s=s)
    EPS = 1.e-6
    # check stiffness matrix
    assert abs(problem.amat[0, 0] - 0.657895) < EPS
    assert abs(problem.amat[0, 1] - (-0.447368)) < EPS
    assert abs(problem.amat[0, 2] - (-0.210526)) < EPS
    assert abs(problem.amat[1, 1] - 0.684211) < EPS
    assert abs(problem.amat[1, 2] - (-0.236842)) < EPS
    assert abs(problem.amat[2, 2] - 0.447368) < EPS
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(problem.amat[i, j] - problem.amat[j, i]) < EPS
    # check loading term
    assert abs(problem.b[0] - 0.126667) < EPS
    assert abs(problem.b[1] - 0) < EPS
    assert abs(problem.b[2] - 0) < EPS
    # matrix is singular
    #solution = problem.solve()
    #assert len(solution) == 3



