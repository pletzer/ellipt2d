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

@pytest.fixture
def square_mesh():
    lx, ly = 1.0, 1.0
    nx1, ny1 = 3, 4
    nx, ny = nx1 - 1, ny1 - 1
    dx, dy = lx/float(nx), ly/float(ny)
    # list of boundary points, going anticlockwise
    boundpts = [(i*dx, 0.) for i in range(nx)] + \
               [(lx, j*dy) for j in range(ny)] + \
               [(lx - i*dx, ly) for i in range(nx)] + \
               [(0., ly - j*dy) for j in range(ny)]
    nbound = len(boundpts)
    # tell the triangulation that these points are on the boundary
    boundmarks = [1 for i in range(nbound)]
    # generate segments, anticlockwise
    boundsegs = [(i, i + 1) for i in range(nbound - 1)] + [(nbound - 1, 0)]

    # triangulate the domain
    mesh = Triangle()
    mesh.set_points(boundpts, markers=boundmarks)
    mesh.set_segments(boundsegs)
    mesh.triangulate(area=0.1)
    return mesh


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
    npoints = simple_one_cell_mesh.get_num_points()
    ncells = len(cells)
    assert ncells == 1
    fxx = fyy = numpy.ones(ncells, numpy.float64)
    fxy = numpy.zeros(ncells, numpy.float64)
    g = numpy.zeros(ncells, numpy.float64)
    s = numpy.zeros(npoints, numpy.float64)
    s[0] = 1.0
    s[1] = 2.0
    s[2] = 3.0
    problem = Ellipt2d(simple_one_cell_mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)
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
    # check the loading term
    print(f'source term: {problem.b}')
    assert abs(problem.b[0] - 7./24.) < EPS
    assert abs(problem.b[1] - 1./3.) < EPS
    assert abs(problem.b[2] - 3./8.) < EPS


def test_one_cell_problem2(simple_one_cell_mesh2):
    cells = simple_one_cell_mesh2.get_triangles()
    npoints = simple_one_cell_mesh2.get_num_points()
    ncells = len(cells)
    assert ncells == 1
    fxx = fyy = numpy.ones(ncells, numpy.float64)
    fxy = numpy.zeros(ncells, numpy.float64)
    g = numpy.zeros(ncells, numpy.float64)
    s = numpy.zeros(npoints, numpy.float64)
    s[0] = 1.0
    s[1] = 2.0
    s[2] = 3.0
    problem = Ellipt2d(simple_one_cell_mesh2, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)
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
    # check the loading term
    vertices = numpy.array([(n[0][0], n[0][1], 0.0) for n in simple_one_cell_mesh2.get_points()])
    jac = numpy.dot((0., 0., 1.), numpy.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    assert abs(problem.b[0] - jac*7./24.) < EPS
    assert abs(problem.b[1] - jac*1./3.) < EPS
    assert abs(problem.b[2] - jac*3./8.) < EPS


def test_laplacian(square_mesh):
    num_points = square_mesh.get_num_points()
    num_cells = square_mesh.get_num_triangles()
    fxx = fyy = numpy.ones(num_cells, numpy.float64)
    fxy = g = numpy.zeros(num_cells, numpy.float64)
    s = numpy.zeros(num_points, numpy.float64)
    # assemble the matrix problem
    equ = Ellipt2d(square_mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)
    nodes = square_mesh.get_points()
    # [(i, x, y), ...]
    boundaryNodes = [(i, nodes[i][0][0], nodes[i][0][1]) for i in range(len(nodes)) if nodes[i][1] == 1]
    # Dirichlet boundary conditions
    dbSouth = {n[0]: 0.0 for n in boundaryNodes if abs(n[2] - 0.) < 1.e-10}
    dbNorth = {n[0]: 1.0 for n in boundaryNodes if abs(n[2] - 1.) < 1.e-10}
    dbWest = {n[0]: 0.0 for n in boundaryNodes if abs(n[1] - 0.) < 1.e-10}
    dbEast = {n[0]: 0.0 for n in boundaryNodes if abs(n[1] - 1.) < 1.e-10}
    equ.setDirichletBoundaryConditions(dbSouth)
    equ.setDirichletBoundaryConditions(dbNorth)
    equ.setDirichletBoundaryConditions(dbWest)
    equ.setDirichletBoundaryConditions(dbEast)
    u = equ.solve()
    assert(abs(max(u) - 1.0) < 1.e-10)
    assert(abs(min(u) - 0.0) < 1.e-10)



def test_exact(square_mesh):
    # exact solution is x * 2*y
    num_points = square_mesh.get_num_points()
    num_cells = square_mesh.get_num_triangles()
    fxx = fyy = numpy.ones(num_cells, numpy.float64)
    fxy = g = numpy.zeros(num_cells, numpy.float64)
    s = numpy.zeros(num_points, numpy.float64)
    # assemble the matrix problem
    equ = Ellipt2d(square_mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)
    nodes = square_mesh.get_points()
    # [(i, x, y), ...]
    boundaryNodes = [(i, nodes[i][0][0], nodes[i][0][1]) for i in range(len(nodes)) if nodes[i][1] == 1]
    # Dirichlet boundary conditions
    # n[0] is the node index, n[1] is x at the node, n[2] is y at the node
    print(boundaryNodes)
    db = {n[0]: n[1] + 2*n[2] for n in boundaryNodes}
    equ.setDirichletBoundaryConditions(db)
    u = equ.solve()
    uexact = numpy.array([node[0][0] + 2*node[0][1] for node in square_mesh.get_points()])
    assert(numpy.fabs(u - uexact).sum() < 1.e-10)


def test_exact_with_s(square_mesh):
    # exact solution is x * 2*y
    nodes = square_mesh.get_points()
    x, y = numpy.array([n[0][0] for n in nodes]), numpy.array([n[0][1] for n in nodes])
    cells = square_mesh.get_triangles()

    num_points = square_mesh.get_num_points()
    num_cells = square_mesh.get_num_triangles()
    fxx = fyy = numpy.ones(num_cells, numpy.float64)
    g = numpy.ones(num_cells, numpy.float64)
    s = x + 2*y
    fxy = numpy.zeros(num_cells, numpy.float64)
    # assemble the matrix problem
    equ = Ellipt2d(square_mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)
    nodes = square_mesh.get_points()
    # [(i, x, y), ...]
    boundaryNodes = [(i, nodes[i][0][0], nodes[i][0][1]) for i in range(len(nodes)) if nodes[i][1] == 1]
    # Dirichlet boundary conditions
    # n[0] is the node index, n[1] is x at the node, n[2] is y at the node
    print(boundaryNodes)
    db = {n[0]: n[1] + 2*n[2] for n in boundaryNodes}
    equ.setDirichletBoundaryConditions(db)
    u = equ.solve()
    uexact = numpy.array([node[0][0] + 2*node[0][1] for node in square_mesh.get_points()])
    assert(numpy.fabs(u - uexact).sum() < 1.e-10)



