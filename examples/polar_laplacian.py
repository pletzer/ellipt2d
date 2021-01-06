"""
Solve polar Laplace equation
"""
import triangle
import ellipt2d
import numpy

#
# 1. create the mesh - (r, theta) coordinates
#

# low corner of the domain
xlo = numpy.array( (0., 0.) )

# x and y sizes of the domain
twopi = 2*numpy.pi
lx = 10.
ly = twopi

# number of x and y points
nx1, ny1 = 11, 11

# number of cells
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
mesh = triangle.Triangle()
mesh.set_points(boundpts, markers=boundmarks)
mesh.set_segments(boundsegs)
mesh.triangulate(area=0.5)

num_points = mesh.get_num_nodes()
num_cells = mesh.get_num_triangles()

nodes = mesh.get_nodes() # [[(x, y), marker], ...]
cells = mesh.get_triangles() # [[(i0, i1, i2), (), [a0, a1, ..]], ...]

#
# 2. create the elliptic solver
#

# cell arrays defining the PDE
# fxx = x, fyy = 1/x for the polar Laplacian
fxx = numpy.zeros(num_cells, numpy.float64)
fyy = numpy.zeros(num_cells, numpy.float64)
icell = 0
for icell in range(len(cells)):
    cell = cells[icell][0]
    i0, i1, i2 = cell
    x0, y0 = nodes[i0][0]
    x1, y1 = nodes[i1][0]
    x2, y2 = nodes[i2][0]
    # cell centre
    xm = (x0 + x1 + x2)/3.
    fxx[icell] = xm
    fyy[icell] = 1.0/xm

fxy = g = numpy.zeros(num_cells, numpy.float64)
s = numpy.zeros(num_points, numpy.float64)

# assemble the matrix problem
equ = ellipt2d.Ellipt2d(mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)

#
# 3. set the boundary conditions
#

# Dirichlet boundary conditions
boundaryNodes = [(i, nodes[i][0][0], nodes[i][0][1]) for i in range(len(nodes)) if nodes[i][1] == 1]
dbSouth = {n[0]: 0.0 for n in boundaryNodes if abs(n[2] - 0.) < 1.e-10}
dbNorth = {n[0]: 0.0 for n in boundaryNodes if abs(n[2] - ly) < 1.e-10}
dbWest = {n[0]: 0.0 for n in boundaryNodes if abs(n[1] - 0.) < 1.e-10}
dbEast = {n[0]: numpy.sin(n[2]) for n in boundaryNodes if abs(n[1] - lx) < 1.e-10}
equ.setDirichletBoundaryConditions(dbSouth)
equ.setDirichletBoundaryConditions(dbNorth)
equ.setDirichletBoundaryConditions(dbWest)
equ.setDirichletBoundaryConditions(dbEast)

#
# 4. solve and save the solution
#
u = equ.solve()
equ.saveVTK(filename='polar_laplacian.vtk', solution=u, sol_name='u')
