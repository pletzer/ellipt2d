"""
Solve Laplace equation in square domain
"""
import triangle
import ellipt2d
import numpy

# 1. create mesh
xlo = numpy.array( (0., 0.) )
lsize = numpy.array( (1., 1.) )
ex = numpy.array( (1., 0.) )
ey = numpy.array( (0., 1.) )

nx1, ny1 = 11, 11
nx, ny = nx1 - 1, ny1 - 1
boundpts = [tuple(xlo + i*ex/float(nx)) for i in range(nx)] + \
           [tuple(xlo + lsize[0]*ex + j*ey/float(ny)) for j in range(ny)] + \
           [tuple(xlo + lsize[0]*ex + lsize[1]*ey - i*ex/float(nx)) for i in range(nx)] + \
           [tuple(xlo + lsize[1]*ey - j*ey/float(ny)) for j in range(ny)]
nbound = len(boundpts)
boundmarks = [1 for i in range(nbound)]
boundsegs = [(i, i + 1) for i in range(nbound - 1)] + [(nbound - 1, 0)]

mesh = triangle.Triangle()
mesh.set_points(boundpts, markers=boundmarks)
mesh.set_segments(boundsegs)
mesh.triangulate(area=0.1)

num_points = mesh.get_num_nodes()
num_cells = mesh.get_num_triangles()

# 2. create elliptic solver
fxx = fyy = numpy.ones(num_cells, numpy.float64)
fxy = g = s = numpy.zeros(num_cells, numpy.float64)
equ = ellipt2d.Ellipt2d(mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)

# 3. set the boundary conditions
dbSouth = {i: 0. for i in range(0, nx)}
dbEast = {i: 0. for i in range(nx, nx + ny)}
dbNorth = {i: 1. for i in range(nx + ny, 2*nx + ny)}
dbWest = {i: 0. for i in range(2*nx + ny, 2*nx + 2*ny)}
db = {**dbSouth, **dbEast, **dbNorth, **dbWest}
equ.setDirichletBoundaryConditions(db)

# 4. solve
u = equ.solve()
equ.saveVTK(filename='laplace_square.vtk', solution=u, sol_name='u')
