"""
Solve Laplace equation in square domain
"""
import triangle
import ellipt2d
import numpy

#
# 1. create the mesh
#

# low corner of the domain
xlo = numpy.array( (0., 0.) )

# x and y sizes of the domain
lsize = numpy.array( (1., 1.) )

# unit vectors in x and y
ex = numpy.array( (1., 0.) )
ey = numpy.array( (0., 1.) )

# number of x and y points
nx1, ny1 = 11, 11

# number of cells
nx, ny = nx1 - 1, ny1 - 1

# list of boundary points, going anticlockwise
boundpts = [tuple(xlo + i*ex/float(nx)) for i in range(nx)] + \
           [tuple(xlo + lsize[0]*ex + j*ey/float(ny)) for j in range(ny)] + \
           [tuple(xlo + lsize[0]*ex + lsize[1]*ey - i*ex/float(nx)) for i in range(nx)] + \
           [tuple(xlo + lsize[1]*ey - j*ey/float(ny)) for j in range(ny)]

nbound = len(boundpts)

# tell the triangulation that these points are on the boundary
boundmarks = [1 for i in range(nbound)]

# generate segments, anticlockwise
boundsegs = [(i, i + 1) for i in range(nbound - 1)] + [(nbound - 1, 0)]

# triangulate the domain
mesh = triangle.Triangle()
mesh.set_points(boundpts, markers=boundmarks)
mesh.set_segments(boundsegs)
mesh.triangulate(area=0.1)

num_points = mesh.get_num_nodes()
num_cells = mesh.get_num_triangles()

#
# 2. create the elliptic solver
#

# cell arrays defining the PDE
fxx = fyy = numpy.ones(num_cells, numpy.float64)
fxy = g = s = numpy.zeros(num_cells, numpy.float64)

# assmeble the matrix problem
equ = ellipt2d.Ellipt2d(mesh, fxx=fxx, fxy=fxy, fyy=fyy, g=g, s=s)

#
# 3. set the boundary conditions
#

# zero everywhere but on the north side where we set the field to be one
db = {i: 0.0 for i in range(nbound)}
for i in range(nx + ny + 1, 2*nx + ny):
	db[i] = 1.0
# the solver expects a dictionary node_index => Dirichlet value
equ.setDirichletBoundaryConditions(db)

#
# 4. solve and save the solution
#
u = equ.solve()
equ.saveVTK(filename='laplace_square.vtk', solution=u, sol_name='u')
