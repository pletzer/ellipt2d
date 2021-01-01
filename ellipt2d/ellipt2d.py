import numpy
from scipy.sparse import csc_matrix, linalg
import time


class Ellipt2d(object):
    """Elliptic solver class -div F grad v + g v = s with F a 2x2 tensor function
       and g, s scalar functions
    """

    def __init__(self, grid, fxx, fxy, fyy, g, s):
        """Constructor
        :param grid: instance of pytriangle
        :param fxx: cell array
        :param fxy: cell array
        :param fyy: cell array
        :param g: cell array
        :param s: cell array
        """
        self.grid = grid

        # node: (x, y)
        self.node = {}

        nodes = grid.get_nodes()

        self.amat = {}
        self.b = numpy.zeros((len(nodes), ), numpy.float64)

        # build the matrix system

        oneSixth = 1./6.
        oneTwelveth = 1/12.
        massMat = oneTwelveth * numpy.ones((3, 3), numpy.float64)
        for j in range(3):
            massMat[j, j] = oneSixth

        sourceVec = numpy.array([oneSixth, oneSixth, oneSixth])

        indexMat = numpy.zeros((3, 3, 2), numpy.int32)

        self.nsize = -1
        icell = 0
        for cell in grid.get_triangles():

            # get the cell values
            fxxcell = fxx[icell]
            fxycell = fxy[icell]
            fyycell = fyy[icell]
            fmat = numpy.array([[fxxcell, fxycell], [fxycell, fyycell]])
            gcell = g[icell]
            scell = s[icell]

            # get the node indices
            i0, i1, i2 = cell[:3][0]
            self.nsize = max(self.nsize, i0, i1, i2)

            indexMat[0, 0, :] = i0, i0
            indexMat[0, 1, :] = i0, i1
            indexMat[0, 2, :] = i0, i2
            indexMat[1, 0, :] = i1, i0
            indexMat[1, 1, :] = i1, i1
            indexMat[1, 2, :] = i1, i2
            indexMat[2, 0, :] = i2, i0
            indexMat[2, 1, :] = i2, i1
            indexMat[2, 2, :] = i2, i2

            # get the coordinates for each vertex
            x0, y0 = nodes[i0][0][:2]
            x1, y1 = nodes[i1][0][:2]
            x2, y2 = nodes[i2][0][:2]
            self.node[i0] = numpy.array((x0, y0))
            self.node[i1] = numpy.array((x1, y1))
            self.node[i2] = numpy.array((x2, y2))

            y12 = y1 - y2
            y20 = y2 - y0
            y01 = y0 - y1
            x12 = x1 - x2
            x20 = x2 - x0
            x01 = x0 - x1

            # twice the area of the cell
            jac = -x01*y20 + x20*y01
            twoJac = 2.*jac
            halfJac = 0.5*jac # triangle area

            # stiffness matrix
            # https://www.math.tu-berlin.de/fileadmin/i26_ng-schmidt/Vorlesungen/IntroductionFEM_SS14/Chap3.pdf
            dmat = numpy.array([[y12, y20, y01], [x12, x20, x01]])
            stiffness = dmat.transpose().dot(fmat).dot(dmat) / twoJac

            # mass term
            mass = halfJac*gcell*massMat

            # source term
            source = jac*scell*sourceVec

            for k0 in range(3):
                self.b[i0] = source[k0]
                for k1 in range(k0, 3):
                    j0, j1 = indexMat[k0, k1]
                    self.amat[j0, j1] = self.amat.get((j0, j1), 0.) + \
                         stiffness[k0, k1] + mass[k0, k1]
                    # symmetric term
                    self.amat[j1, j0] = self.amat[j0, j1]

            icell += 1

        self.nsize += 1


    def getCoords(self, i):
        """
        Get the coordinates of node
        :param i: node index
        :returns (x, y) values
        """
        return self.node[i]


    def setNaturalBoundaryConditions(self, values):
        """Apply natural boundary conditions n . F grad v = alpha - beta v
        :param values: dictionary {(i, j): (alpha, beta), ...} with i, j node indices. The order
                       should counterclockwise for external boundary edges and clockwise for 
                       internal boundary edges
        """
        oneSixth = 1./6.
        oneThird = 1./3.
        for ij, ab in values.items():

            i, j = ij
            aedge, bedge = ab

            # coordinates of the two nodes connecting the edge
            pi = self.getCoords(i)
            pj = self.getCoords(j)

            # length of edge
            ds = numpy.sqrt( (pj - pi).dot(pj - pi) )

            # contribution to the source
            aval = aedge*ds*0.5
            self.b[i] += aval
            self.b[j] += aval

            # contribution to the stiffness matrix
            bval = bedge*ds*oneSixth
            self.amat[i, j] += bval
            self.amat[j, i] += bval
            aval = bedge*ds*oneThird
            self.amat[i, i] += bval
            self.amat[j, j] += bval


    def setDirichletBoundaryConditions(self, values):
        """Apply Dirichlet boundary conditions, forcing the solution to take prescribed values at nodes
        :param values: dictionary {i: value, ...} with i node indices. Index i could be an internal node.
        """
        LARGE = 1.656746e15
        for i, value in values.items():
            # in principle we should be setting the row and one and zeros but this requires extracting the 
            # connectivity of node to neighboring nodes
            self.amat[i, i] = LARGE
            self.b[i] = LARGE*value


    def solve(self):
        """Solve system
        :returns solution vector
        """
        indices, data = zip(*self.amat.items())
        i_inds = [index[0] for index in indices]
        j_inds = [index[1] for index in indices]
        spmat = csc_matrix( (data, (i_inds, j_inds)), shape=(self.nsize, self.nsize))
        lumat = linalg.splu(spmat)
        return lumat.solve(self.b)



    def saveVTK(self, filename, solution, sol_name='u'):
        """Save the solution to a VTK file
        :param filename: file name
        :param solution: solution vector
        :param sol_name: solution name
        """
        with open(filename, 'w') as f:
            date = time.ctime( time.time() )
            nnodes = self.grid.get_num_nodes()
            ncells = self.grid.get_num_triangles()
            f.write('# vtk DataFile Version 2.0\n')
            f.write(f'produced by Ellipt2d on {date}\n')
            f.write('ASCII\n')
            f.write('DATASET UNSTRUCTURED_GRID\n')
            f.write(f'POINTS {nnodes} DOUBLES\n')
            for node in self.grid.get_nodes():
                x, y = node[0][:2]
                f.write(f'{x} {y} 0.0\n')
            f.write(f'CELLS {ncells} {ncells*4}\n')
            for cell in self.grid.get_triangles():
                i0, i1, i2 = cell[:3][0]
                f.write(f'3 {i0} {i1} {i2}\n')
            f.write(f'CELL_TYPES {ncells}\n')
            for i in range(ncells):
                f.write('5\n')
            f.write(f'POINT_DATA {nnodes}\n')
            f.write(f'SCALARS {sol_name} double\n')
            f.write('LOOKUP_TABLE default\n')
            for i in range(nnodes):
                f.write(f'{solution[i]:10.8e}\n')




