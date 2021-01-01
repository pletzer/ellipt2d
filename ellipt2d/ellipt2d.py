import numpy
from scipy.sparse import csc_matrix, linalg


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



    def applyFluxBoundaryConditions(self, alpha):
        """Apply flux boundary conditions
        :param alpha: flux 
        """
        pass


    def solve(self):
        """Solve system
        :returns solution vector
        """
        indices, data = zip(*self.amat.items())
        i_inds = [index[0] for index in indices]
        j_inds = [index[1] for index in indices]
        spmat = csc_matrix( (data, (i_inds, j_inds)), shape=(self.nsize, self.nsize))
        lumat = linalg.splu(spmat)
        return lumat.dot(self.b)



    def saveVTK(self, filename):
        pass

