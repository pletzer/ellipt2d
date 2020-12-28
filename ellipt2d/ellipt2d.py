import numpy


class Ellipt2d(object):
    """Elliptic solver class -div f grad v + g v = s"""

    def __init__(self, grid, f, g, s):
        """Constructor
        :param grid: instance of pytriangle
        :param f: cell array
        :param g: cell array
        :param s: nodal array
        """
        self.grid = grid

        nodes = grid.get_nodes()

        self.amat = {}
        self.b = numpy.zeros((len(nodes), ), numpy.float64)

        # build the matrix system

        oneSixth = 1./6.
        oneTwelveth = 1/12.
        massMat = numpy.array([[oneSixth, oneTwelveth, oneTwelveth],
            	               [oneTwelveth, oneSixth, oneTwelveth],[
            	               [oneTwelveth, oneTwelveth, oneSixth]]])

        sourceVec = numpy.array([oneSixth, oneSixth, oneSixth])

        icell = 0
        for cell in grid.get_triangle():

        	# get the cell values
        	fcell = f[icell]
        	gcell = g[icell]
        	scell = s[icell]

            # get the node indices
            i0, i1, i2 = cell[:3]

            indexMat = numpy.array([[(i0, i0), (i0, i1), (i0, i2)],
            	                    [(i1, i0), (i1, i1), (i1, i2)],
            	                    [(i2, i0), (i2, i1), (i2, i2)]])

            # get the coordinates of each node
            x0, y0 = nodes[i0][:2]
            x1, y1 = nodes[i1][:2]
            x2, y2 = nodes[i2][:2]

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
            stiffness = (fcell/twoJac) * dmat.transpose().dot(dmat)

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



    def applyFluxBoundaryConditions(self, alpha):
        """Apply flux boundary conditions
        :param alpha: flux 
        """
        pass


    def solve(self):
        pass


    def saveVTK(self, filename):
        pass

