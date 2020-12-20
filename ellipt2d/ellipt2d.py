import numpy


class Ellipt2d(object):
    """Elliptic solver class -div f grad v + g v = s"""

    def __init__(self, grid, fxx, fyy, g, s):
        """Constructor
        :param grid: instance of pytriangle
        :param fxx: cell array
        :param fyy: cell array
        :param g: cell array
        :param s: nodal array
        """
        self.grid = grid

        nodes = grid.get_nodes()

        self.amat = {}
        self.b = numpy.zeros((len(nodes), ), numpy.float64)

        # build the matrix system

        icell = 0
        for cell in grid.get_triangle():

            # get the node indices
            ia, ib, ic = cell[:3]

            # get the coordinates of each node
            xa, ya = nodes[ia][:2]
            xb, yb = nodes[ib][:2]
            xc, yc = nodes[ic][:2]

            xba = xb - xa
            xca = xc - xa
            yba = yb - ya
            yca = yc - ya

            xcb = xc - xb
            ycb = yc - yb

            # twice the area of the cell
            jac = xba*yca - xca*yba

            # source term
            sa, sb, sc = s[ia], s[ib], s[ic]
            self.b[ia] += jac*(sa/12. + sb/24. + sc/24.)
            self.b[ib] += jac*(sb/12. + sc/24. + sa/24.)
            self.b[ic] += jac*(sc/12. + sa/24. + sb/24.)


            # g term
            gcell = g[icell]
            self.amat[ia, ia] = self.amat.get((ia, ia), 0.) + gcell*jac/12.
            self.amat[ia, ib] = self.amat.get((ia, ib), 0.) + gcell*jac/24.
            self.amat[ia, ic] = self.amat.get((ia, ic), 0.) + gcell*jac/24.
            self.amat[ib, ib] = self.amat.get((ib, ib), 0.) + gcell*jac/12.
            self.amat[ib, ic] = self.amat.get((ib, ic), 0.) + gcell*jac/24.
            self.amat[ic, ic] = self.amat.get((ic, ic), 0.) + gcell*jac/12.

            # f terms
            fxxcell = fxx[icell]
            fyycell = fyy[icell]
            self.amat[ia, ia] += 0.5 * ( fxxcell*ycb*ycb + fyycell*xcb*xcb ) / jac
            self.amat[ia, ib] += 0.5 * (-fxxcell*yca*ycb + fyycell*xca*xcb ) / jac
            self.amat[ia, ic] += 0.5 * ( fxxcell*yba*ycb + fyycell*xba*xcb ) / jac

            self.amat[ib, ib] += 0.5 * ( fxxcell*yca*yca + fyycell*xca*xca ) / jac
            self.amat[ib, ic] += 0.5 * (-fxxcell*yca*yba - fyycell*xca*xba ) / jac

            self.amat[ic, ic] += 0.5 * ( fxxcell*yba*yba + fyycell*xba*xba ) / jac


            # symetric terms
            self.amat[ib, ia] = self.amat[ia, ib]
            self.amat[ic, ia] = self.amat[ia, ic]
            self.amat[ic, ib] = self.amat[ib, ic]

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

