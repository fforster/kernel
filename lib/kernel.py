import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#import scipy.linalg as scipylinalg

class kernel(object):

    # initiliaze given number of variables
    def __init__(self, nvar):

        self.nvar = nvar
        self.ivarf = np.loadtxt("ivar_%i.dat" % (self.nvar), dtype = int) - 1  # 9, 57, 81, 289, 625
        self.nf = np.shape(self.ivarf)[0]

        print("   Kernel initialized with filter size %i and %i free parameters" % (self.nf, self.nvar))

    # build vector of i and j position for given equation number
    def doieq(self, npsf):

        self.ieq2i = np.zeros(npsf * npsf, dtype = int)
        self.ieq2j = np.zeros(npsf * npsf, dtype = int)
        for i in range(npsf):
            for j in range(npsf):
                ieq = i * npsf + j
                self.ieq2i[ieq] = i
                self.ieq2j[ieq] = j

    # solve kernel given stamp dimensions, pairs of extended stars, always go from first to second stars
    def solve(self, npsf, pairs):

        # Number of stars
        nstars = len(pairs)

        # save stamp dimensions
        self.npsf = npsf
        
        # dimensions
        npsf2 = self.npsf * self.npsf
        nfh = int(self.nf / 2)

        # compute ieq2i and ie2j: array of filter i and j ordered by equation number
        self.doieq(self.npsf)

        # linear system:
        # sum_j=1..n X_ij beta_j = Y_i, i = 1..m, for n variables & m equations
        # exact solution of this system is beta = (X^T X)^-1 X^T Y
        # in this case there are nstars * npsf * npsf equations
        # and nvar variables
        X = np.zeros((nstars * npsf2, self.nvar)) # equation derivatives
        Y = np.zeros(nstars * npsf2) # rhs

        # iterate among stars
        for i, pair in enumerate(pairs):

            # iterate over filter coordinates to fill X and Y
            for k in range(self.nf):
                for l in range(self.nf):
                    # recover variable number
                    ivar = self.ivarf[k, l]
                    # if variable number is -1 ignore
                    if ivar == -1:
                        continue
                    # fill all equations corresponding to star i, for variable ivar (trick: use ieq2i and ieq2j)
                    X[i * npsf2: (i + 1) * npsf2, ivar] \
                        = X[i * npsf2: (i + 1) * npsf2, ivar] + pair[0][self.ieq2i + k, self.ieq2j + l]
            # fill rhs
            Y[i * npsf2: (i + 1) * npsf2] = pair[1][nfh: -(nfh + 1), nfh: -(nfh + 1)][self.ieq2i, self.ieq2j]

        # solve filter
        mat = np.dot(X.transpose(), X)
        rhs = np.dot(X.transpose(), Y)

        # L2
        #solvars = scipylinalg.solve(mat, rhs)
        solvars = np.linalg.solve(mat, rhs)
        
        # recover filter
        self.solfilter = np.zeros((self.nf, self.nf)) # initialize
        # loop among filter numbers
        for k in range(self.nf):
            for l in range(self.nf):
                ivar = self.ivarf[k, l]
                if ivar == -1:
                    continue
                self.solfilter[k, l] = solvars[ivar]

        return

    # do convolution of stamp of size (nf + npsf) x (nf + npsf), returns image of size npsf x npsf
    def convolve(self, stamp):

        # check dimensions
        nstamp, _ = np.shape(stamp)
        if nstamp != self.nf + self.npsf:
            return "WARNING: stamp of wrong dimensions"

        # initialize output stamp
        stampout = np.zeros((self.npsf, self.npsf))
        
        # iterate over filter coordinates to fill X and Y
        for k in range(self.nf):
            for l in range(self.nf):
                # fill all equations corresponding to star i, for variable ivar (trick: use ieq2i and ieq2j)
                stampout[self.ieq2i, self.ieq2j] = stampout[self.ieq2i, self.ieq2j] + stamp[self.ieq2i + k, self.ieq2j + l] * self.solfilter[k, l]

        return stampout
        
#
#    # compute filter characteristics
#    kratio = np.sum(solvars) / np.sum(np.abs(solvars))
#    ksupport = np.sum(solfilter * rfilter) / np.sum(solfilter)
#    knorm = np.sum(solfilter)
#    knorm2 = knorm * knorm
#    knormsum2 = np.sum(solfilter**2)
#
#    # save train stars and filter
#    if savestars:
#        fileout = "%s/%s/%s/CALIBRATIONS/stars_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
#        np.save(fileout, np.dstack([psf1s, psf2s]))
#        fileout = "%s/%s/%s/CALIBRATIONS/kernel_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
#        np.save(fileout, solfilter)
#        
#    return nclose, solfilter, kratio, ksupport, knorm2, knormsum2
