import numpy as np

import porepy as pp
import pygeon as pg

from main import LocalSolver
from time import perf_counter
from datetime import datetime

class Clock(object):
    """Class for measuring (computational) time intervals. Objects of this class have the following attributes:

    tstart (float): time at which the clock was started (in seconds).
    tstop  (float): time at which the clock was stopped (in seconds).
    """

    def __init__(self):
        """Creates a new clock."""
        self.tstart = 0
        self.tstop = 0

    def start(self):
        """Starts the clock."""
        self.tstart = perf_counter()

    def stop(self):
        """Stops the clock."""
        self.tstop = perf_counter()

    def elapsed(self):
        """Returns the elapsed time between the calls .start() and .stop()."""
        dt = self.tstop-self.tstart

        if(dt<0):
            raise RuntimeError("Clock still running.")
        else:
            return dt

    def elapsedTime(self):
        """Analogous to .elapsed() but returns the output in string format."""
        return self.parse(self.elapsed())

    def now(self):
        """Returns the current date and time."""
        c = str(datetime.now())
        date = c[:c.find(" ")]
        c = c[c.find(" ")+1:]
        c = c[:c.find(".")]
        return "%s (%s)" % (c, date)

    def parse(self, time):
        """Converts an amount of seconds in a string of the form '# hours #minutes #seconds'."""
        h = time//3600
        m = (time-3600*h)//60
        s = time-3600*h-60*m

        if(h>0):
            return ("%d hours %d minutes %.2f seconds" % (h,m,s))
        elif(m>0):
            return ("%d minutes %.2f seconds" % (m,s))
        else:
            return ("%.2f seconds" % s)

    def shortparse(self, time):
        """Analogous to Clock.parse but uses the format '#h #m #s'."""
        h = time//3600
        m = (time-3600*h)//60
        s = time-3600*h-60*m

        if(h>0):
            return ("%d h %d m %.2f s" % (h,m,s))
        elif(m>0):
            return ("%d m %.2f s" % (m,s))
        else:
            return ("%.2f s" % s)


if __name__ == "__main__":
    # NOTE: difficulty to converge for RBM
    folder = "examples/case1/"
    step_size = 0.05
    keyword = "elasticity"
    tol = 1e-8
    nsamples = 200

    dim = 2
    sd = pg.unit_grid(dim, step_size, as_mdg=False)
    sd.compute_geometry()

    def sample(seed):
        np.random.seed(seed)
        mu, lambd = 0.1 + 1.9*np.random.rand(2)
        force = (1e-3)*(0.5 + 1.5*np.random.rand())
        body_force = -1e-2*(0.5 + 1.5*np.random.rand())
    
        data = {pp.PARAMETERS: {keyword: {"mu": mu, "lambda": lambd}}}
        solver = LocalSolver(sd, data, keyword, False, body_force, force)

        # step 1
        sf = solver.compute_sf()

        # step 2
        s0 = solver.compute_s0_cg(sf, tol=tol, verbose=False)

        # step 3
        s, u, r = solver.compute_all(s0, sf)
        return [mu, lambd, force, body_force], s0, s, u, r, solver

    params, s0, s, u, r = [], [], [], [], []

    clock = Clock()
    for j in range(nsamples):
        if(j==0):
            clock.start()
        pj, s0j, sj, uj, rj, solver = sample(j)
        if(j==0):
            clock.stop()
            raw_eta = clock.elapsed()*(nsamples-1)
            eta = clock.shortparse(raw_eta)
            string = str(datetime.fromtimestamp(datetime.now().timestamp()+raw_eta))[:-7]
            date, hour = string.split(" ")
            when = hour + " of " + date[-2:] + date[-6:-2] + date[:4]
            print(">> Generating %d samples.\n   ETA: %s\n   Expected to finish at: %s." % (nsamples, eta, when))
        params.append(pj)
        s0.append(s0j)
        s.append(sj)
        u.append(uj)
        r.append(rj)
        
    output = dict()
    output['mass'] = solver.D
    output['B'] = solver.B
    output['params'] = np.stack(params)
    output['s0'] = np.stack(s0)
    output['s'] = np.stack(s)
    output['u'] = np.stack(u)
    output['r'] = np.stack(r)

    np.savez("snapshots-case1.npz", **output)
    print("\n>> Done.")