#!/usr/bin/env python3

# Problem 1 by Hansem Ro <hansem7@uw.edu>

import circuit_size_solver as solver
from decoder import *
import numpy as np

inputs = np.array(["A0", "A1", "A2", "A3", "Ax0", "Ax1", "Ax2", "Ax3"])
dec = decoder(inputs, topo="1")
