#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) Hansem Ro <hansemro@outlook.com>

import circuit_size_solver as solver
import numpy as np

# EE476 Final 2021 Autumn Example
#
# Note: In the original problem, inverter with drive 'x' is connected to a
# x=5f load inverter with and a 10f capacitor. To adjust the problem to work
# with the solver, the load cap and inv were combined.
#
# A--[Inv(x=1f)]--[Inv(x='w')]--+--[Inv(x='x')]--+
#                               |                |
#                              ===c_wire=10f    ===c_load=15f
#                               |                |
#                               v                v
#
# Expected solutions:
#  drive('w') = 4.24
#  drive('x') = 7.97

# Using logical_unit class
# Create inputs (as numpy array of strings)
inputs = np.array(["A"])
# Create top level module
top = solver.circuit_module(inputs)
# Add units starting from input side
top.add_inv(input="A", output="net1", Cin=1e-15, drive=1e-15, name="inv1")
top.add_inv(input="net1", output="net2", drive="w", name="inv2")
top.add_cap(input="net2", output="net_cap_wire", Cin=10e-15, name="Cwire")
top.add_inv(input="net2", output="net3", drive="x", name="inv3")
top.add_cap(input="net3", output="net_cap_load", Cin=15e-15, name="Cload")
top.solve()
