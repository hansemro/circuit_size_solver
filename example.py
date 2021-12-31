#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) Hansem Ro <hansemro@outlook.com>

import circuit_size_solver as solver
import numpy as np

# Testing cvxpy: Book example 4.15 (W&H)
# By: Hansem Ro <hansem7@uw.edu>
#
#                   _____
# A--------------->|NAND2|
# B--[Inv(x=1)]-+->|(x2) |---+
#               |  |_____|   |     _____
#               +->|NOR2 |   +--->|NOR3 |
# C--------------->|(x3) |------->|(x4) |----+----[Inv(x5)]----+
#                  |_____|   +--->|_____|    |                 |
#                            |              ===c4=10          ===c5=12
# D--------------------------+               |                 |
#                                            v                 v
#
# Expected solutions:
#  Minimum arrival time = 23.44
#  x2 = 1.62
#  x3 = 1.62
#  x4 = 3.37
#  x5 = 6.35

# Using logical_unit class
# Create inputs (as numpy array of strings)
inputs = np.array(["A", "B", "C", "D"])
# Create top level module
top = solver.circuit_module(inputs)
# Add units starting from input side
top.add_inv("B", "net_inv1", drive=1, name="inv1")
top.add_unit(np.array(["A", "net_inv1"]), "net_nand2", type="nand", drive="x2", name="nand2")
top.add_unit(np.array(["net_inv1", "C"]), "net_nor2", type="nor", drive="x3", name="nor2")
top.add_unit(np.array(["net_nand2", "net_nor2", "D"]), "net_nor3", type="nor", drive="x4", name="nor3")
top.add_cap("net_nor3", "net_c4", 10, name="c4")
top.add_inv("net_nor3", "net_inv2", drive="x5", name="inv2")
top.add_cap("net_inv2", "net_c5", 12, name="c5")
top.solve()
