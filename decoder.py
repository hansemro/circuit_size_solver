#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) Hansem Ro <hansemro@outlook.com>

import circuit_size_solver as solver
import numpy as np

# SI Prefix Dictionary
# from https://stackoverflow.com/a/10970888
_prefix = {
    'y': 1e-24,  # yocto
    'z': 1e-21,  # zepto
    'a': 1e-18,  # atto
    'f': 1e-15,  # femto
    'p': 1e-12,  # pico
    'n': 1e-9,   # nano
    'u': 1e-6,   # micro
    'm': 1e-3,   # mili
    'c': 1e-2,   # centi
    'd': 1e-1,   # deci
    'k': 1e3,    # kilo
    'M': 1e6,    # mega
    'G': 1e9,    # giga
    'T': 1e12,   # tera
    'P': 1e15,   # peta
    'E': 1e18,   # exa
    'Z': 1e21,   # zetta
    'Y': 1e24,   # yotta
}

# decoder class
#
#           ________
#  A[0] -->|Decoder |--> word0
# ~A[0] -->|(x)     |--> word1
#  A[1] -->|        |--> ...
# ~A[1] -->|        |--> word(2^N)
#  ...  -->|        |
#  A[N] -->|        |
# ~A[N] -->|________|
#
#
#   Generates various topologies:
#    - 1 : "NOR4"
#    - 2 : "NAND4-INV"
#    - 3 : "NAND2-NOR2"
#    - 4 : "INV-NAND4-INV"
#    - 5 : "NAND4-INV-INV-INV"
#    - 6 : "NAND2-NOR2-INV-INV"
class decoder:

    # Constructor:
    # @param inputs: numpy array with the decoder's inputs
    # @param depth: (default: 16)
    # @param width: (default: 32 (bits))
    # @param topo: (default: 1: "NOR4")
    def __init__(self, inputs: np.ndarray, depth=16, width=32, topo=1, inv_cap=1*_prefix.get("f"), output_load_cap=None):
        assert int(np.log2(depth)) == inputs.size/2
        self.inputs = inputs
        self.top = solver.top_module(inputs)
        self.depth = depth
        self.width = width
        self.topo = topo
        if output_load_cap is not None:
            self.output_load_cap = output_load_cap
        else:
            self.output_load_cap = width * 4*_prefix.get("f")
        if topo == 2:
            self.topo_detailed = "NAND4-INV"
        elif topo == 3:
            self.topo_detailed = "NAND2-NOR2"
        elif topo == 4:
            self.topo_detailed = "INV-NAND4-INV"
        elif topo == 5:
            self.topo_detailed = "NAND4-INV-INV-INV"
        elif topo == 6:
            self.topo_detailed = "NAND2-NOR2-INV-INV"
        else:
            self.topo_detailed = "NOR4"
        print("topo: ", self.topo_detailed)
        # construct depth number of words as logical units
        for i in range(0,depth):
            sub_inputs = self.__gen_pattern(i)
            sub_output = "word" + str(i)
            tmp_word = word(sub_inputs, sub_output, width=width, topo=topo)
    
    def __gen_pattern(self, num):
        assert num >= 0 and self.depth > num
        ret = np.array([])
        tmp = num
        for i in range(0,int(np.log2(self.depth))):
            bit = tmp % 2
            tmp = tmp / 2
            if int(bit) == 1:
                ret = np.append(ret, self.inputs[i])
            else:
                ret = np.append(ret, self.inputs[i + int(np.log2(self.depth))])
        return ret

# word class:
#
# Example: word9 = (A[3:0] == 3'b1001)
#            _________
# A[0] ---> |word_dec |---+ word9
#~A[1] ---> |width(32)|   |
#~A[2] ---> |         |  ===c_word=4f*width
# A[3] ---> |_________|   |
#                         v
#
# word_dec logic: A[0]&~A[1]&~A[2]&A[3] = AND4(A[0], ~A[1], ~A[2], A[3])
#
# Topologies for word9:
#
# NOR4                  : ~(~A[0]+A[1]+A[2]+~A[3]) = NOR4(word9, ~A[0], A[1], A[2], ~A[3])
# NAND4-INV             : ~~(A[0]&~A[1]&~A[2]&A[3]) = NAND4(net_nand4, A[0], ~A[1], ~A[2], A[3]); INV(word9, net_nand4)
# NAND2-NOR2            : ~(~(A[0]&~A[1])+~(~A[2]&A[3])) = NAND2(net_nand2_1, A[0], ~A[1]); NAND2(net_nand2_2, ~A[2], A[3]); NOR2(word9, net_nand2_1, net_nand2_2)
# INV-NAND4-INV         : INV(B[3:0], {~A[3], A[2], A[1], ~A[0]}); NAND4(net_nand4, B); INV(word9, net_nand4)
# NAND4-INV-INV-INV     : ~~~~(A[0]&~A[1]&~A[2]&A[3]) = NAND4(net_nand4, A[0], ~A[1], ~A[2], A[3]); INV(net_inv_1, net_nand4); INV(net_inv_2, net_inv_1); INV(word9, net_inv_2);
# NAND2-NOR2-INV-INV    :
class word(solver.logical_unit):

    # Constructor:
    def __init__(self, inputs: np.ndarray, output, width, topo, inv_cap=1*_prefix.get("f"), output_load_cap=None):
        self.inputs = inputs
        self.width = width
        self.type = "atomic"
        self.type = "word_block"
        self.inv_cap = inv_cap
