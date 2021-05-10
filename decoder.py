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
    def __init__(self, inputs: np.ndarray, depth=16, width=32, topo=1, inv_cap=1*_prefix.get("f"), output_load_cap_factor=4*_prefix.get("f")):
        assert int(np.log2(depth)) == inputs.size/2
        self.inputs = inputs
        self.top = solver.top_module(inputs)
        self.depth = depth
        self.width = width
        self.topo = topo
        inv = False
        self.output_load_cap = width * output_load_cap_factor
        if topo == 2:
            self.topo_detailed = "NAND4-INV"
        elif topo == 3:
            self.topo_detailed = "NAND2-NOR2"
        elif topo == 4:
            self.topo_detailed = "INV-NAND4-INV"
            inv = True
        elif topo == 5:
            self.topo_detailed = "NAND4-INV-INV-INV"
        elif topo == 6:
            self.topo_detailed = "NAND2-NOR2-INV-INV"
        else:
            self.topo_detailed = "NOR4"
            inv = True
        print("topo: ", self.topo_detailed)
        # construct depth number of words as logical units
        for i in range(0,depth):
            sub_inputs = self.__gen_pattern(i, inv=inv)
            print(sub_inputs)
            sub_output = "word" + str(i)
            tmp_wire_cap = self.__estimate_wire_cap(i)
            tmp_word = word(sub_inputs, sub_output, width=width, topo=topo, inv_cap=inv_cap, output_load_cap=self.output_load_cap)
            
    # __gen_pattern: generate an array of input patterns for given word number.
    # Setting inv to True will invert the result.
    # Assumptions: inputs are given in the following format:
    #       inputs = np.array(["A[0]", "A[1]", ..., "Ax[0]", "Ax[1]", ...])
    #       where A and Ax are complimentary
    # @param word_num: word number
    # @param inv: invert array result
    def __gen_pattern(self, word_num, inv=False):
        assert word_num >= 0 and self.depth > word_num
        ret = np.array([])
        tmp = word_num
        for i in range(0,int(np.log2(self.depth))):
            bit = tmp % 2
            tmp = tmp / 2
            if (inv and int(bit) == 0) or (not inv and int(bit) == 1):
                ret = np.append(ret, self.inputs[i])
            else:
                ret = np.append(ret, self.inputs[i + int(np.log2(self.depth))])
        return ret
    
    # __estimate_wire_cap: returns an estimated wire cap for given word number.
    # Assumptions:
    #   - 0.2fF/um of wire
    #   - word decoder has vertical pitch of 3.6um
    # @param word_num: word number
    # @param wire_cap_per_um: (default: 0.2f) capacitance per um of wire
    # @param word_ver_pitch: (default: 3.6u) vertical pitch of 1 word
    def __estimate_wire_cap(self, word_num, wire_cap_per_um = 0.2 * _prefix.get("f"), word_ver_pitch=3.6 * _prefix.get("u")):
        assert word_num >= 0
        
        length = (word_num + 1) * word_ver_pitch
        return wire_cap_per_um * length

# word class:
#
# Example: word9 = (A[3:0] == 3'b1001)
#            __________
# A[0] ---> |word_block|---+ word9
#~A[1] ---> |width(32) |   |
#~A[2] ---> |g, x      |  ===c_word=4f*width
# A[3] ---> |__________|   |
#                          v
#
# word_dec logic: A[0]&~A[1]&~A[2]&A[3] = AND4(A[0], ~A[1], ~A[2], A[3])
#
# Topologies for word9:
#
# NOR4                  : ~(~A[0]+A[1]+A[2]+~A[3]) = NOR4(word9, ~A[0], A[1], A[2], ~A[3])
# NAND4-INV             : ~~(A[0]&~A[1]&~A[2]&A[3]) = NAND4(net_nand4, A[0], ~A[1], ~A[2], A[3]); INV(word9, net_nand4)
# NAND2-NOR2            : ~(~(A[0]&~A[1])+~(~A[2]&A[3])) = NAND2(net_nand2_1, A[0], ~A[1]); NAND2(net_nand2_2, ~A[2], A[3]); NOR2(word9, net_nand2_1, net_nand2_2)
# INV-NAND4-INV         : INV(B[3:0], {~A[3], A[2], A[1], ~A[0]}); NAND4(net_nand4, B); INV(word9, net_nand4)
# NAND4-INV-INV-INV     : Same as NAND4-INV, but with two additional inverters at the end
# NAND2-NOR2-INV-INV    : Same as NAND2-NOR2, but with two additional inverters at the end
class word(solver.logical_unit):

    # Constructor:
    def __init__(self, inputs: np.ndarray, output, width, topo, inv_cap=1*_prefix.get("f"), output_load_cap=None):
        self.inputs = inputs
        self.width = width
        self.type = "atomic"
        self.type = "word_block"
        self.inv_cap = inv_cap