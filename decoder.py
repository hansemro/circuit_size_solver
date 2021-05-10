#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) Hansem Ro <hansemro@outlook.com>

import circuit_size_solver as solver
from circuit_size_solver import logical_unit
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
    # @param depth: (default: 16) (other dimensions are currently unsupported)
    # @param width: (default: 32 (bits)) (other dimensions are currently unsupported)
    # @param topo: (default: 1: "NOR4") (topologies for 16x32b decoder only)
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
            word_inputs = self.__gen_pattern(i, inv=inv)
            print(word_inputs)
            word_output = "word" + str(i)
            word_name = "word_block" + str(i)
            word_wire_cap = self.__estimate_wire_cap(i)
            tmp_word = word16_32b(word_inputs, word_output, width=width, topo=topo, name=word_name, inv_cap=inv_cap, output_load_cap=self.output_load_cap)
            tmp_word.print_props()

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
        return wire_cap_per_um * length * self.depth

# word16 class:
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
class word16_32b(solver.top_module):

    # Constructor:
    def __init__(self, inputs: np.ndarray, output: str, width, topo, name: str, inv_cap=1*_prefix.get("f"), output_load_cap=None):
        assert topo is not None
        self.topo = topo
        self.width = width
        self.inv_cap = inv_cap
        self.g = solver._G(self.__get_g_arr_from_topo())
        self.p = solver._P(self.__get_p_arr_from_topo())
        super().__init__(inputs, output, type="atomic")
        self.name = name
        self.type_detailed = "word_block"
        self.add_units()
    
    def add_units(self):
        if (self.topo == "2"):
            nand4_name = self.name + "nand4"
            nand4_net = "net_" + nand4_name
            self.add_unit(self.inputs, nand4_net, "nand", name=nand4_name)
            self.add_inv(nand4_net, self.output, name=self.name + "inv")
        elif (self.topo == "3"):
            nand2_1_name = self.name + "nand2_1"
            nand2_2_name = self.name + "nand2_2"
            nand2_net1 = "net_" + nand2_1_name
            nand2_net2 = "net_" + nand2_2_name
            self.add_unit(self.inputs[0:1], nand2_net1, "nand", name=nand2_1_name)
            self.add_unit(self.inputs[2:3], nand2_net2, "nand", name=nand2_2_name)
            self.add_unit(np.array([nand2_net1, nand2_net2]), self.output, "nor", name=self.name + "nor2")
        elif (self.topo == "4"):
            inv_1_name = self.name + "inv_1"
            inv_2_name = self.name + "inv_2"
            inv_3_name = self.name + "inv_3"
            inv_4_name = self.name + "inv_4"
            inv_5_name = self.name + "inv_5"
            inv_1_net = "net_" + inv_1_name
            inv_2_net = "net_" + inv_2_name
            inv_3_net = "net_" + inv_3_name
            inv_4_net = "net_" + inv_4_name
            nand4_name = self.name + "nand4"
            nand4_net = "net_" + nand4_name
            self.add_inv(self.inputs[0], inv_net1, inv_1_name)
            self.add_unit(np.array([inv_1_net, inv_2_net, inv_3_net, inv_4_net]), nand4_net, "nand", name=nand4_name)
            self.add_inv(nand4_net, self.output, name=inv_5_name)
        elif (self.topo == "5"):
            nand4_name = self.name + "nand4"
            nand4_net = "net_" + nand4_name
            inv_1_name = self.name + "inv_1"
            inv_2_name = self.name + "inv_2"
            inv_3_name = self.name + "inv_3"
            inv_1_net = "net_" + inv_1_name
            inv_2_net = "net_" + inv_2_name
            self.add_unit(self.inputs, nand4_net, "nand", name=nand4_name)
            self.add_inv(nand4_net, inv_1_net, name=inv_1_name)
            self.add_inv(inv_1_net, inv_2_net, name=inv_2_name)
            self.add_inv(inv_2_net, self.output, name=inv_3_name)
        elif (self.topo == "6"):
            nand2_1_name = self.name + "nand2_1"
            nand2_2_name = self.name + "nand2_2"
            nand2_net1 = "net_" + nand2_1_name
            nand2_net2 = "net_" + nand2_2_name
            nor2_name = self.name + "nor2"
            nor2_net = "net_" + nor2_name
            inv_1_name = self.name + "inv_1"
            inv_2_name = self.name + "inv_2"
            inv_3_name = self.name + "inv_3"
            inv_1_net = "net_" + inv_1_name
            inv_2_net = "net_" + inv_2_name
            self.add_unit(self.inputs[0:1], nand2_net1, "nand", name=nand2_1_name)
            self.add_unit(self.inputs[2:3], nand2_net2, "nand", name=nand2_2_name)
            self.add_unit(np.array([nand2_net1, nand2_net2]), nor2_net, "nor", name=nor2_name)
            self.add_inv(nor2_net, inv_1_net, name=inv_1_name)
            self.add_inv(inv_1_net, inv_2_net, name=inv_2_name)
            self.add_inv(inv_2_net, self.output, name=inv_3_name)
        else:
            self.add_unit(self.inputs, self.output, "nor", name=self.name + "nor4")

    def __get_g_arr_from_topo(self):
        assert self.topo is not None
        ret = np.array([])
        if (self.topo == "2"):
            ret = np.append(ret, solver.g_nand(4))
            ret = np.append(ret, solver.g_inv())
        elif (self.topo == "3"):
            ret = np.append(ret, solver.g_nand(2))
            ret = np.append(ret, solver.g_nand(2))
            ret = np.append(ret, solver.g_nor(2))
        elif (self.topo == "4"):
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_nand(4))
            ret = np.append(ret, solver.g_inv())
        elif (self.topo == "5"):
            ret = np.append(ret, solver.g_nand(4))
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
        elif (self.topo == "6"):
            ret = np.append(ret, solver.g_nand(2))
            ret = np.append(ret, solver.g_nand(2))
            ret = np.append(ret, solver.g_nor(2))
            ret = np.append(ret, solver.g_inv())
            ret = np.append(ret, solver.g_inv())
        else:
            ret = np.append(ret, solver.g_nor(4))
        return ret

    def __get_p_arr_from_topo(self):
        assert self.topo is not None
        ret = np.array([])
        if (self.topo == "2"):
            ret = np.append(ret, solver.p_nand(4))
            ret = np.append(ret, solver.p_inv())
        elif (self.topo == "3"):
            ret = np.append(ret, solver.p_nand(2))
            ret = np.append(ret, solver.p_nand(2))
            ret = np.append(ret, solver.p_nor(2))
        elif (self.topo == "4"):
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_nand(4))
            ret = np.append(ret, solver.p_inv())
        elif (self.topo == "5"):
            ret = np.append(ret, solver.p_nand(4))
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
        elif (self.topo == "6"):
            ret = np.append(ret, solver.p_nand(2))
            ret = np.append(ret, solver.p_nand(2))
            ret = np.append(ret, solver.p_nor(2))
            ret = np.append(ret, solver.p_inv())
            ret = np.append(ret, solver.p_inv())
        else:
            ret = np.append(ret, solver.p_nor(4))
        return ret
    
    # add_unit: Add a unit with specified inputs to a new output net.
    # @param inputs: str numpy array of input net names 
    # @param output: name of output net
    # @param type: type of logic unit
    # @param Cin: (optional) unit capacitance
    # @param drive: (optional) unit drive number or string alias
    # @param name: (optional) unit name
    def add_unit(self, inputs: np.ndarray, output: str, type, Cin=None, drive=None, name=None):
        # Check if all inputs exist
        for input in inputs:
            assert input in self.nets
        # Check if output is a new net
        assert output not in self.nets
        self.nets.add(output)
        tmp = logical_unit(inputs, output, type, Cin=Cin, drive=drive, name=name)
        self.nodes[output].append(tmp)
    
    def add_inv(self, input: str, output: str, drive=None, name=None):
        self.add_unit(np.array([input]), output, "inv", drive=drive, name=name)
    
    def add_cap(self, input: str, output: str, Cin, name=None):
        self.add_unit(np.array([input]), output, "cap", Cin=Cin, name=name)

