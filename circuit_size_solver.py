#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) Hansem Ro <hansemro@outlook.com>

# Install numpy and cvxpy
# pip3 install numpy cvxpy
import numpy as np
import cvxpy as cp
from collections import defaultdict

# circuit_size_solver: Circuit sizing solver
#
#   General principles:
#       1) units can only be added if its input nets exist.
#       2) every unit has only 1 output
#       3) no floating nets allowed, so add a capacitor on the net.
#
#   Basic units:
#       - "nand"
#       - "nor"
#       - "inv"
#       - "cap"
#
#   Usage example:
#   # Testing cvxpy: Book example 4.15 (W&H)
#   # By: Hansem Ro <hansem7@uw.edu>
#   #
#   #                   _____
#   # A--------------->|NAND2|
#   # B--[Inv(x=1)]-+->|(x2) |---+
#   #               |  |_____|   |     _____
#   #               +->|NOR2 |   +--->|NOR3 |
#   # C--------------->|(x3) |------->|(x4) |----+----[Inv(x5)]----+
#   #                  |_____|   +--->|_____|    |                 |
#   #                            |              ===c4=10          ===c5=12
#   # D--------------------------+               |                 |
#   #                                            v                 v
#   #
#   # Expected solutions:
#   #  Minimum arrival time = 23.44
#   #  x2 = 1.62
#   #  x3 = 1.62
#   #  x4 = 3.37
#   #  x5 = 6.35
#   
#   # Using logical_unit class
#   # Create inputs (as numpy array of strings)
#   inputs = np.array(["A", "B", "C", "D"])
#   # Create top level module
#   top = solver.top_module(inputs)
#   # Add units starting from input side
#   top.add_inv("B", "net_inv1", drive=1, name="inv1")
#   top.add_unit(np.array(["A", "net_inv1"]), "net_nand2", type="nand", drive="x2", name="nand2")
#   top.add_unit(np.array(["net_inv1", "C"]), "net_nor2", type="nor", drive="x3", name="nor2")
#   top.add_unit(np.array(["net_nand2", "net_nor2", "D"]), "net_nor3", type="nor", drive="x4")
#   top.add_cap("net_nor3", "net_c4", 10, name="c4")
#   top.add_inv("net_nor3", "net_inv2", drive="x5", name="inv2")
#   top.add_cap("net_inv2", "net_c5", 12, name="c5")
#   top.solve()
#

# Table of logical effort for logical units from page 156 of W&H:

def g_inv(n=1):
    return 1

def g_nand(n=2):
    return (n+2)/3

def g_nor(n=2):
    return (2*n+1)/3

# tristate, multiplexer
def g_mux(n=2):
    return 2

# XOR, XNOR
def g_xor(n=2):
    l_half = 2*n*np.ones(n//2)
    range_max = n//2
    if (n % 2 != 0):
        l_half = np.append(l_half,2*n)
        range_max += 1
    for i in range(1,range_max):
        l_half[i] += 2*n*i
    r_half = np.flip(l_half[0:n//2])
    return np.concatenate((l_half, r_half))

# Table of parasitic effort for logical units from page 157 of W&H:

def p_inv(n=1):
    return 1

def p_nand(n=2):
    return n

def p_nor(n=2):
    return n

# tristate, multiplexer
def p_mux(n=1):
    return 2*n

# Common expressions

## Stage expressions

### Electrical effort
def _h(Cout,Cin):
    return Cout/Cin

### Branching effort
def _b(Conpath,Coffpath):
    return (Conpath+Coffpath)/Conpath

### Effort
def _f(g,h):
    return g*h

### Drive
def _x(Cin,g):
    return Cin/g

### Delay
def _d(f,p):
    return f+p

def _d(Cout,x,p):
    return (Cout/x)+p

## Path expressions

### Logical effort
def _G(gs: np.array):
    ret = np.product(gs)

### Electrical effort
def _H(Cout,Cin):
    return Cout/Cin

### Branching effort
def _B(bs: np.array):
    return np.product(bs)

### Effort
def _F(G,B,H):
    return G*B*H

### Effort delay
def _D_F(fs: np.array):
    return np.sum(fs)

### Parasitic delay
def _P(ps: np.array):
    return np.sum(ps)

### Delay
def _D(ds: np.array):
    return np.sum(ds)

def _D(D_F, P):
    return D_F + P

# logical unit class:
#   At the most fundamental, an output is driven by 1 or more inputs:
#       output <-- unit <-- inputs
#
#   Inputs come from a source or another unit.
#
#   More complex cases involve multiple outputs:
#                    _______
#       output1 <-- |decoder| <-- input1
#          ...  <-- |       | <-- ...
#       outputM <-- |_______| <-- inputN
#
#   To simplify this, we can break the logical unit into smaller units
#   with only 1 output:
#
#       output1 <-- [logicA] <-- inputs
#
#          ...  <-- [logicB] <-- inputs
#
#       output1 <-- [logicC] <-- inputs
#
#   With this output assumption, we can perceive an output as
#   a node of a tree:
#
#       Example:
#                    ____
#       outputH <-- |NAND| <-- inputA
#                   |    | <-- [INV] <-- inputB
#                   |____| <-+
#                            |   ___
#                            +--|NOR| <-- inputC
#                               |___| <-- inputD
#
#
#   We can further simplify this by treating the output net
#   and its logical unit driver as a single node.
#   The inputs will be driven logical unit of type None so
#   that the inputs can be identified as such.
#
#   Algorithm idea: since an input can fan-out to 1 or multiple outputs,
#   we can build a tree starting from one of the output nodes.
#   Each output node will have an associated arrival time expression.
#   We will have a super output (with maximum of arrival times of all
#   logical nodes) to find the maximum arrival time delay among the
#   various outputs. The solver will minimize the super class to determine
#   optimal solution involving all output nodes.
#
class logical_unit:

    # Constructor
    # @param inputs: numpy array of input net names
    # @param output: name of output net
    # @param type: type of logic unit
    # @param Cin: (optional) unit capacitance
    # @param drive: (optional) unit drive number or string alias
    # @param name: (optional) unit name
    def __init__(self, inputs: np.ndarray, output, type=None, Cin=None, drive=None, name=None):
        self.output = output
        self.inputs = inputs
        self.name = name
        self.type = type
        self.type_detailed = None
        self.g = None
        self.p = None
        if isinstance(drive,str):
            self.drive = cp.Variable(pos=True, name=drive)
        elif isinstance(drive,int):
            assert drive > 0
            self.drive = cp.Constant(drive)
        else:
            self.drive = drive
        self.Cin = Cin
        if (type is not None and type != "top" and type != "pseudo"):
            if (type != "cap"):
                assert Cin is None
                self.type = "atomic"
                self.type_detailed = type
                if (name is None):
                    self.name = self.type_detailed + str(self.inputs.size) +  "_" + output
                if (type == "inv"):
                    self.g = g_inv()
                    self.p = p_inv()
                elif (type == "nand"):
                    self.g = g_nand(inputs.size)
                    self.p = p_nand(inputs.size)
                elif (type == "nor"):
                    self.g = g_nor(inputs.size)
                    self.p = p_nor(inputs.size)
                # elif (type == "mux"):
                #     self.g = g_mux(inputs.size)
                #     self.p = p_mux(inputs.size)
                # Cin = g*drive
                if (self.drive is None):
                    self.drive = cp.Variable(pos=True, name="x_" + self.name)
                self.Cin = cp.multiply(self.g,self.drive)
            elif (type == "cap"):
                assert Cin is not None
                if (name is None):
                    self.name = self.type + "_" + self.output

    # print_props: print logical unit properties
    def print_props(self):
        print("name: ", self.name)
        print("type: ", self.type)
        if (self.type == "atomic"):
            print("type_detailed: ", self.type_detailed)
            print("g: ", self.g)
            print("p: ", self.p)
            if (hasattr(self, 'd')):
                print("d: ", self.d)
            if (hasattr(self, 'a')):
                print("a: ", self.a)
        print("inputs: ", self.inputs)
        print("output: ", self.output)
        print("num_inputs: ", self.inputs.size)
        print("Cin: ", self.Cin)
        print("drive: ", self.drive)

# Class for top level module with known inputs
class top_module(logical_unit):

    # Constructor:
    # @param inputs: numpy array of input net names
    def __init__(self, inputs: np.ndarray):
        super().__init__(inputs, None, "top")
        # create array of output nets
        self.outputs = np.array([])
        # create set for nets
        self.nets = set()
        # create graph for subnodes
        # key: output net
        # value: net driver unit
        self.nodes = defaultdict(list)
        # add inputs to nets set and add pseudo units for inputs to nodes graph
        for input in inputs:
            assert input not in self.nets
            self.nets.add(input)
            tmp = logical_unit(np.array([]), input, "pseudo")
            self.nodes[input].append(tmp)
    
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
    
    # get_net: returns net associated to unit name.
    # If there is no match, it will return None.
    # @param name: unit name
    def get_net(self, name):
        for net in self.nets:
            for unit in self.nodes[net]:
                if name == unit.name:
                    return net
        return None

    # get_unit: returns driver unit associated to net or name
    # If there is no match, it will return None.
    # @param net: unit net
    # @param name: unit name (if net is specified, it will skip name search)
    def get_unit(self, net=None, name=None):
        if (net is not None):
            assert net in self.nets
            for unit in self.nodes[net]:
                return unit
        elif (name is not None):
            net = self.get_net(name)
            if (net is not None):
                return self.get_unit(name=net)
        return None

    # get_fanout: generate a fanout array for given net
    # @param net: net to inspect
    def get_fanout(self, net):
        ret = np.array([])
        # iterate through all units and add to return
        # list if its input contains net
        for net_check in self.nodes:
            unit = self.get_unit(net=net_check)
            if net in str(unit.inputs[:]):
                # print("adding unit %s to list" % unit.name)
                ret = np.append(ret, unit.name)
        return ret

    # print_props: print properties of all nodes
    def print_props(self):
        for net in self.nodes:
            for unit in self.nodes[net]:
                unit.print_props()
                print("fanout: ", self.get_fanout(net))
                print()

    # get_cap: returns cap expression at given net
    # @param net: net to inspect
    def get_cap(self, net):
        assert net in self.nets
        assert self.check_module()
        cap = 0
        fanout = self.get_fanout(net)
        for name in fanout:
            ret_name = self.get_net(name)
            if ret_name is not None:
                unit = self.get_unit(net=ret_name)
                cap += unit.Cin
        return cap
    
    # check_module: returns true if all nodes are connected and false otherwise
    def check_module(self):
        visited = set()
        for net in self.nets:
            found = False
            unit = self.get_unit(net=net)
            if unit.type == "cap":
                found = True
                visited.add(net)    
            for tmp in self.nodes:
                if (found):
                    break
                elif (tmp != net):
                    unit_tmp = self.get_unit(net=tmp)
                    if net in str(unit_tmp.inputs[:]):
                        visited.add(net)
                        found = True
                        break
        for net in self.nets:
            if net not in visited:
                return False
        return True

    # __get_a: recursively get arrival time for driver unit
    # associated with given net
    def __get_a(self, net):
        assert self.check_module()
        unit = self.get_unit(net=net)
        if (unit is not None):
            if (unit.type == "pseudo"):
                return cp.Constant(0)
            elif (unit.type == "atomic"):
                if (unit.inputs.size == 1):
                    a_tmp = self.__get_a(unit.inputs[0])
                    if a_tmp.is_zero():
                        return unit.d
                    return a_tmp + unit.d
                expr = cp.Constant(0)
                for input in unit.inputs:
                    a_tmp = self.__get_a(input)
                    if expr.is_zero():
                        expr = a_tmp
                    elif not a_tmp.is_zero():
                        expr = cp.maximum(expr, a_tmp)
                if expr.is_zero():
                    return unit.d
                return expr + unit.d
    
    # __max_a: recursively get max arrival time
    def __max_a(self):
        logical_nets = np.array([])
        for net in self.nets:
            unit = self.get_unit(net=net)
            if unit is not None and unit.type is not None and unit.type == "atomic":
                logical_nets = np.append(logical_nets, net)
        self.add_unit(logical_nets, "net_global", "atomic", name="pseudo_global")
        self.add_cap("net_global", "net_fake_cap", 0, name="fake_cap")
        self.get_unit(net="net_global").d = 0
        a = self.__get_a("net_global")
        self.get_unit(net="net_global").a = a
        return a


    # solve: solve and print optimal sizes
    def solve(self):
        assert self.check_module()
        # generate delay statement for each module
        for net in self.nets:
            unit = self.get_unit(net=net)
            if unit.type is not None and unit.type != "pseudo" and unit.type != "cap":
                unit.d = self.get_cap(net)/unit.drive
                if (unit.p is not None):
                    unit.d += unit.p
                # print("d_", unit.name, " = ", unit.d)
        # build arrival time expressions while considering fanout
        for net in self.nets:
            unit = self.get_unit(net=net)
            if unit.type is not None and unit.type != "pseudo" and unit.type != "cap":
                unit.a = self.__get_a(net)
                # print("a_", unit.name, " = ", unit.a)
        # build final expression
        self.a = self.__max_a()
        # create problem and solve
        self.problem = cp.Problem(cp.Minimize(self.a))
        print("Problem solvable: ", self.problem.is_dgp(dpp=True))
        print("Minimum arrival time: ", self.problem.solve(gp=True))

        # print solution
        for net in self.nets:
            unit = self.get_unit(net=net)
            if unit.type is not None and unit.type == "atomic" and unit.name != "pseudo_global":
                print("name: ", unit.name, " drive: ", unit.drive.value)