#!/usr/bin/python3

import argparse
import re

import time

"""
Overview of allocator:
    1) verify cmd args
    2) read input file
    3) parse file to some intermediate format
    4) run analysis on code
    5) allocate registers
    6) generate code including spill
    
How registers are processed:
    - As each instruction is parsed, Operands are constructed. Operands are stored in each Instruction object. 
    - As the operands are parsed, virtual registers are detected and created and stored in the Instruction.
"""


class Operand:
    """
    Represents an operand in an instruction. Can be extended to provide special functionality such as overriding
    the generate function to return a physical register translation.
    """
    def __init__(self, text, lineno):
        self._value = text
        self._line = lineno

    def __hash__(self):
        return hash(self.value())

    def __eq__(self, other):
        if isinstance(other, Operand):
            return self.value() == other.value()
        return False

    def __repr__(self):
        return "Operand<%s>" % self.value()

    def value(self):
        """
        :return: the string value of the operand as written in the input program
        """
        return self._value

    def line(self):
        """
        :return: returns the line number (or instruction number) this operand was referenced on
        """
        return self._line

    def generate(self):
        """
        :return: the string value that is to be used in the generated output
        """
        return self._value


class VirtualRegister(Operand):
    next_id = 1

    def __init__(self, text, lineno):
        super().__init__(text, lineno)
        self._physical = None
        self._id = VirtualRegister.next_id
        VirtualRegister.next_id += 1

    def __repr__(self):
        return "VirtualReg<%s>" % self.value()

    def id(self):
        return self._id

    def set_register(self, reg):
        self._physical = reg

    def generate(self):
        if self._physical:
            return self._physical
        return "v" + self.value()

    def is_assigned(self):
        return bool(self._physical)


class RegisterMetaData:
    def __init__(self, tag):
        self._tag = tag
        self._occurrences = []
        self._count = 0
        self._first = -1
        self._last = -1
        self._rematerialize = False
        self._initialize = -1

    def __repr__(self):
        return "VRegMetaData<%s>" % self.tag()

    def __lt__(self, other):
        return self.tag() < other.tag()

    def tag(self):
        return self._tag

    def occurrence(self, vreg, init=False, rematerialize=False):
        global filename
        if init and self.init_line() != -1:
            print("%s:%d: Illegal reassignment of virtual register '%s'" % (filename, vreg.line(), self.tag()))
            print("Cannot allocate registers. Exiting")
            quit(1)
        elif not init and self.init_line() == -1:
            print("%s:%d: Illegal use of virtual register '%s' before assignment" % (filename, vreg.line(), self.tag()))
            print("Cannot allocate registers. Exiting")
            quit(1)

        self._occurrences.append(vreg)
        self._count += 1

        if init:
            self._initialize = vreg.line()

        if rematerialize:
            self._rematerialize = True

    def rematerializable(self):
        return self._rematerialize

    def init_line(self):
        return self._initialize

    def num_occurrences(self):
        return self._count

    def occurrences(self):
        return self._occurrences

    def first(self):
        return self.occurrences()[0]

    def last(self):
        return self.occurrences()[-1]

    def span(self):
        return self.last().line() - self.first().line()

    def alive_at(self, line):
        if self.first().line() == self.last().line() == line:
            return True
        return self.first().line() <= line < self.last().line()

    def next_occurrence(self, instance):
        occurrences = self.occurrences()
        for i in range(len(occurrences)):
            occ = occurrences[i]

            if occ.id() == instance.id() and i < len(occurrences) - 1:
                return occurrences[i+1]
        return None

    def occurrence_after(self, lineno):
        for occ in self.occurrences():
            if occ.line() > lineno:
                return occ
        return None


class Instruction:

    parser = re.compile(r"(?m)^\s*(\w+)\s+(\w+(?:\s*,\s*(?:-)?\w+)?)?(?:\s*=>\s*(\w+(?:\s*,\s*(?:-)?\w+)?))?")

    def __init__(self, data, lineno):
        match = Instruction.parser.match(data)
        opcode, src, dest = match.group(1, 2, 3)
        self.lineno = lineno

        self.opcode = opcode
        self.src = self.parse_operands(src)
        self.dest = self.parse_operands(dest)

    def line(self):
        return self.lineno

    def generate(self):
        src = tuple(s.generate() for s in self.src)
        dest = tuple(d.generate() for d in self.dest)
        src_str = (" " + ", ".join(src))
        dest_str = (" => " + ", ".join(dest)) if dest else ""
        return self.opcode + src_str + dest_str

    def parse_operands(self, operands):
        # if operand is None or empty string, return empty list
        if not operands:
            return ()
        # otherwise split and strip space
        operands = [op.strip().lower() for op in operands.split(",")]

        def generator(op):
            if op.startswith("r") and op[1] != "0":
                return VirtualRegister(op, self.line())
            return Operand(op, self.line())

        return tuple(map(generator, operands))

    def registers(self):
        return tuple(filter(lambda x: isinstance(x, VirtualRegister), self.src + self.dest))

    def copy(self):
        return Instruction(self.generate(), self.lineno)


class InstructionList(list):
    def __init__(self, data):
        super().__init__(self)
        ignored = re.compile(r'(?m)^\s*(//.*)?$')  # ignore white space and comment lines

        lineno = 0
        for line in data.split("\n"):
            if ignored.match(line):
                continue
            self.append(Instruction(line, lineno))
            lineno += 1
        self.lineno = lineno

    def num_lines(self):
        return self.lineno


filename = None
num_physical = 0
input_code = None
metadata = None
physical = None
feasible = None
verbose = False
output_code = InstructionList("")


def physical_regs():
    global num_physical
    return ["r%d" % i for i in range(1, num_physical+1)]


def compute_metadata():
    global input_code, metadata
    metadata = {}

    for i in input_code:
        for reg in i.registers():
            init = False
            rematerialize = False
            if reg in i.dest and not i.opcode.startswith("store"):
                init = True
                if i.opcode == "loadI":
                    rematerialize = True

            if reg not in metadata:
                metadata[reg] = RegisterMetaData(reg.value())
            metadata[reg].occurrence(reg, init, rematerialize)


def generate_output():
    """
    Job of generate_output: take input code and produce actual output.
    At this point, an allocation algorithm has already assigned physical registers to every virtual register.
    For each instruction:
        For each virtual register not being initialized:
            if it is assigned to a physical register and is not in that physical register:
                store the value in the physical register
                load the virtual register into the assigned physical register
            if it is assigned to a physical register and in the register:
                continue to next register
            if it is not assigned to a physical register:
                load value into feasible register
                assign feasible register to virtual register

        add instruction to output_code

        for each virtual register being initialized:
            if it is not assigned to a physical register:
                assign virtual register to feasible register
                store feasible register

    """
    global input_code, output_code, physical, feasible, metadata

    assignments = {}
    for reg in physical:
        # all physical registers currently not assigned to anything
        assignments[reg] = None

    for instruction in input_code:
        # separate registers by assigned and unassigned
        assigned = tuple(filter(lambda x: x.is_assigned(), instruction.registers()))
        unassigned = tuple(filter(lambda x: not x.is_assigned(), instruction.registers()))

        # this loop generates loading spill code with feasible registers
        for reg in unassigned:
            # for any unassigned registers being read, we generate spill code to load the value
            if metadata[reg].first().line() < instruction.line():
                # get a feasible register and calculate offset where the virtual reg should be stored
                fr = feasible.pop(0)
                feasible.append(fr)
                offset = int(reg.value()[1:], 10) * 4
                # create the load instruction, and assign the physical register to the virtual register(s)
                load_instr = Instruction("loadAI r0, -%d => %s" % (offset, fr), 0)
                load_instr.registers()[0].set_register(fr)
                reg.set_register(fr)
                # add load instruction to output
                output_code.append(load_instr)

        # this loop generates spill code for assigned registers -- stores old value and loads new one
        for vreg in assigned:
            assignment = vreg.generate()
            if assignments[assignment] != vreg:
                # if the register is not currently in the assigned physical register, we must load it
                in_physical = assignments[assignment]

                if in_physical:
                    # if there's another register present, and it is still alive, we must store it
                    md = metadata[in_physical]
                    if md.alive_at(instruction.line()) and not md.rematerializable():
                        # store the assigned register if the register is still alive and can't be rematerialized
                        offset = int(in_physical.value()[1:], 10) * 4
                        store_instr = Instruction("storeAI %s => r0, -%d" % (assignment, offset), 0)
                        store_instr.registers()[0].set_register(assignment)
                        output_code.append(store_instr)

                vreg_md = metadata[vreg]
                if vreg_md.first().line() != instruction.line() and vreg_md.rematerializable():
                    # if it's not the initialization and the value is rematerializable, rematerialize it
                    init_instr = input_code[vreg_md.init_line()].copy()
                    init_instr.registers()[0].set_register(assignment)
                    output_code.append(init_instr)
                elif vreg_md.first().line() != instruction.line():
                    # if it's not rematerializable, load it's value from memory
                    offset = int(vreg.value()[1:], 10) * 4
                    load_instr = Instruction("loadAI r0, -%d => %s" % (offset, assignment), 0)
                    load_instr.registers()[0].set_register(assignment)
                    output_code.append(load_instr)
                assignments[assignment] = vreg  # mark that the physical register has a new virtual value

        output_code.append(instruction)

        # this loop generates storage spill code using feasible registers
        for vreg in unassigned:
            if metadata[vreg].first().line() == instruction.line():
                fr = feasible.pop(0)
                offset = int(vreg.value()[1:], 10) * 4
                vreg.set_register(fr)
                store_instr = Instruction("storeAI %s => r0, -%d" % (fr, offset), 0)
                store_instr.registers()[0].set_register(fr)
                output_code.append(store_instr)
                feasible.append(fr)

    for instr in output_code:
        print(instr.generate())


def simple_top_down():
    global num_physical, input_code, output_code, metadata, physical, feasible, verbose

    ranking = sorted(metadata.values(), key=lambda x: x.num_occurrences(), reverse=True)

    if len(ranking) > num_physical:
        # only reserve feasible set if we have more virtual registers than physical registers
        feasible = physical[:2]
        physical = physical[2:]
        ranking = ranking[:len(physical)]

    assignments = {}
    for md in ranking:
        # assign registers
        assignments[md.tag()] = physical.pop()
        if verbose:
            print("// Assigning physical %s to virtual reg %s" % (assignments[md.tag()], md.tag()))


def top_down():
    global num_physical, input_code, output_code, metadata, physical, feasible, verbose

    live_sets = []
    max_live = 0

    # calculate live ranges and max live
    for instr in input_code:
        # get all registers alive at this line number
        live_set = set(filter(lambda x: x.alive_at(instr.line()), metadata.values()))
        live_sets.append(live_set)
        max_live = max(max_live, len(live_set))

    feasible = []
    physical = physical_regs()

    if max_live > num_physical:
        # only allocate feasible registers if necessary
        feasible = physical[:2]
        physical = physical[2:]
        if verbose:
            print("// Feasible set: [%s, %s]" % (feasible[0], feasible[1]))
    elif verbose:
        print("// No feasible used")

    spilled_set = set()

    # spill necessary registers on any lines where the number of live registers is greater than the available
    for live_set in live_sets:
        # remove any registers already spilled
        live_set = live_set.difference(spilled_set)
        # spill registers as needed
        while len(live_set) > len(physical):
            # spill virtual register with lowest number of occurrences (break ties by spilling larger span)
            # a larger span, means 65535 - span will be lower, and thus be lower than a shorter span
            spilled_reg = min(live_set, key=lambda x: (x.num_occurrences() << 16) | (65535 - x.span()))
            live_set.discard(spilled_reg)
            spilled_set.add(spilled_reg)

    available_set = set(physical)
    previous_set = set()

    if verbose:
        print("// spilled set: ", sorted(spilled_set))

    # go through each instruction and assign virtual registers to physical registers
    for instr in input_code:
        live_set = live_sets.pop(0)
        live_set = live_set.difference(spilled_set)
        if verbose:
            print("// input %d -> " % instr.line(), instr.generate())
            print("// live set: ", ["v" + r.tag() for r in live_set])

        for reg_md in previous_set:
            # return any allocations ending on this line to the available set
            if reg_md.last().line() <= instr.line():
                if verbose:
                    print("// ", reg_md.last(), " returning physical ", reg_md.last().generate())
                available_set.add(reg_md.last().generate())

        for reg_md in live_set:
            if reg_md.first().line() == instr.line():
                # if it's not assigned, assign it to an available register
                if verbose:
                    print("// available: ", available_set)
                assignment = available_set.pop()
                if verbose:
                    print("// assigning physical ", assignment, " to ", "v" + reg_md.first().value())
                for instance in reg_md.occurrences():
                    # assign the specified register to every use of this virtual register
                    instance.set_register(assignment)
        previous_set = live_set
        if verbose:
            print("// output -> ", instr.generate(), "\n")


def custom():
    return []


def bottom_up():
    global input_code, output_code, metadata, physical, verbose

    physical = physical_regs()
    assignments = dict()

    for reg in physical:
        assignments[reg] = None

    def best_candidate(lineno, init_line):
        furthest = None
        furthest_vreg = None
        furthest_line = 0

        for reg in physical:
            if not assignments[reg]:
                # if no virtual register is assigned to this physical register, return it
                if verbose:
                    print("// ", reg, " is not assigned")
                return reg
            # get metadata of virtual reg stored in physical reg
            info = metadata[assignments[reg]]
            if not info.alive_at(lineno) and not info.alive_at(lineno - 1):
                # if the virtual register residing in the physical expired before this line, return this physical reg
                if verbose:
                    print("// v", info.tag(), " has been expired - returning ", reg, sep="")
                return reg
            elif not info.alive_at(lineno) and init_line == lineno:
                # if it expires this line, and this line is initializing the current vreg, return this physical reg
                if verbose:
                    print("// v", info.tag(), " expires this line - returning ", reg, sep="")
                return reg
            # otherwise the virtual register is live

            next_occurrence = info.occurrence_after(lineno)
            if next_occurrence and next_occurrence.line() > furthest_line:
                # record this register if the virtual register stored in it is used furthest in the future
                furthest = reg
                furthest_vreg = info
                furthest_line = next_occurrence.line()

            elif next_occurrence and next_occurrence.line() == furthest_line:
                # if there's a tie in distance, see if either is rematerializable
                if not furthest_vreg.rematerializable() and info.rematerializable():
                    # if the new one is rematerializable and the old one isn't, record it instead
                    furthest = reg
                    furthest_vreg = info
                    furthest_line = next_occurrence.line()

        # if we got here, we return the physical register containing the furthest use virtual reg
        if verbose:
            print("// v", furthest_vreg.tag(), " has furthest use - returning ", furthest, sep="")
            for i in physical:
                print("// ", i, " contains v", assignments[i].value(), sep="")

        return furthest

    for instr in input_code:
        if verbose:
            print("// input %d -> " % instr.line(), instr.generate())
        for reg in instr.registers():
            if not reg.is_assigned() or assignments[reg.generate()] != reg:
                # if it's not assigned, or has been spilled, assign it
                assignment = best_candidate(instr.line(), metadata[reg].init_line())
                assignments[assignment] = reg
                # set all future uses of this virtual register to use this physical register
                reg.set_register(assignment)
                reg = metadata[reg].next_occurrence(reg)
                while reg:
                    reg.set_register(assignment)
                    reg = metadata[reg].next_occurrence(reg)
        if verbose:
            print("// output -> ", instr.generate(), "\n")


def read_file():
    global filename
    try:
        input_file = open(filename, mode="r")
        data = input_file.read()
        input_file.close()
        return data
    except IOError:
        print("File not found in path - exiting")
        exit(1)


def execute_allocation(algorithm):
    global input_code, output_code, physical, feasible

    file_content = read_file()
    input_code = InstructionList(file_content)
    algorithms = {"b": bottom_up, "s": simple_top_down, "t": top_down, "o": custom}
    # call requested allocation algorithm
    alloc_algorithm = algorithms[algorithm]

    physical = physical_regs()
    feasible = []

    start_time = time.time()
    try:
        compute_metadata()
        alloc_algorithm()
        generate_output()
    except Exception as e:
        time.sleep(1)
        raise e
    end_time = time.time()

    print("// total time for execution: %0.5f ms" % ((end_time - start_time)*1000))

    print("\n\n//Original: %d\n//Output: %d\n//Spill code: %d\n" %
          (len(input_code), len(output_code), len(output_code)-len(input_code)))
    #print("\n//Virtual Registers: %d\n//Spilled Registers: %d\n" %
    #      (len(code.virtual_regs()), len([reg for reg in code.virtual_regs() if reg.spilled()])))


def main():
    global filename, num_physical, verbose
    parser = argparse.ArgumentParser(description="CS415 Project 1 -- local register allocation")
    parser.add_argument("k", type=int, help="Number of physical registers")
    parser.add_argument("algorithm", choices=["b", "s", "t", "o"], help="Allocation algorithm")
    parser.add_argument("file", help="ILOC input file")
    parser.add_argument("-v", "--verbose", help="Verbose output - print allocation information", action='store_true')

    args = vars(parser.parse_args())
    num_physical = args["k"]
    algorithm = args["algorithm"]
    filename = args["file"]
    verbose = args["verbose"]

    execute_allocation(algorithm)


if __name__ == "__main__":
    main()
