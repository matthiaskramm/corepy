# Prototypes for InstructionStream Version 2

# Goals:
#   o Separation of InstructionStream and BinaryInstructionStream
#     - Same idea as Instruction/MachineInstruction
#     - IS has all user level IS operations
#     - BIS is the machine specific binary rendering
#   o Modular instruction streams
#     - Somewhere between basic blocks and functions
#     - Can be used to implement any type of code block abstraction  
#   o Composable instruction streams
#     - Joins, branches
#     - Interleaving, pipelining
#   o Branch/label abstractions (reinvent goto!)
#   o Register abstractions
#     - Register 'ports'
#     - Named registers
#   o Procedural features
#     - optional stack (for IA-32)
#     - link register

# Ambitious Goals:
#   o Capture advanced programming contructs easily
#     - Continuations
#     - PIM
#     - etc

# Tanimoto example

# This is an intersting idea... objects that when constructed are ISs
# for the function.  Kinda like the ideas used for synthesis, but more
# self contained.
div = DivIS()
tan = Tanimoto()

block_proc = BlockProc(kernel = Tanimoto, x_addr = Register_Port, y_addr = Register_Port)


code = IS2()
x_addr = code.named_register('x_addr')
y_addr = code.named_register('y_addr')

data = make_array()
x_code = DataStreamIS(data)

block_proc.set_x_stream(x_code)
