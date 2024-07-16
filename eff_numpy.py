#   #     # #     # #     # ######  #     # 
#   ##    # #     # ##   ## #     #  #   #  
#   # #   # #     # # # # # #     #   # #   
#   #  #  # #     # #  #  # ######     #    
#   #   # # #     # #     # #          #    
#   #    ## #     # #     # #          #    
#   #     #  #####  #     # #          #    

from tracer import Tracer
from xdsl.builder import ImplicitBuilder
from stablehlo import ConstantOp, IfOp, ReturnOp
from pennylane import pytrees

def absolute(arr):
    return abs(arr)

def arccos(arr): ...
def acosh(arr): ...

def add(x, y):
    return x + y

def all(arr, axis=None, out=None, keepdims=False, *, where=None): ...

#         #    #    #     #
#         #   # #    #   #
#         #  #   #    # #
#         # #     #    #
#   #     # #######   # #
#   #     # #     #  #   #
#    #####  #     # #     #

def cond(pred, true_fun, false_fun, *operands):
    # Create two blocks one for each path
    # Trace each path
    # Create IfOp
    from xdsl.ir import Region, Block
    from eff_types import i1, si64
    from eff_type_aliases import TensorType
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    ty = TensorType(i1, [])
    val = DenseIntOrFPElementsAttr.from_list(ty, [pred])
    predOp = ConstantOp(val, ty).results[0]
    
    true_region  = Region(Block([], arg_types=[]))
    false_region = Region(Block([], arg_types=[]))

    with ImplicitBuilder(true_region.blocks[0]):
        ret_true = true_fun(*operands)

    ret_true_flat, true_shape = pytrees.flatten(ret_true)
    ret_ssa_true = [tracer.ssaval for tracer in ret_true_flat]
    with ImplicitBuilder(true_region.blocks[0]):
        ReturnOp(*ret_ssa_true)

    with ImplicitBuilder(false_region.blocks[0]):
        ret_false = false_fun(*operands)

    ret_false_flat, false_shape = pytrees.flatten(ret_false)
    ret_ssa_false = [tracer.ssaval for tracer in ret_false_flat]
    with ImplicitBuilder(false_region.blocks[0]):
        ReturnOp(*ret_ssa_false)

    assert true_shape == false_shape
    for tval, fval in zip(ret_true_flat, ret_false_flat):
        tty = tval.type
        fty = fval.type
        assert tty == fty
    result_tys = [tval.type for tval in ret_true_flat]
    ifOp = IfOp(predOp, true_region, false_region, result_tys)
    from tracer import StableHLOTracer
    outtracers = [StableHLOTracer(res) for res in ifOp.results]
    return pytrees.unflatten(outtracers, true_shape)





