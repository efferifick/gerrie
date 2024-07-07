import functools
from py2ty import create_tracers
from pennylane import pytrees

def trace(callable):

    @functools.wraps(callable)
    def wrapper(*args, **kwargs):
        from xdsl.builder import ImplicitBuilder
        from xdsl.dialects.func import FuncOp, Return
        from xdsl.ir import Region

        tracerargs, tracerkwargs = create_tracers((args, kwargs))
        flat_intracers, _shape = pytrees.flatten((tracerargs, tracerkwargs))
        inputty = [tracer.type for tracer in flat_intracers]
        block = get_block_from_tracers(tracerargs, tracerkwargs, [])
        with ImplicitBuilder(block):
            tracerrets = callable(*tracerargs, **tracerkwargs)
        flat_outtracers, out_shape = pytrees.flatten(tracerrets)
        flat_outssavals = [tracer.ssaval for tracer in flat_outtracers]
        outputty = [tracer.type for tracer in flat_outtracers]
        with ImplicitBuilder(block):
            Return(*flat_outssavals)
        name = callable.__name__
        functy = inputty, outputty
        region = Region(block)
        funcOp = FuncOp(name, functy, region)
        return funcOp, out_shape
        

    return wrapper

def get_block_from_tracers(args, kwargs, rets):
    tracers, _shape = pytrees.flatten((args, kwargs, rets))
    blocks = [tracer.owner for tracer in tracers]
    assert all(map(lambda x: blocks[0] == x, blocks))
    return blocks[0]

#  ████████ ███████ ███████ ████████ ███████ 
#     ██    ██      ██         ██    ██      
#     ██    █████   ███████    ██    ███████ 
#     ██    ██           ██    ██         ██ 
#     ██    ███████ ███████    ██    ███████ 

import unittest

class TestTrace(unittest.TestCase):

    def test_trace_produces_callable(self):

        def identity(x):
            return x

        assert callable(trace(identity))

    def test_call_tracer(self):

        import numpy
        @trace
        def identity(x, y):
            z = x + abs(y)
            z2 = z & z
            return z2.astype(numpy.float32)

        one = numpy.array(1)
        x, _ = identity(one, one)
        print(x)

if "__main__" == __name__:
    unittest.main()
