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
        from eff_numpy import cond
        @trace
        def main(x):
            return x + x

        one = numpy.array(1, dtype=numpy.int64)
        x, shape = main(one)
        print(x)
        import jax
        from jax._src.interpreters import mlir
        module = mlir.ir.Module.parse(str(x), context=mlir.make_ir_context())
        bytecode = mlir.module_to_bytecode(module)
        client = jax._src.xla_bridge.backends()["cpu"]
        loaded = client.compile(bytecode)
        jax.config.update("jax_enable_x64", True)
        a = jax.numpy.array(1000, dtype=jax.numpy.int64)
        flat_data = loaded.execute([a])
        out = pytrees.unflatten(flat_data, shape)
        print(out)

if "__main__" == __name__:
    unittest.main()
