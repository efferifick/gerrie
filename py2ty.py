def create_tracers(pyval):
    return replace_concrete_values_with_tracers(pyval)

import builtins
import numpy

from xdsl.dialects.builtin import i32, i64
from stablehlo import i1
from stablehlo import si2, si4, si8, si16, si32, si64
from stablehlo import ui2, ui4, ui8, ui16, ui32, ui64
from stablehlo import f16, f32, f64
from stablehlo import complexf32, complexf64
from stablehlo import TensorType

def convert_python_type_to_mlir_type(pytype):
    match pytype:
        case builtins.bool:    return i1
        case builtins.int:     return i64
        case builtins.float:   return f64
        case builtins.complex: return complexf64
    msg = f"Unknown conversion from type {pytype} to MLIR type."
    raise ValueError(msg)

import unittest
class TestPythonTypeToMLIRType(unittest.TestCase):
    def test_python_to_mlir_type(self):
        input = [bool, int, float, complex]
        output = [i1, i64, f64, complexf64]
        for arg, exp in zip(input, output):
            with self.subTest(arg=arg, exp=exp):
                obs = convert_python_type_to_mlir_type(arg)
                self.assertEqual(obs, exp)

def convert_numpy_dtype_to_mlir_type(dtp):
    match dtp:
        case builtins.bool    | numpy.bool_   : return i1
        case numpy.int8       | numpy.byte    : return si8
        case numpy.uint8      | numpy.ubyte   : return ui8
        case numpy.int16      | numpy.short   : return si16
        case numpy.uint16     | numpy.ushort  : return ui16
        case numpy.int32      | numpy.intc    : return si32
        case numpy.uint32     | numpy.uintc   : return ui32
        case numpy.int64                      : return si64
        case numpy.uint64                     : return ui64
        case numpy.float16    | numpy.half    : return f16
        case numpy.float32    | numpy.single  : return f32
        case numpy.float64    | numpy.double  : return f64
        case numpy.complex64  | numpy.csingle : return complexf32
        case numpy.complex128 | numpy.cdouble : return complexf64
    msg = f"Unknown conversion from dtype {dtp} to MLIR type."
    raise ValueError(msg)

def convert_numpy_array_to_mlir_type(pyval):
    assert isinstance(pyval, numpy.ndarray)
    element_type = convert_numpy_dtype_to_mlir_type(pyval.dtype)
    return TensorType(element_type, pyval.shape)

class TestConvertNumpyArray(unittest.TestCase):
    def test_convert_numpy_array(self):
        convert_numpy_array_to_mlir_type(numpy.array(0))


def get_mlir_type_from_python_value(pyval):
    ty = type(pyval)
    match ty:
        case numpy.ndarray: return convert_numpy_array_to_mlir_type(pyval)
        case _:             return convert_python_type_to_mlir_type(ty)

class TestGetMLIRTypeFromPythonValue(unittest.TestCase):
    def test_python_val_to_mlir_type(self):
        input = [False, 0, 0., 0j]
        output = [i1, i64, f64, complexf64]
        for arg, exp in zip(input, output):
            with self.subTest(arg=arg, exp=exp):
                obs = get_mlir_type_from_python_value(arg)
                self.assertEqual(obs, exp)

def get_mlir_types_from_python_values(flat_pyvals):
    return map(get_mlir_type_from_python_value, flat_pyvals)

from pennylane import pytrees
def replace_concrete_values_with_mlir_types(pyval):
    flat_vals, shape = pytrees.flatten(pyval)
    flat_mlirtys = get_mlir_types_from_python_values(flat_vals)
    return pytrees.unflatten(flat_mlirtys, shape)

class TestReplaceConcreteValuesWithMLIRTypes(unittest.TestCase):
    def test_python_unflattened_val_to_mlir_type(self):
        input = [[False], (0,), {"a":0.}, 0j]
        output = [[i1], (i64,), {"a":f64}, complexf64]
        for arg, exp in zip(input, output):
            with self.subTest(arg=arg, exp=exp):
                obs = replace_concrete_values_with_mlir_types(arg)
                self.assertEqual(obs, exp)

from xdsl.ir import Block
from pennylane import pytrees
def replace_concrete_values_with_block_arguments(pyval):
    shaped_mlirtys = replace_concrete_values_with_mlir_types(pyval)
    mlirtys, shape = pytrees.flatten(shaped_mlirtys)
    block = Block([], arg_types=mlirtys)
    ssavalues = block.args
    return pytrees.unflatten(ssavalues, shape)


class TestPythonValueToSSAValue(unittest.TestCase):

    def test_python_to_mlir_type(self):
        from xdsl.ir.core import BlockArgument
        input = [False, 0, 0., 0j]
        for arg in input:
            with self.subTest(arg=arg):
                obs = replace_concrete_values_with_block_arguments(arg)
                assert isinstance(obs, BlockArgument)

from tracer import StableHLOTracer, Tracer
def get_tracer_from_ssavalue(ssavalue):
    return StableHLOTracer(ssavalue)

class TestPythonValueToTracer(unittest.TestCase):
    def test_python_value_to_tracer(self):
        block = Block([], arg_types=[i1, i64, f64, complexf64])
        for arg in block.args:
            with self.subTest(arg=arg):
                obs = get_tracer_from_ssavalue(arg)
                assert isinstance(obs, Tracer)

def replace_concrete_values_with_tracers(pyval):
    shaped_ssavalues = replace_concrete_values_with_block_arguments(pyval)
    ssavals, shape = pytrees.flatten(shaped_ssavalues)
    tracers = map(get_tracer_from_ssavalue, ssavals)
    return pytrees.unflatten(tracers, shape)

if "__main__" == __name__:
    unittest.main()
