# Token types represent tokens, i.e. opaque values produced and consumed by
# some operations. Tokens are used for imposing execution order on operations
# as described in the Execution section.
from xdsl.irdl import irdl_attr_definition
from xdsl.ir import ParametrizedAttribute

@irdl_attr_definition
class TokenType(ParametrizedAttribute):
    name = "token"

# Tensor types represent tensors, i.e. multidimensional arrays.
from xdsl.dialects.builtin import TensorType, UnrankedTensorType

# Element Types
# Boolean type represents boolean values true and false.
# BooleanType ::= 'i1'
from xdsl.dialects.builtin import i1

# Integer types can be either signed (si) or unsigned (ui)
# IntegerType ::= SignedIntegerType | UnsignedIntegerType
# SignedIntegerType ::= 'si2' | 'si4' | 'si8' | 'si16' | 'si32' | 'si64'
# UnsignedIntegerType ::= 'ui2' | 'ui4' | 'ui8' | 'ui16' | 'ui32' | 'ui64'
from xdsl.dialects.builtin import IntegerType, Signedness
# and have one of the supported bit widths (2, 4, 8, 16, 32 or 64)

si2  = IntegerType(2,  Signedness.SIGNED)
si4  = IntegerType(4,  Signedness.SIGNED)
si8  = IntegerType(8,  Signedness.SIGNED)
si16 = IntegerType(16, Signedness.SIGNED)
si32 = IntegerType(32, Signedness.SIGNED)
si64 = IntegerType(64, Signedness.SIGNED)


ui2  = IntegerType(2,  Signedness.UNSIGNED)
ui4  = IntegerType(4,  Signedness.UNSIGNED)
ui8  = IntegerType(8,  Signedness.UNSIGNED)
ui16 = IntegerType(16, Signedness.UNSIGNED)
ui32 = IntegerType(32, Signedness.UNSIGNED)
ui64 = IntegerType(64, Signedness.UNSIGNED)


# Floating-point types can be one of the following: 
# bf16 type corresponding to the bfloat16 format described in BFloat16:
# The secret to high performance on Cloud TPUs.
from xdsl.dialects.builtin import bf16

# f16, f32 and f64 types corresponding to respectively
# binary16 ("half precision"), binary32 ("single precision")
# and binary64 ("double precision") formats described in the IEEE 754 standard.
from xdsl.dialects.builtin import f16, f32, f64


# Complex types represent complex values that have a real part
# and an imaginary part of the same element type. Supported
# complex types are complex<f32> (both parts are of type f32)
# and complex<f64> (both parts are of type f64).
from xdsl.dialects.builtin import ComplexType
complex32 = ComplexType(f32)
complex64 = ComplexType(f64)


from typing import Annotated, TypeAlias
I1 = Annotated[IntegerType, i1]
StableHLOBoolean : TypeAlias = (I1)

SI2  = Annotated[IntegerType, si2]
SI4  = Annotated[IntegerType, si4]
SI8  = Annotated[IntegerType, si8]
SI16 = Annotated[IntegerType, si16]
SI32 = Annotated[IntegerType, si32]
SI64 = Annotated[IntegerType, si64]
StableHLOSignedInteger : TypeAlias = (SI2 | SI4 | SI8 | SI32 | SI64)

UI2  = Annotated[IntegerType, ui2]
UI4  = Annotated[IntegerType, ui4]
UI8  = Annotated[IntegerType, ui8]
UI16 = Annotated[IntegerType, ui16]
UI32 = Annotated[IntegerType, ui32]
UI64 = Annotated[IntegerType, ui64]
StableHLOUnsignedInteger : TypeAlias = (UI2 | UI4 | UI8 | UI32 | UI64)

from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, Float64Type

StableHLOFloat : TypeAlias = (BFloat16Type | Float16Type | Float32Type | Float64Type)

Complex32 = Annotated[ComplexType, f32]
Complex64 = Annotated[ComplexType, f64]

StableHLOComplex : TypeAlias = (Complex32 | Complex64)
StableHLOElementType : TypeAlias = (StableHLOBoolean | StableHLOSignedInteger | StableHLOUnsignedInteger | StableHLOFloat | StableHLOComplex)

StableHLORankedTensor : TypeAlias = TensorType[StableHLOElementType]
StableHLOUnrankedTensor : TypeAlias = UnrankedTensorType[StableHLOElementType]
StableHLOTensor : TypeAlias = (StableHLORankedTensor | StableHLOUnrankedTensor)

from xdsl.irdl import AnyOf, IRDLOperation, irdl_op_definition, operand_def, prop_def, result_def
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
@irdl_op_definition
class AbsOp(IRDLOperation):
    name = "stablehlo.abs"
    operand = operand_def(StableHLORankedTensor)
    result = result_def(StableHLORankedTensor)

    def __init__(self, operand, result_ty):
        super().__init__(operands=(operand,),
                         result_types=(result_ty,))

    def verify_(self):
        msg1 = f'''{AbsOp.name} has constraint "shape(result) = shape(operand)"'''
        operandty = self.operand.type
        resultty = self.result.type
        msg1 += f'instead has mismatched "shape(result) = {resultty} and shape(operand) = {operandty}'
        assert resultty.shape == operandty.shape, msg1
        msg2 = f"{AbsOp.name} has constraint"
        msg2 += "    baseline_element_type(result) is defined as:"
        msg2 += "    complex_element_type(element_type(operand)) if is_complex(operand)"
        msg2 += "    baseline_element_type(operand) otherwise."
        resultety = resultty.element_type
        operandety = operandty.element_type
        assert resultety == operandety, msg2

import unittest
class TestAbsOp(unittest.TestCase):
    def test_abs_op(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        one = ConstantOp(val, ty)
        absop = AbsOp(one, ty)
        expected = """%0 = "stablehlo.abs"(%1) : (tensor<1xsi64>) -> tensor<1xsi64>"""
        observed = str(absop)
        assert expected == observed

    def test_constraint_same_shapes(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        one = ConstantOp(val, ty)
        ty2 = TensorType(si64, [2])
        absop = AbsOp(one, ty2)
        with self.assertRaises(AssertionError):
            absop.verify_()

    def test_constraint_same_etype(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        one = ConstantOp(val, ty)
        ty2 = TensorType(si32, [1])
        absop = AbsOp(one, ty2)
        with self.assertRaises(AssertionError):
            absop.verify_()

@irdl_op_definition
class AddOp(IRDLOperation):
    name = "stablehlo.add"
    lhs = operand_def(StableHLORankedTensor)
    rhs = operand_def(StableHLORankedTensor)
    result = result_def(StableHLORankedTensor)

    def __init__(self, lhs, rhs, result):
        super().__init__(operands=(lhs, rhs),
                         result_types=(result,))

    @staticmethod
    def msgC1():
        msg =  'If the operation uses non-quantized tensors: '
        msg += '    type(lhs) = type(rhs) = type(result)'
        return msg

    def C1(self):
        assert self.lhs.type == self.rhs.type, AddOp.msgC1()
        assert self.lhs.type == self.result.type, AddOp.msgC1()

    def verify_(self):
        C1(self)

class TestAddOp(unittest.TestCase):
    def test_add_op(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        addOp = AddOp(c1, c1, ty)
        expected = '%0 = "stablehlo.add"(%1, %1) : (tensor<1xsi64>, tensor<1xsi64>) -> tensor<1xsi64>'
        observed = str(addOp)
        assert expected == observed

    def test_bad_type(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        ty2 = TensorType(si64, [2])
        c1_0 = ConstantOp(val, ty2)
        addOp = AddOp(c1, c1_0, ty)
        with self.assertRaises(AssertionError, msg=AddOp.msgC1()):
            addOp.C1()


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "stablehlo.constant"
    value = prop_def(DenseIntOrFPElementsAttr)
    output = result_def(StableHLORankedTensor)

    def __init__(self, value, tensor_type):
        properties = { "value" : value }
        super().__init__(result_types=(tensor_type,),
                         properties=properties)
    def verify_(self):
        msg = f'''{ConstantOp.name} has constraint "type(value) = type(output)"'''
        assert self.value.type == self.output.type, msg

class TestConstantOp(unittest.TestCase):
    def test_constant_op(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        expected = """%0 = "stablehlo.constant"() <{"value" = dense<1> : tensor<1xsi64>}> : () -> tensor<1xsi64>"""
        observed = str(ConstantOp(val, ty))
        assert expected == observed

    def test_raises_error(self):
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        expected = """%0 = "stablehlo.constant"() <{"value" = dense<1> : tensor<1xsi64>}> : () -> tensor<1xsi64>"""
        ty2 = TensorType(si64, [2])
        op = ConstantOp(val, ty2)
        with self.assertRaises(AssertionError):
            op.verify_()

if "__main__" == __name__:
    unittest.main()
