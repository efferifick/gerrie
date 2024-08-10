import unittest
from typing import Optional

# StableHLO is an operation set for high-level operations (HLO) in machine learning (ML) models. 
# StableHLO works as a portability layer between different ML frameworks and ML compilers:
# ML frameworks that produce StableHLO programs are compatible with ML compilers that consume StableHLO programs.
#
# Our goal is to simplify and accelerate ML development by creating more interoperability between various ML frameworks (such as TensorFlow, JAX and PyTorch) and ML compilers (such as XLA and IREE).
# Towards that end, this document provides a specification for the StableHLO programming language.
#
# This specification contains three major sections. 
# First, the Programs section describes the structure of StableHLO programs which consist of StableHLO functions which themselves consist of StableHLO ops.
# Within that structure, the Ops section specifies the semantics of individual ops.
# The Execution section provides semantics for all these ops executing together within a program.
# Finally, the Notation section discusses the notation used throughout the specification.
#
# To view the spec from a previous release of StableHLO, open the repo at the tagged release of interest.
# For example, the StableHLO v0.19.0 Spec.
# To view changes that occurred at each minor version bump of StableHLO, refer to the version log in VhloDialect.td.
#
# Programs
#
#   Program ::= {Func}

from xdsl.dialects.builtin import ModuleOp  # as Program

# StableHLO programs consist of an arbitrary number of StableHLO functions.
# Below is an example program with a function @main which has 3 inputs (%image, %weights and %bias) and 1 output. 
# The body of the function has 6 ops.
# 
#   func.func @main(
#    %image: tensor<28x28xf32>,
#    %weights: tensor<784x10xf32>,
#     %bias: tensor<1x10xf32>
#   ) -> tensor<1x10xf32> {
#    %0 = "stablehlo.reshape"(%image) : (tensor<28x28xf32>) -> tensor<1x784xf32>
#    %1 = "stablehlo.dot"(%0, %weights) : (tensor<1x784xf32>, tensor<784x10xf32>) -> tensor<1x10xf32>
#    %2 = "stablehlo.add"(%1, %bias) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
#    %3 = "stablehlo.constant"() {value = dense<0.0> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
#    %4 = "stablehlo.maximum"(%2, %3) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
#    "func.return"(%4): (tensor<1x10xf32>) -> ()
#  }
#
# Functions
#
#   Func        ::= 'func' '.' 'func' FuncId FuncInputs FuncOutputs '{' FuncBody '}'
#   FuncInputs  ::= '(' [FuncInput {',' FuncInput}] `)`
#   FuncInput   ::= ValueId ':' ValueType
#   FuncOutputs ::= ['->' FuncOutput, {',' FuncOutput}]
#   FuncOutput  ::= ValueType
#   FuncBody    ::= {Op}

from xdsl.dialects.func import FuncOp  # as Func

# StableHLO functions (which are also called named functions) have an identifier, inputs/outputs and a body. 
# In the future, we are planning to introduce additional metadata for functions to achieve better compatibility with HLO (#425, #626, #740, #744).
#
# Identifiers
#
#   FuncId  ::= '@' letter {letter | digit}
#   ValueId ::= '%' digit {digit}
#             | '%' letter {letter | digit}
#   letter  ::= 'a' | ... | 'z' | 'A' | ... | 'Z' | '_'
#   digit   ::= '0' | ... | '9'
#
# StableHLO identifiers are similar to identifiers in many programming languages, with two peculiarities
# 1) all identifiers have sigils which distinguish different kinds of identifiers,
# 2) value identifiers can be completely numeric to simplify generation of StableHLO programs.
#
# Types
#
# StableHLO types are categorized into value types (which are also called first-class types) which represent StableHLO values and non-value types which describe other program elements.
# StableHLO types are similar to types in many programming languages, with the main peculiarity being StableHLO's domain-specific nature which results in some unusual outcomes (e.g. scalar types are not value types).
#
#   TensorType ::= 'tensor' '<' Shape TensorElementType '>'
#   Shape ::= {DimensionSize 'x'}
#   DimensionSize ::= digit {digit} | '?'

from xdsl.dialects.builtin import TensorType as BaseTensorType

# Tensor types represent tensors, i.e. multidimensional arrays.
# They have a shape and an element type, where a shape represents non-negative or unknown dimension sizes in the ascending order of the corresponding dimensions (which are also called axes) numbered from 0 to R-1
# The number of dimensions R is called rank. 
# For example, tensor<2x3xf32> is a tensor type with shape 2x3 and element type f32.
#  It has two dimensions (or, in other words, two axes) - 0th dimension and 1st dimension - whose sizes are 2 and 3. Its rank is 2.
#
# Shapes can be partially or completely unknown (dynamic), e.g. tensor<?x2xf64> is partially unknown and tensor<?x?xf64> is completely unknown
# Dynamic dimension sizes are represented using a ?. Shapes cannot be unranked.
#
# In the future, we are planning to explore extending tensor types beyond dimension sizes and element types, for example, to include layouts (#629) and sparsity (#1078).

from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition, ParameterDef
from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntegerAttr, IntegerType, FloatAttr, AnyFloat as FloatType, NoneAttr, Signedness
from xdsl.utils.exceptions import VerifyException

def min_value(type: IntegerType) -> int:
    width = type.width.data
    signedness = type.signedness
    is_signed = signedness == Signedness.SIGNED
    exponent = width - 1 if is_signed else width
    value = -(2 ** exponent) if is_signed else 0
    return value

def max_value(type: IntegerType) -> int:
    width = type.width.data
    signedness = type.signedness
    is_signed = signedness == Signedness.SIGNED
    exponent = width - 1 if is_signed else width
    value = (2 ** exponent) - 1
    return value

# Quantized element types represent integer values of a storage type in the range from storage_min to storage_max (inclusive) that correspond to floating-point values of an expressed type.
# For a given integer value i, the corresponding floating-point value f can be computed as f = (i - zero_point) * scale, where scale and zero_point are called quantization parameters.
# The storage_min and storage_max are optional in the grammar, but have default values of min_value(storage_type) and max_value(storage_type) respectively.

@irdl_attr_definition
class QuantizedTensorElementType(ParametrizedAttribute, TypeAttribute):
    name = "quant.uniform"

    storage_type: ParameterDef[IntegerType]
    storage_min: ParameterDef[IntegerAttr]
    storage_max: ParameterDef[IntegerAttr]
    expressed_type: ParameterDef[FloatType]
    quantization_dimension: ParameterDef[IntegerAttr|NoneAttr]
    scales: ParameterDef[ArrayAttr[FloatAttr]]
    zero_points: ParameterDef[ArrayAttr[IntegerAttr]]

    def __init__(self
                 , storage_type: IntegerType
                 , expressed_type: FloatType
                 , scales: ArrayAttr[FloatAttr]
                 , zero_points: ArrayAttr[IntegerAttr]
                 , storage_min: Optional[IntegerAttr] = None
                 , storage_max: Optional[IntegerAttr] = None
                 , quantization_dimension: Optional[IntegerAttr|NoneAttr] = None):
        storage_min = self.get_storage_min(storage_type, storage_min)
        storage_max = self.get_storage_max(storage_type, storage_max)
        quantization_dimension = self.get_quantization_dimension(quantization_dimension)
        params = (storage_type
                  , storage_min
                  , storage_max
                  , expressed_type
                  , quantization_dimension
                  , scales
                  , zero_points)
        super().__init__(params)


    @staticmethod
    def get_storage_min(storage_type: IntegerType, user_storage_min: Optional[IntegerAttr] = None) -> IntegerAttr:
        # The storage_min is optional in the grammar.
        if user_storage_min is not None:
            return user_storage_min

        # but has default value of min_value(storage_type)
        storage_min = min_value(storage_type)
        return IntegerAttr(storage_min, storage_type)

    @staticmethod
    def get_storage_max(storage_type: IntegerType, user_storage_max: Optional[IntegerAttr] = None) -> IntegerAttr:
        # The storage_min is optional in the grammar.
        if user_storage_max is not None:
            return user_storage_max

        # but has default value of max_value(storage_type)
        storage_max = max_value(storage_type)
        return IntegerAttr(storage_max, storage_type)

    @staticmethod
    def get_quantization_dimension(user_dimension: Optional[IntegerAttr|NoneAttr]):
        match user_dimension:
            case None:
                return NoneAttr()
        return user_dimension

    @staticmethod
    def C1(storage_min, storage_type):
        """(C1) type(storage_min) = storage_type."""
        is_satisfied = storage_min.type == storage_type
        err_msg = "Constrain C1 type(storage_min) = storage_type is not satisfied"
        match is_satisfied:
            case False:
                raise VerifyException(err_msg)

    @staticmethod
    def C2(storage_max, storage_type):
        """(C2) type(storage_min) = storage_type."""
        is_satisfied = storage_max.type == storage_type
        err_msg = "Constrain C2 type(storage_max) = storage_type is not satisfied"
        match is_satisfied:
            case False:
                raise VerifyException(err_msg)

    @staticmethod
    def C3(storage_type, storage_min, storage_max):
        """(C3) min_value(storage_type) <= storage_min < storage_max <= max_value(storage_type)"""
        storage_min = storage_min.value.data
        storage_max = storage_max.value.data
        if not (min_value(storage_type) <= storage_min < storage_max <= max_value(storage_type)):
            err_msg = """(C3) min_value(storage_type) <= storage_min < storage_max <= max_value(storage_type)"""
            raise VerifyException(err_msg)

    @staticmethod
    def C4(scales, expressed_type):
        for scale in scales.data:
            if scale.type != expressed_type:
                raise VerifyException("C4")

    @staticmethod
    def C5(scales):
        for scale in scales.data:
            if not (0.0 < scale.value.data):
                raise VerifyException("C5")

    @staticmethod
    def C6(scales):
        from math import isfinite
        for scale in scales.data:
            scale = scale.value.data
            if not (isfinite(scale)):
                raise VerifyException("C6")

    @staticmethod
    def C7(storage_min, storage_max, zero_points):
        storage_min = storage_min.value.data
        storage_max = storage_max.value.data
        zero_points = zero_points.data
        for zero_point in zero_points:
            zero_point = zero_point.value.data
            if not (storage_min <= zero_point <= storage_max):
                raise VerifyException("C7")

    @staticmethod
    def C8(zero_points, storage_type):
        zero_points = zero_points.data
        for zero_point in zero_points:
            if not (zero_point.type == storage_type):
                raise VerifyException("C8")

    @staticmethod
    def C9(scales, zero_points):
        scales = scales.data
        zero_points = zero_points.data
        if not (len(scales) == len(zero_points)):
            raise VerifyException("C9")

    @staticmethod
    def C10(quantization_dimension, scales):
        is_empty = isinstance(quantization_dimension, NoneAttr)
        if is_empty:
            scales = scales.data
            if not (len(scales) == 1):
                raise VerifyException("C10")

    @staticmethod
    def C11(quantization_dimension):
        is_empty = isinstance(quantization_dimension, NoneAttr)
        if not is_empty:
            quantization_dimension = quantization_dimension.value.data
            if not (0 <= quantization_dimension):
                raise VerifyException("C11")

    def verify_(self):
        self.C1(self.storage_min, self.storage_type)
        self.C2(self.storage_max, self.storage_type)
        self.C3(self.storage_type, self.storage_min, self.storage_max)
        self.C4(self.scales, self.expressed_type)
        self.C5(self.scales)
        self.C6(self.scales)
        self.C7(self.storage_min, self.storage_max, self.zero_points)
        self.C8(self.zero_points, self.storage_type)
        self.C9(self.scales, self.zero_points)
        self.C10(self.quantization_dimension, self.scales)
        self.C11(self.quantization_dimension)

    # TODO: parse_parameters
    # TODO: print_parameters

#   QuantizedTensorType ::= 'tensor' '<' Shape QuantizedTensorElementType '>'
#   QuantizedTensorElementType ::= '!quant.uniform' '<'
#                     QuantizationStorageType
#                     ['<' QuantizationStorageMin ':' QuantizationStorageMax '>']
#                     ':' QuantizationExpressedType
#                     [':' QuantizationDimension]
#                     ',' QuantizationParameters '>'
#   QuantizationStorageType ::= IntegerType
#   QuantizationStorageMin ::= IntegerConstant
#   QuantizationStorageMax ::= IntegerConstant
#   QuantizationExpressedType ::= FloatType
#   QuantizationDimension ::= IntegerConstant
#   QuantizationParameters ::= QuantizationParameter
#                            | '{' QuantizationParameter {',' QuantizationParameter} '}'
#   QuantizationParameter ::= QuantizationScale ':' QuantizationZeroPoint
#   QuantizationScale ::= FloatConstant
#   QuantizationZeroPoint ::= IntegerConstant

class TestC1(unittest.TestCase):

    def test_C1_passes(self):
        from xdsl.dialects.builtin import i16
        values = [IntegerAttr(2, i16)]
        types = [i16]
        for value, type in zip(values, types):
            with self.subTest(value=value, type=type):
                QuantizedTensorElementType.C1(value, type)

    def test_C1_errors(self):
        from xdsl.dialects.builtin import i16, i32
        values = [IntegerAttr(2, i16)]
        types = [i32]
        for value, type in zip(values, types):
            with self.subTest(value=value, type=type):
                with self.assertRaises(VerifyException):
                    QuantizedTensorElementType.C1(value, type)

class TestC3(unittest.TestCase):

    def test_C3_passes(self):
        from xdsl.dialects.builtin import i16
        min = IntegerAttr(0, i16)
        max = IntegerAttr(1, i16)
        QuantizedTensorElementType.C3(i16, min, max)

    def test_C3_min_lower(self):
        from xdsl.dialects.builtin import i16,IntegerType
        si16 = IntegerType(16, Signedness.SIGNED)
        min = IntegerAttr(-1, si16)
        max = IntegerAttr(1, si16)
        with self.assertRaises(VerifyException):
            QuantizedTensorElementType.C3(i16, min, max)

    def test_C3_min_larger_than_max(self):
        from xdsl.dialects.builtin import i16
        min = IntegerAttr(1, i16)
        max = IntegerAttr(0, i16)
        with self.assertRaises(VerifyException):
            QuantizedTensorElementType.C3(i16, min, max)

class TestC4(unittest.TestCase):

    def test_C4_error(self):
        from xdsl.dialects.builtin import f64, f32
        scales = ArrayAttr([FloatAttr(0.0, f32), FloatAttr(2.0, f32)])
        with self.assertRaises(VerifyException):
            QuantizedTensorElementType.C4(scales, f64)

class TestC6(unittest.TestCase):
    def test_C6_error(self):
        from xdsl.dialects.builtin import f32
        scales = ArrayAttr([FloatAttr(float("inf"), f32), FloatAttr(2.0, f32)])
        with self.assertRaises(VerifyException):
            QuantizedTensorElementType.C6(scales)

class TestQuantizedTensorElementType(unittest.TestCase):
    def test_simple_initialization(self):
        from xdsl.dialects.builtin import i16, f64
        scales = ArrayAttr([FloatAttr(1.0, f64)])
        zero_points = ArrayAttr([IntegerAttr(0, i16)])
        q = QuantizedTensorElementType(i16, f64, scales, zero_points)
        q.verify_()

# Quantized tensor types represent tensors with quantized elements.
# These tensors are exactly the same as regular tensors, except that their elements have quantized element types, instead of regular element types.
# In quantized tensors, quantization can be per-tensor, meaning, having one scale and zero_point for the entire tensor or can be per-axis, meaning, having multiple scales and zero_points, one pair per slice of a particular dimension quantization_dimension.
# More formally, in a tensor t with per-axis quantization, there are dim(t, quantization_dimension) slices of the quantization_dimension: t[:, ..., 0, ..., :], t[:, ..., 1, ..., :], etc.
# All elements in the ith slice use scales[i] and zero_points[i] as their quantization parameters.

from collections.abc import Iterable
from typing import Generic
from xdsl.ir import Attribute, AttributeCovT
from xdsl.dialects.builtin import ContainerType, ShapedType

@irdl_attr_definition
class TensorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "tensor"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]
    encoding: ParameterDef[Attribute]
    quantized_tensor_element_type: ParameterDef[QuantizedTensorElementType|NoneAttr]

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
        encoding: Attribute = NoneAttr(),
        quantized_tensor_element_type: Attribute = NoneAttr(),
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__([shape, element_type, encoding, quantized_tensor_element_type])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

@irdl_attr_definition
class TokenType(ParametrizedAttribute, TypeAttribute):
    name = "stablehlo.token"

@irdl_attr_definition
class TupleType(ParametrizedAttribute, TypeAttribute):
    name = "stablehlo.tuple"
    element_types: ParameterDef[ArrayAttr[TypeAttribute]]

    def __init__(self, element_types):
        super().__init__((element_types,))

    # TODO: parse / print


if __name__ == '__main__':
    unittest.main()
