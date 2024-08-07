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

from xdsl.dialects.builtin import TensorType

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
#
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
#
from xdsl.ir import Attribute, Data, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import ParameterDef, irdl_attr_definition
from xdsl.dialects.builtin import FloatAttr, IntegerType, IntAttr
from xdsl.dialects.builtin import AnyFloat as FloatType
from xdsl.dialects.builtin import i64, f64
from xdsl.dialects.builtin import Signedness

@irdl_attr_definition
class _QuantizationParameter(TypeAttribute, ParametrizedAttribute):
    name = "quant.param"
    quantization_expressed_type: ParameterDef[FloatType]
    quantization_scale: ParameterDef[FloatAttr]
    quantization_storage_type: ParameterDef[IntegerType]
    quantization_zero_point: ParameterDef[IntAttr]

    def __init__(self
                 , quantization_scale
                 , quantization_zero_point
                 , quantization_expressed_type = f64
                 , quantization_storage_type = i64):
        if isinstance(quantization_scale, float):
            quantization_scale = FloatAttr(quantization_scale, quantization_expressed_type)
        if isinstance(quantization_zero_point, int):
            quantization_zero_point = IntAttr(quantization_zero_point)
        params = [quantization_expressed_type, quantization_scale, quantization_storage_type, quantization_zero_point]
        super().__init__(params)

from xdsl.parser import AttrParser
from xdsl.printer import Printer

@irdl_attr_definition
class QuantizedTensorElementType(Data, TypeAttribute):
    name = "quant.uniform"
    quantization_storage_type: IntegerType
    quantization_storage_min: IntAttr
    quantization_storage_max: IntAttr
    quantization_expressed_type: FloatType
    quantized_dimension: IntAttr
    quantization_parameters: list[tuple[float, int]]


    def __init__(self
                 , quantization_parameters
                 , quantization_storage_type = i64
                 , quantization_storage_min = None
                 , quantization_storage_max = None
                 , quantization_expressed_type = f64
                 , quantized_dimension = None):

        assert isinstance(quantization_parameters, list)

        signedness = quantization_storage_type.signedness
        width = quantization_storage_type.width.data
        is_signed = signedness == Signedness.SIGNED
        signedwidth = width - 1 if signedness == Signedness.SIGNED else width

        if quantization_storage_min is None:
            quantization_storage_min = 2**signedwidth if is_signed else 0
        if quantization_storage_max is None:
            quantization_storage_max = 2**signedwidth - 1

        if isinstance(quantization_storage_min, int):
            quantization_storage_min = IntAttr(quantization_storage_min)
        if isinstance(quantization_storage_max, int):
            quantization_storage_max = IntAttr(quantization_storage_max)

        self.quantization_storage_type = quantization_storage_type
        self.quantization_storage_min = quantization_storage_min
        self.quantization_storage_max = quantization_storage_max
        self.quantization_expressed_type = quantization_expressed_type
        self.quantized_dimension = quantized_dimension
        self.quantization_parameters = quantization_parameters

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            quantization_storage_type = parser.parse_attribute()
        return [quantization_storage_type]
    
    def print_parameter(self, printer: Printer) -> None:
        def print_intattr_data(arg: IntAttr):
            printer.print_string(f"{arg.data}")

        def print_quantization_parameter(param: tuple[float, int]):
            printer.print_string(f"{param[0]} : {param[1]}")

        with printer.in_angle_brackets():
            printer.print_attribute(self.quantization_storage_type)
            with printer.in_angle_brackets():
                printer.print_list([self.quantization_storage_min, self.quantization_storage_max], print_intattr_data)
            printer.print_string(":")
            printer.print_attribute(self.quantization_expressed_type)
            if self.quantized_dimension:
                printer.print_string(":")
                printer.print_intattr_data(self.quantized_dimension)
            printer.print(",")
            if len(self.quantization_parameters) == 1:
                print_quantization_parameter(self.quantization_parameters[0])
            else:
                printer.print_string("{")
                printer.print_list(self.quantization_parameters, print_quantization_parameter)
                printer.print_string("}")

q = QuantizedTensorElementType([(0.0, 0), (1.0, 1)])
print(q)

del ParametrizedAttribute, TypeAttribute
del ParameterDef, irdl_attr_definition
del FloatAttr, IntegerType, IntAttr
del FloatType
del i64, f64



#
# Token types represent tokens, i.e. opaque values produced and consumed by some operations.
# Tokens are used for imposing execution order on operations as described in the Execution section.

from xdsl.ir import Data, TypeAttribute
from xdsl.irdl import irdl_attr_definition

@irdl_attr_definition
class TokenType(Data, TypeAttribute):
    name = "stablehlo.token"

    def __init__(self):
        super().__init__(None)

    @classmethod
    def parse_parameter(cls, parser): ...

    def print_parameter(self, printer): ...

