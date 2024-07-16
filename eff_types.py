#   ####### #     # ######  #######  #####  
#      #     #   #  #     # #       #     # 
#      #      # #   #     # #       #       
#      #       #    ######  #####    #####  
#      #       #    #       #             # 
#      #       #    #       #       #     # 
#      #       #    #       #######  #####    

from xdsl.irdl import irdl_attr_definition
from xdsl.ir import ParametrizedAttribute, TypeAttribute

# Token types represent tokens, i.e. opaque values produced and consumed by
# some operations. Tokens are used for imposing execution order on operations
# as described in the Execution section.
@irdl_attr_definition
class TokenType(ParametrizedAttribute, TypeAttribute):
    name = "stablehlo.token"

# Tensor types represent tensors, i.e. multidimensional arrays.
from xdsl.dialects.builtin import TensorType

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

i2  = IntegerType(2,  Signedness.UNSIGNED)
i4  = IntegerType(4,  Signedness.UNSIGNED)
i8  = IntegerType(8,  Signedness.UNSIGNED)
i16 = IntegerType(16, Signedness.UNSIGNED)
i32 = IntegerType(32, Signedness.UNSIGNED)
i64 = IntegerType(64, Signedness.UNSIGNED)


ui2  = IntegerType(2,  Signedness.UNSIGNED)
ui4  = IntegerType(4,  Signedness.UNSIGNED)
ui8  = IntegerType(8,  Signedness.UNSIGNED)
ui16 = IntegerType(16, Signedness.UNSIGNED)
ui32 = IntegerType(32, Signedness.UNSIGNED)
ui64 = IntegerType(64, Signedness.UNSIGNED)


# Floating-point types can be one of the following: 
# bf16 type corresponding to the bfloat16 format described in BFloat16:
# The secret to high performance on Cloud TPUs.
from xdsl.dialects.builtin import BFloat16Type
from xdsl.dialects.builtin import bf16

# f16, f32 and f64 types corresponding to respectively
# binary16 ("half precision"), binary32 ("single precision")
# and binary64 ("double precision") formats described in the IEEE 754 standard.
from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type
from xdsl.dialects.builtin import f16, f32, f64


# Complex types represent complex values that have a real part
# and an imaginary part of the same element type. Supported
# complex types are complex<f32> (both parts are of type f32)
# and complex<f64> (both parts are of type f64).
from xdsl.dialects.builtin import ComplexType
complexf32 = ComplexType(f32)
complexf64 = ComplexType(f64)

