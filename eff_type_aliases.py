#   ####### #     # ######  #######       #    #       ###    #     #####  #######  #####
#      #     #   #  #     # #            # #   #        #    # #   #     # #       #     #
#      #      # #   #     # #           #   #  #        #   #   #  #       #       #
#      #       #    ######  #####      #     # #        #  #     #  #####  #####    #####
#      #       #    #       #          ####### #        #  #######       # #             #
#      #       #    #       #          #     # #        #  #     # #     # #       #     #
#      #       #    #       #######    #     # ####### ### #     #  #####  #######  #####

from typing import Annotated, TypeAlias

from eff_types import TokenType
from eff_types import IntegerType
from eff_types import i1
from eff_types import i2, i4, i8, i16, i32, i64
from eff_types import si2, si4, si8, si16, si32, si64
from eff_types import ui2, ui4, ui8, ui16, ui32, ui64
from eff_types import BFloat16Type, Float16Type, Float32Type, Float64Type
from eff_types import bf16, f16, f32, f64
from eff_types import ComplexType
from eff_types import complexf32, complexf64
from eff_types import TensorType

I1 = Annotated[IntegerType, i1]
StableHLOBoolean : TypeAlias = (I1)

I2  = Annotated[IntegerType, i2]
I4  = Annotated[IntegerType, i4]
I8  = Annotated[IntegerType, i8]
I16 = Annotated[IntegerType, i16]
I32 = Annotated[IntegerType, i32]
I64 = Annotated[IntegerType, i64]
StableHLOSignlessInteger : TypeAlias = (I2 | I4 | I8 | I32 | I64)

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

StableHLOFloat : TypeAlias = (BFloat16Type | Float16Type | Float32Type | Float64Type)

Complex32 = Annotated[ComplexType, f32]
Complex64 = Annotated[ComplexType, f64]

StableHLOComplex : TypeAlias = (Complex32 | Complex64)
StableHLOElementType : TypeAlias = (StableHLOBoolean | StableHLOSignlessInteger | StableHLOSignedInteger | StableHLOUnsignedInteger | StableHLOFloat | StableHLOComplex)

StableHLORankedTensor : TypeAlias = TensorType[StableHLOElementType]
StableHLOTensor : TypeAlias = StableHLORankedTensor 

