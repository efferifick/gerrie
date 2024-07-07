class Tracer:
    def __init__(self, ssaval):
        self.ssaval = ssaval

    @property
    def owner(self):
        return self.ssaval.owner

    @property
    def type(self):
        return self.ssaval.type

from stablehlo import TensorType
from stablehlo import AbsOp, AddOp, AndOp, ConvertOp

class StableHLOTracer(Tracer):
    def get_or_create_tracer(self, val):
        # All python values will be converted
        # to tensors.
        match val:
            case StableHLOTracer(): return val
        raise ValueError("TODO")

    def cast(self, val, ty):
        otherTy = val.type
        match otherTy == ty:
            case True: return val
        raise ValueError("TODO")

    def unop(self, unop):
        ssaret = unop(self.ssaval, self.type)
        ssaret.verify_()
        return StableHLOTracer(ssaret.result)

    def binop(self, binop, other):
        other_tracer = self.get_or_create_tracer(other)
        casted = self.cast(other_tracer, self.type)
        otherval = casted.ssaval
        ssaret = binop(self.ssaval, otherval, self.type)
        ssaret.verify_()
        return StableHLOTracer(ssaret.result)

    def __abs__(self):
        return self.unop(AbsOp)

    def __and__(self, other):
        return self.binop(AndOp, other)

    def __add__(self, other):
        return self.binop(AddOp, other)

    def astype(self, dtype):
        from py2ty import convert_numpy_dtype_to_mlir_type
        e_ty = convert_numpy_dtype_to_mlir_type(dtype)
        old_type = self.type
        c_ty = TensorType(e_ty, old_type.shape)
        convertop = ConvertOp(self.ssaval, c_ty)
        convertop.verify_()
        return StableHLOTracer(convertop.result)

