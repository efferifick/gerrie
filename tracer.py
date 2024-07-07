class Tracer:
    def __init__(self, ssaval):
        self.ssaval = ssaval

    @property
    def owner(self):
        return self.ssaval.owner

    @property
    def type(self):
        return self.ssaval.type

    def get_or_create_tracer(self, val): ...
    def cast(self, val, ty): ...

from stablehlo import AddOp

class StableHLOTracer(Tracer):
    def get_or_create_tracer(self, val):
        # All python values will be converted
        # to tensors.
        from py2ty import create_tracers
        match val:
            case StableHLOTracer(): return val
            case _: return create_tracers(val)

    def cast(self, val, ty):
        otherTy = val.type
        match otherTy == ty:
            case True: return val
        msg = "TODO"
        raise ValueError(msg)

    def __add__(self, other):
        other_tracer = self.get_or_create_tracer(other)
        casted = self.cast(other_tracer, self.type)
        otherval = casted.ssaval
        ssaret = AddOp(self.ssaval, otherval, self.type)
        return Tracer(ssaret.result)
