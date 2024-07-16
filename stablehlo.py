
#   ####### ######  ####### ######     #    ####### ### ####### #     #  #####
#   #     # #     # #       #     #   # #      #     #  #     # ##    # #     #
#   #     # #     # #       #     #  #   #     #     #  #     # # #   # #
#   #     # ######  #####   ######  #     #    #     #  #     # #  #  #  #####
#   #     # #       #       #   #   #######    #     #  #     # #   # #       #
#   #     # #       #       #    #  #     #    #     #  #     # #    ## #     #
#   ####### #       ####### #     # #     #    #    ### ####### #     #  #####

from collections.abc import Sequence
import unittest

from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation, Operand, OpResult, VarOperand, VarRegion
from xdsl.irdl import irdl_op_definition, operand_def, prop_def, region_def, result_def, var_operand_def, var_region_def, var_result_def
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr
from xdsl.traits import IsTerminator

from eff_type_aliases import TypeAlias

from eff_type_aliases import TensorType
from eff_type_aliases import I1
from eff_type_aliases import SI2, SI4, SI8, SI16, SI32, SI64
from eff_type_aliases import UI2, UI4, UI8, UI16, UI32, UI64
from eff_type_aliases import BFloat16Type
from eff_type_aliases import Float16Type, Float32Type, Float64Type

from eff_type_aliases import TokenType
from eff_type_aliases import StableHLOBoolean
from eff_type_aliases import StableHLOSignedInteger, StableHLOUnsignedInteger
from eff_type_aliases import StableHLOFloat
from eff_type_aliases import StableHLOComplex
from eff_type_aliases import StableHLOElementType
from eff_type_aliases import StableHLOTensor


@irdl_op_definition
class AbsOp(IRDLOperation):
    name = "stablehlo.abs"
    operand = operand_def(StableHLOTensor)
    result = result_def(StableHLOTensor)

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

class TestAbsOp(unittest.TestCase):
    def test_abs_op(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        one = ConstantOp(val, ty)
        absop = AbsOp(one, ty)
        expected = """%0 = "stablehlo.abs"(%1) : (tensor<1xsi64>) -> tensor<1xsi64>"""
        observed = str(absop)
        assert expected == observed

    def test_constraint_same_shapes(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        one = ConstantOp(val, ty)
        ty2 = TensorType(si64, [2])
        absop = AbsOp(one, ty2)
        with self.assertRaises(AssertionError):
            absop.verify_()

    def test_constraint_same_etype(self):
        from eff_types import si32, si64
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
    lhs = operand_def(StableHLOTensor)
    rhs = operand_def(StableHLOTensor)
    result = result_def(StableHLOTensor)

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
        self.C1()

class TestAddOp(unittest.TestCase):
    def test_add_op(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        addOp = AddOp(c1, c1, ty)
        expected = '%0 = "stablehlo.add"(%1, %1) : (tensor<1xsi64>, tensor<1xsi64>) -> tensor<1xsi64>'
        observed = str(addOp)
        assert expected == observed

    def test_bad_type(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        ty2 = TensorType(si64, [2])
        c1_0 = ConstantOp(val, ty2)
        addOp = AddOp(c1, c1_0, ty)
        with self.assertRaises(AssertionError, msg=AddOp.msgC1()):
            addOp.C1()

@irdl_op_definition
class AfterAllOp(IRDLOperation):
    name = "stablehlo.after_all"
    inputs = var_operand_def(TokenType)
    result = result_def(TokenType)

    def __init__(self, inputs, result):
        super().__init__(operands=[inputs],
                         result_types=(result,))

class TestAfterAllOp(unittest.TestCase):
    def test_after_all(self):
        from xdsl.ir import Block
        tokentype = TokenType()
        block = Block([], arg_types=[tokentype, tokentype])
        tokenvals = block.args
        afterallop = AfterAllOp(tokenvals, tokentype)
        observed = str(afterallop)
        expected = '%0 = "stablehlo.after_all"(%1, %2) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token'
        assert observed == expected

ReplicaGroupsType : TypeAlias = TensorType[SI64]
@irdl_op_definition
class AllGatherOp(IRDLOperation):
    name = "stablehlo.all_gather"
    inputs: VarOperand = var_operand_def(StableHLOTensor)
    all_gather_dim = prop_def(SI64)
    replica_groups = prop_def(ReplicaGroupsType)
    channel_id = prop_def(SI64)
    use_global_device_ids = prop_def(I1)
    result = var_result_def(StableHLOTensor)

    def __init__(self, operands : Sequence[SSAValue], all_gather_dim, replica_groups, channel_id, use_global_device_ids, result):
        properties = {"all_gather_dim" : all_gather_dim,
                      "replica_groups" : replica_groups,
                      "channel_id" : channel_id,
                      "use_global_device_ids" : use_global_device_ids}
        super().__init__(operands=(operands,),
                         result_types=(result,),
                         properties=properties)

    @staticmethod
    def msgC1():
        return "0 <= all_gather_dim < rank(operands...)"

    def C1(self):
        """Missing verification"""

    def verify_(self):
        self.C1()

class TestAllGatherOp(unittest.TestCase):
    def test_all_gather_op(self):
        from eff_types import i1, si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        si_c1 = IntegerAttr(1, si64)
        i1_c1 = IntegerAttr(1, i1)
        all_gather_op = AllGatherOp([c1], si_c1, val, si_c1, i1_c1, [ty])

# ALL_REDUCE
# ALL_TO_ALL

TensorIntegerType : TypeAlias = TensorType[StableHLOBoolean | StableHLOSignedInteger | StableHLOUnsignedInteger]
@irdl_op_definition
class AndOp(IRDLOperation):
    name = "stablehlo.and"
    lhs : Operand = operand_def(TensorIntegerType)
    rhs : Operand = operand_def(TensorIntegerType)
    result : OpResult = result_def(TensorIntegerType)

    def __init__(self, lhs, rhs, result):
        super().__init__(operands=(lhs, rhs),
                         result_types=(result,))
    def verify_(self):
        assert self.lhs.type == self.rhs.type
        assert self.rhs.type == self.result.type

class TestAndOp(unittest.TestCase):
    def test_and_op(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        assert AndOp(c1, c1, ty)

TensorFloatAndComplex : TypeAlias = TensorType[StableHLOFloat | StableHLOComplex]
@irdl_op_definition
class Atan2Op(IRDLOperation):
    name = "stablehlo.atan2"
    lhs = operand_def(TensorFloatAndComplex)
    rhs = operand_def(TensorFloatAndComplex)
    result = result_def(TensorFloatAndComplex)

    def __init__(self, lhs, rhs, result):
        super().__init__(operands=(lhs, rhs),
                         result_types=(result,))
    def verify_(self):
        assert self.lhs.type == self.rhs.type
        assert self.rhs.type == self.result.type

class TestAtan2Op(unittest.TestCase):
    def test_atan2_op(self):
        from eff_types import f64
        ty = TensorType(f64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        assert Atan2Op(c1, c1, ty)

# BATCH_NORM_GRAD
# BATCH_NORM_INFERENCE
# BATCH_NORM_TRAINING

@irdl_op_definition
class BitcastConvertOp(IRDLOperation):
    name = "stablehlo.bitcast_convert"
    input = operand_def(StableHLOTensor)
    result = result_def(StableHLOTensor)

    def __init__(self, input, result):
        super().__init__(operands=(input,),
                         result_types=(result,))

class TestBitcastConvertOp(unittest.TestCase):
    def test_bitcast_convert_op(self):
        from eff_types import f64
        ty = TensorType(f64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        c1 = ConstantOp(val, ty)
        assert BitcastConvertOp(c1, ty)

# BROADCAST_IN_DIM

TensorSI32 : TypeAlias = TensorType[SI32]
@irdl_op_definition
class CaseOp(IRDLOperation):
    name = "stablehlo.case"
    index = operand_def(TensorSI32)
    branches : VarRegion = var_region_def("single_block")
    results = var_result_def(StableHLOTensor | TokenType)

    def __init__(self, index, branches, results):
        super().__init__(operands=(index,),
                         result_types=(results,),
                         regions=(branches,))

@irdl_op_definition
class ConvertOp(IRDLOperation):
    name = "stablehlo.convert"
    input = operand_def(StableHLOTensor)
    result = result_def(StableHLOTensor)

    def __init__(self, input, result):
        super().__init__(operands=(input,),
                         result_types=(result,))

    def verify_(self):
        assert self.input.type.shape == self.result.type.shape

TensorI1 : TypeAlias = TensorType[I1]
@irdl_op_definition
class IfOp(IRDLOperation):
    name = "stablehlo.if"
    pred = operand_def(TensorI1)
    true_branch = region_def("single_block")
    false_branch = region_def("single_block")
    outputs = var_result_def(StableHLOTensor | TokenType)

    def __init__(self, pred, true_branch, false_branch, results):
        super().__init__(operands=(pred,),
                         result_types=(results,),
                         regions=(true_branch, false_branch),)

@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "stablehlo.return"
    input = var_operand_def(StableHLOTensor)
    traits = frozenset([IsTerminator()])

    def __init__(self, input):
        super().__init__(operands=(input,))

@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "stablehlo.constant"
    value = prop_def(DenseIntOrFPElementsAttr)
    output = result_def(StableHLOTensor)

    def __init__(self, value, tensor_type):
        properties = { "value" : value }
        super().__init__(result_types=(tensor_type,),
                         properties=properties)
    def verify_(self):
        msg = f'''{ConstantOp.name} has constraint "type(value) = type(output)"'''
        assert self.value.type == self.output.type, msg

class TestConstantOp(unittest.TestCase):
    def test_constant_op(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        expected = """%0 = "stablehlo.constant"() <{"value" = dense<1> : tensor<1xsi64>}> : () -> tensor<1xsi64>"""
        observed = str(ConstantOp(val, ty))
        assert expected == observed

    def test_raises_error(self):
        from eff_types import si64
        ty = TensorType(si64, [1])
        val = DenseIntOrFPElementsAttr.from_list(ty, [1])
        expected = """%0 = "stablehlo.constant"() <{"value" = dense<1> : tensor<1xsi64>}> : () -> tensor<1xsi64>"""
        ty2 = TensorType(si64, [2])
        op = ConstantOp(val, ty2)
        with self.assertRaises(AssertionError):
            op.verify_()

#   ######  ###    #    #       #######  #####  #######
#   #     #  #    # #   #       #       #     #    #
#   #     #  #   #   #  #       #       #          #
#   #     #  #  #     # #       #####   #          #
#   #     #  #  ####### #       #       #          #
#   #     #  #  #     # #       #       #     #    #
#   ######  ### #     # ####### #######  #####     #

from xdsl.ir import Dialect

StableHLO = Dialect(
        "stablehlo",
        [
            AbsOp,
            AddOp,
            AfterAllOp,
            AllGatherOp,
            AndOp,
            Atan2Op,
            BitcastConvertOp,
            CaseOp,
            ConstantOp,
            ConvertOp,
            ReturnOp,
        ]
    )

if "__main__" == __name__:
    unittest.main()
