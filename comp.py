from config import *
from queue import Queue
from slither import Slither
from slither.core.declarations import Contract
from typing import Tuple, List
from slither.core.expressions import TypeConversion, Identifier, AssignmentOperation
from slither.core.solidity_types import UserDefinedType

logger = get_logger()


@logger.catch()
def analysis_depend_contract(file_path: str, _contract_name: str, _solc_version: str, _solc_path) -> (
        Tuple)[List, Slither]:
    res = set()  # Danh sách các hợp đồng cần triển khai
    sl = Slither(file_path, solc=_solc_path)
    to_be_deep_analysis = Queue()  # Danh sách các hợp đồng cần phân tích sâu
    to_be_deep_analysis.put(_contract_name)
    while not to_be_deep_analysis.empty():
        c = to_be_deep_analysis.get()
        contract = sl.get_contract_from_name(c)
        if len(contract) != 1:
            logger.warning("Theo lý thuyết, chỉ có thể tìm thấy một hợp đồng dựa trên tên")
            return [], sl
        contract = contract[0]
        # 1. Phân tích các biến trạng thái được ghi
        for v in contract.all_state_variables_written:
            if not v.initialized and isinstance(v.type, UserDefinedType) and hasattr(v.type, "type") and isinstance(
                    v.type.type, Contract):
                res.add(v.type.type.name)
                logger.debug("Phát hiện hợp đồng phụ thuộc thông qua phân tích biến trạng thái được ghi: {}".format(v.type.type.name))
        for f in contract.functions:
            # 2. Phân tích tham số của các hàm trong hợp đồng
            for p in f.parameters:
                if isinstance(p.type, UserDefinedType) and hasattr(p.type, "type") and isinstance(p.type.type,
                                                                                                  Contract):
                    res.add(p.type.type.name)
                    logger.debug("Phát hiện hợp đồng phụ thuộc thông qua phân tích tham số hàm: {}".format(p.type.type.name))
            # 3. Phân tích các biến được ghi, nếu là kiểu hợp đồng thì cũng cần triển khai
            for v in f.variables_written:
                if hasattr(v, "type") and isinstance(v.type, UserDefinedType) and hasattr(v.type,
                                                                                          "type") and isinstance(
                    v.type.type, Contract):
                    res.add(v.type.type.name)
                    logger.debug("Phát hiện hợp đồng phụ thuộc thông qua phân tích biến được ghi (cục bộ và trạng thái): {}".format(v.type.type.name))
        # 3. Phân tích mối quan hệ kế thừa trong hợp đồng, thêm vào hàng đợi phân tích
        for inherit in contract.inheritance:
            if inherit.name not in res:
                to_be_deep_analysis.put(inherit.name)
    if _contract_name in res:
        logger.debug("Hợp đồng chính được tìm thấy trong danh sách phụ thuộc, cần loại bỏ")
        res.remove(_contract_name)
    # 4. Kiểm tra bytecode của các hợp đồng phụ thuộc, loại bỏ các hợp đồng rỗng
    compilation_unit = sl.compilation_units[0].crytic_compile_compilation_unit
    for depend_c in res.copy():
        if compilation_unit.bytecode_runtime(depend_c) == "" or compilation_unit.bytecode_runtime(depend_c) == "":
            logger.debug(f"Bytecode của hợp đồng phụ thuộc {depend_c} rỗng, đã loại bỏ")
            res.remove(depend_c)

    logger.info("Các hợp đồng phụ thuộc: " + str(res) + ", Tổng số: " + str(len(sl.contracts)) + " hợp đồng, Cần triển khai: " + str(
        len(res)) + " hợp đồng")
    return list(res), sl


def analysis_main_contract_constructor(file_path: str, _contract_name: str, sl: Slither = None):
    if sl is None:
        sl = Slither(file_path, solc=SOLC_BIN_PATH)
    contract = sl.get_contract_from_name(_contract_name)
    assert len(contract) == 1, "Theo lý thuyết, chỉ có thể tìm thấy một hợp đồng dựa trên tên"
    contract = contract[0]
    logger.info(f"=== Bắt đầu phân tích constructor của hợp đồng {_contract_name} ===")
    
    # 1. Phân tích constructor trong hợp đồng
    constructor = contract.constructor
    if constructor is None:  # Không có constructor
        logger.info("Hợp đồng không có constructor")
        return []
    
    logger.info(f"Đã tìm thấy constructor, bắt đầu phân tích tham số")
    # 1. Lấy tất cả tham số của constructor, nếu tên không phải address thì đặt là YA_DO_NOT_KNOW, các tham số khác tạm thời khởi tạo dưới dạng list, list lưu luồng dữ liệu
    res = []
    for p in constructor.parameters:
        logger.info(f"\nPhân tích tham số: {p.name}")
        logger.info(f"  Kiểu dữ liệu: {p.type}")
        
        if (hasattr(p.type, "type") and hasattr(p.type.type, "kind") and p.type.type.kind == "contract"):
            logger.info(f"  Tham số {p.name} là kiểu hợp đồng: {p.type.type.name}")
            logger.info(f"  Giá trị mặc định: {p.name} (địa chỉ hợp đồng)")
            res.append((p.name, "contract", p.name, [p.type.type.name]))
        elif hasattr(p.type, "name"):
            if p.type.name != "address":
                logger.info(f"  Tham số {p.name} là kiểu thông thường: {p.type.name}")
                logger.info(f"  Giá trị mặc định: YA_DO_NOT_KNOW")
                res.append((p.name, p.type.name, "YA_DO_NOT_KNOW", ["YA_DO_NOT_KNOW"]))
            else:
                logger.info(f"  Tham số {p.name} là kiểu địa chỉ")
                logger.info(f"  Giá trị mặc định: {p.name} (địa chỉ)")
                res.append((p.name, p.type.name, [p.name], []))
        else:  # Có thể là mảng
            logger.warning(f"  Tham số {p.name} có thể là kiểu mảng, chưa hỗ trợ")
            return None
            
    logger.info("\nBắt đầu phân tích luồng dữ liệu bên trong constructor")
    # 2. Phân tích luồng dữ liệu bên trong constructor
    for exps in constructor.expressions:  # Phân tích các biểu thức bên trong constructor, xác định luồng dữ liệu đến các biến trạng thái
        if isinstance(exps, AssignmentOperation):
            exps_right = exps.expression_right
            exps_left = exps.expression_left
            logger.info(f"\nPhân tích biểu thức gán: {exps_left} = {exps_right}")
            if isinstance(exps_right, Identifier) and isinstance(exps_left, Identifier):
                for cst_param in res:
                    if isinstance(cst_param[2], list) and exps_right.value.name in cst_param[2]:
                        logger.info(f"  Phát hiện luồng dữ liệu: {exps_right.value.name} -> {exps_left.value.name}")
                        cst_param[2].append(exps_left.value.name)
            elif isinstance(exps_right, TypeConversion) and isinstance(exps_left, Identifier):
                param_name, param_map_contract_name = extract_param_contract_map(exps_right)
                if param_name is not None and param_map_contract_name is not None:
                    logger.info(f"  Phát hiện chuyển đổi hợp đồng: {param_name} -> {param_map_contract_name}")
                    for cst_param in res:
                        if isinstance(cst_param[2], list) and param_name in cst_param[2]:
                            cst_param[3].append(param_map_contract_name)
        elif isinstance(exps, TypeConversion):
            param_name, param_map_contract_name = extract_param_contract_map(exps)
            if param_name is not None and param_map_contract_name is not None:
                logger.info(f"  Phát hiện chuyển đổi hợp đồng: {param_name} -> {param_map_contract_name}")
                for cst_param in res:
                    if isinstance(cst_param[2], list) and param_name in cst_param[2]:
                        cst_param[3].append(param_map_contract_name)
                        
    # Chuyển đổi res
    ret = []
    logger.info("\n=== Kết quả phân tích tham số constructor ===")
    for p_name, p_type, _, p_value in res:
        if p_type == "address" and len(p_value) == 0:
            p_value = ["YA_DO_NOT_KNOW"]
        p_value = list(set(p_value))
        assert len(p_value) == 1, "Theo lý thuyết, mỗi tham số chỉ có một giá trị mong đợi"
        ret.append(f"{p_name} {p_type} {p_value[0]}")
        logger.info(f"Tham số: {p_name}")
        logger.info(f"  Kiểu: {p_type}")
        logger.info(f"  Giá trị: {p_value[0]}")
    
    logger.info("=== Kết thúc phân tích constructor ===")
    return ret


def extract_param_contract_map(exps: TypeConversion):
    inner_exp = exps.expression
    if isinstance(inner_exp, Identifier) \
            and isinstance(exps.type, UserDefinedType) \
            and hasattr(exps.type, "type") \
            and isinstance(exps.type.type, Contract):
        return inner_exp.value.name, exps.type.type.name
    else:
        return None, None
