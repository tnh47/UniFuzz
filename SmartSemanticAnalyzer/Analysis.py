import re
import json
import subprocess
import logging
import os
import argparse
from google import genai  # Gemini API

from slither.slither import Slither
from slither.core.expressions import Identifier, TypeConversion, AssignmentOperation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_solc_version(contract_path):
    with open(contract_path, "r") as f:
        content = f.read()
        match = re.search(r'pragma\s+solidity\s+([^;]+);', content)
        if match:
            raw_version = match.group(1).strip()
            version_match = re.search(r'(\d+\.\d+\.\d+)', raw_version)
            if version_match:
                return version_match.group(1)
    return "unknown"

def extract_slither_data(contract_path, solc_path):
    if os.path.exists("slither_output.json"):
        os.remove("slither_output.json")

    result = subprocess.run(
        ["slither", contract_path, "--json", "slither_output.json", "--solc", solc_path],
        capture_output=True, text=True
    )

    logging.info("Slither stdout:\n%s", result.stdout)
    logging.info("Slither stderr:\n%s", result.stderr)

    if os.path.exists("slither_output.json"):
        with open("slither_output.json", "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                logging.warning("Failed to parse JSON from Slither: %s", e)
                return None

        solc_version = extract_solc_version(contract_path)

        # Lấy tên contract chính
        contracts = data.get("results", {}).get("contracts", [])
        main_contract = contracts[-1].get("name") if contracts else None

        constructor_args = []
        if main_contract:
            try:
                constructor_args = analysis_main_contract_constructor(
                    contract_path, main_contract, solc_path=solc_path
                )
            except Exception as e:
                logging.warning("Constructor analysis failed: %s", e)

        # Gộp lại dữ liệu
        updated_data = {
            "solc_version": solc_version,
            **data,
            "constructor_args": constructor_args
        }

        with open("slither_output.json", "w") as file:
            json.dump(updated_data, file, indent=4)

        return updated_data
    else:
        logging.warning("slither_output.json was not found after execution.")
        return None


def extract_param_contract_map(conversion):
    if hasattr(conversion, 'expression') and isinstance(conversion.expression, Identifier):
        return conversion.expression.value.name, str(conversion.type)
    return None, None

def analysis_main_contract_constructor(file_path: str, _contract_name: str, sl: Slither = None, solc_path="/usr/bin/solc"):
    if sl is None:
        sl = Slither(file_path, solc=solc_path)
    contract = sl.get_contract_from_name(_contract_name)
    assert len(contract) == 1, "Expected a single contract with the given name"
    contract = contract[0]

    constructor = contract.constructor
    if constructor is None:
        return []

    res = []
    for p in constructor.parameters:
        if (hasattr(p.type, "type") and hasattr(p.type.type, "kind") and p.type.type.kind == "contract"):
            res.append((p.name, "contract", p.name, [p.type.type.name]))
        elif hasattr(p.type, "name"):
            if p.type.name != "address":
                res.append((p.name, p.type.name, "YA_DO_NOT_KNOW", ["YA_DO_NOT_KNOW"]))
            else:
                res.append((p.name, p.type.name, [p.name], []))
        else:
            return None

    for exps in constructor.expressions:
        if isinstance(exps, AssignmentOperation):
            exps_right = exps.expression_right
            exps_left = exps.expression_left
            if isinstance(exps_right, Identifier) and isinstance(exps_left, Identifier):
                for cst_param in res:
                    if isinstance(cst_param[2], list) and exps_right.value.name in cst_param[2]:
                        cst_param[2].append(exps_left.value.name)
            elif isinstance(exps_right, TypeConversion) and isinstance(exps_left, Identifier):
                param_name, param_map_contract_name = extract_param_contract_map(exps_right)
                if param_name and param_map_contract_name:
                    for cst_param in res:
                        if isinstance(cst_param[2], list) and param_name in cst_param[2]:
                            cst_param[3].append(param_map_contract_name)
        elif isinstance(exps, TypeConversion):
            param_name, param_map_contract_name = extract_param_contract_map(exps)
            if param_name and param_map_contract_name:
                for cst_param in res:
                    if isinstance(cst_param[2], list) and param_name in cst_param[2]:
                        cst_param[3].append(param_map_contract_name)

    ret = []
    for p_name, p_type, _, p_value in res:
        if p_type == "address" and len(p_value) == 0:
            p_value = ["YA_DO_NOT_KNOW"]
        p_value = list(set(p_value))
        assert len(p_value) == 1, "Expected exactly one inferred value per constructor parameter"
        ret.append({"name": p_name, "type": p_type, "value": p_value[0]})
    return ret

def extract_contract_info(slither_data, contract_path, solc_path):
    contract_info = {
        "contract_name": None,
        "file_path": contract_path,
        "solc_version": None,
        "depend_contracts": [],
        "functions": [],
        "constructor_args": [],
        "parameters": [],
        "functions_to_fuzz": [],
        "vulnerabilities": []
    }

    if not slither_data or "results" not in slither_data:
        return contract_info

    for contract in slither_data.get("results", {}).get("contracts", []):
        contract_info["contract_name"] = contract.get("name")
        contract_info["solc_version"] = slither_data.get("solc_version", "unknown")

        for dep in contract.get("dependencies", []):
            contract_info["depend_contracts"].append(dep.get("name"))

        for func in contract.get("functions", []):
            f_info = {
                "name": func.get("name"),
                "visibility": func.get("visibility"),
                "parameters": func.get("parameters", [])
            }
            contract_info["functions"].append(f_info)

            if func.get("name") == "constructor":
                for arg in func.get("parameters", []):
                    contract_info["constructor_args"].append({
                        "name": arg.get("name", "param"),
                        "type": arg.get("type", "unknown"),
                        "value": "0x123..." if "address" in arg.get("type", "") else "100"
                    })
            else:
                if func.get("visibility") in ["public", "external"]:
                    contract_info["functions_to_fuzz"].append(func.get("name"))

        for detector in slither_data.get("results", {}).get("detectors", []):
            vuln = {
                "name": detector.get("check"),
                "severity": detector.get("impact", "Medium"),
                "location": detector.get("elements", [{}])[0].get("source_mapping", {}).get("lines", ["unknown"])[0],
                "description": detector.get("description")
            }
            contract_info["vulnerabilities"].append(vuln)

    # Analyze constructor with Slither
    if contract_info["contract_name"]:
        try:
            constructor_details = analysis_main_contract_constructor(contract_path, contract_info["contract_name"], solc_path=solc_path)
            if constructor_details:
                contract_info["constructor_args"] = constructor_details
        except Exception as e:
            logging.warning("Constructor analysis error: %s", e)

    return contract_info

def analyze_with_gemini(prompt, api_key):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logging.error("Gemini API error: %s", e)
        return None

def main():
    parser = argparse.ArgumentParser(description="Smart Contract Analysis with Slither and Gemini")
    parser.add_argument("--api-key", help="Google API Key (or use GOOGLE_API_KEY env var)")
    parser.add_argument("--contract-path", required=True, help="Path to the Solidity smart contract")
    parser.add_argument("--solc-path", default="/usr/bin/solc", help="Path to solc binary")
    parser.add_argument("--output", default="analysis_output.txt", help="Output file for Gemini result")

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.error("Missing Gemini API key. Use --api-key or set GOOGLE_API_KEY.")
        return

    contract_path = args.contract_path
    solc_path = args.solc_path

    if not os.path.exists(contract_path):
        logging.error(f"Contract file not found at: {contract_path}")
        return

    slither_data = extract_slither_data(contract_path, solc_path)
    contract_info = extract_contract_info(slither_data, contract_path, solc_path)

    if slither_data:
        prompt = f"""
        You are a security analyst. The following is a JSON output from Slither's static analysis of a Solidity smart contract located at {contract_path}:

        {json.dumps(slither_data, indent=4)}

        Based on this data, return a concise JSON object with these keys only:
        - contract_name
        - file_path
        - solc_version
        - depend_contracts
        - constructor_args (each with: name, type, value)
        - functions (each with: name, visibility, parameters)
        - vulnerabilities (each with: name, severity, location, description)
        - functions_to_fuzz (list of function names that are public/external)

        Please return only JSON as output, without explanations.
        """
        gemini_response = analyze_with_gemini(prompt, api_key)
        if gemini_response:
            match = re.search(r"\{[\s\S]*\}", gemini_response)
            json_data = None
            if match:
                try:
                    json_data = json.loads(match.group())
                except json.JSONDecodeError as e:
                    logging.error(f"Gemini JSON parse error: {e}")

            with open(args.output, "w") as f:
                f.write(gemini_response)

            logging.info(f"Analysis saved to {args.output}")
            print("Gemini Analysis Result:")
            print(gemini_response)

if __name__ == "__main__":
    main()
