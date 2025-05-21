import re
from math import atan


def get_constants(header_file: str) -> dict:

    with open(header_file, "r") as file:
        constants = {}
        for line in file:
            define = re.match(r"#define\s+(\w+)\s+(.*)", line)
            if define is None:
                continue
            # Remove comments
            line = define.group().split("//")[0]
            # Get key-value pairs
            line = line.removeprefix("#define ").strip()
            args = line.split(maxsplit=1)
            if len(args) != 2:
                continue
            key, value = args
            # Ignore type conversion
            value = value.removeprefix("(double)")
            # Perform atan
            # print(value)
            if value.find("atan") >= 0:
                v = float(re.findall(r"atan\((.*)\)", value)[0])
                value = re.sub(r"atan\(.*\)", f"{atan(v)}", value)
            try:
                # Convert to formated string
                value = re.sub(r"([a-zA-Z_]\w*)", r"{\1}", value).format(**constants)
            except Exception:
                pass
            try:
                # Get value
                value = eval(value)
            except Exception:
                pass
            # Interpret as string
            if isinstance(value, str):
                value = re.sub(r"[\"\']", "", value)
            constants[key] = value
    return constants
