import re
from typing import List, Tuple, Dict

DEFAULT_REGEX = re.compile(r".+=.+")


class SklearnObject:
    def __init__(self, name, file_name, line_nr, options):
        self.name: str = name
        self.file_name: str = file_name
        self.line_nr: int = line_nr
        self.options: List[Tuple] = options   


def get_key_value_pairs(data: Dict) -> List[Tuple]:
    return [(key, value) for key, value in data.items()]


def check_for_default_values(module_name: str, option_data: List[Tuple], sklearn_data: Dict) -> List[Tuple]:
    options = []

    for key in sklearn_data:
        if key["name"] == module_name:
            params = key["params"]
            for option in option_data:
                for param in params:
                    if option[0] == param[0]:
                        default = param[1]
                        if DEFAULT_REGEX.search(default):
                            default_parts = default.split("=")
                            default_value = default_parts[1]
                            if default_value == option[1]:
                                options.append((option[0], option[1], "default"))
                            else:
                                options.append((option[0], option[1], "custom"))
                        else:
                            options.append((option[0], option[1], "required"))
                    else: 
                        options.append((option[0], option[1], "unknown"))

    return options
