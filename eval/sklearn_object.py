from typing import List, Tuple

class SklearnObject:
    def __init__(self, name, file_name, line_nr, options):
        self.name: str = name
        self.file_name: str = file_name
        self.line_nr: int = line_nr
        self.options: List[Tuple] = options   

