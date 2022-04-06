from typing import List, Tuple

class MachineLearningModule:
    def __init__(self, name, file_name, line_nr, options):
        self.name: str = name
        self.file_name: str = file_name
        self.line_nr: int = line_nr
        self.options: List[Tuple] = options


class SklearnModule(MachineLearningModule):
    def __init__(self, name, file_name, line_nr, options):
        super().__init__(name, file_name, line_nr, options)


class TensorflowModule(MachineLearningModule):
    def __init__(self, name, file_name, line_nr, options):
        super().__init__(name, file_name, line_nr, options)


class PytorchObject(MachineLearningModule):
    def __init__(self, name, file_name, line_nr, options):
        super().__init__(name, file_name, line_nr, options)
