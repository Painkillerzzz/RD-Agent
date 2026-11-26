from __future__ import annotations

from rdagent.core.experiment import Experiment, FBWorkspace, Task


class ZooSpecTask(Task):
    """Rewrite the current best spec according to a hypothesis."""

    def __init__(self, name: str, description: str, base_code: str, stem: str) -> None:
        super().__init__(name=name, description=description)
        self.base_code = base_code
        self.stem = stem


class ZooSpecWorkspace(FBWorkspace):
    SPEC_FILE = "spec.py"

    @property
    def spec_code(self) -> str:
        return self.file_dict.get(self.SPEC_FILE, "")

    def execute(self, *args, **kwargs):  # the runner executes through the zoo evaluator
        raise NotImplementedError


class ZooSpecExperiment(Experiment[ZooSpecTask, ZooSpecWorkspace, ZooSpecWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = ZooSpecWorkspace()
        self.result: dict | None = None  # agent view of the evaluator result (DEV + VAL bucket + gates)
        self.check_log: str = ""
