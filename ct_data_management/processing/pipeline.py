from abc import ABC, abstractmethod


class PipelinePart(ABC):
    @abstractmethod
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        pass


class PipelineStack(PipelinePart):
    def __init__(self, pipeline_parts):
        self._pipeline_parts = pipeline_parts

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        for pp in self._pipeline_parts:
            data, params = pp(data, params)
        return data, params
