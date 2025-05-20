from abc import ABC, abstractmethod
from pandas import DataFrame

class LightcurveReprocessor(ABC):

    @classmethod
    @abstractmethod
    def can_process(cls, metadata) -> bool: ...
    
    @abstractmethod
    def reprocess(self, lcurve:DataFrame, **kwargs) -> DataFrame: ...


class ReprocessFactory:
    _processors = []
    
    @classmethod
    def register(cls, processor):
        cls._processors.append(processor)
    
    @classmethod
    def get(cls, data=None, metadata=None) -> LightcurveReprocessor:
        for processor in cls._processors:
            if processor.can_process(metadata):
                return processor()
        raise ValueError("No suitable processor found")