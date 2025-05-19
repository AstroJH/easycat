from abc import ABC, abstractmethod

class LightcurveReprocessor(ABC):

    @classmethod
    @abstractmethod
    def can_process(cls, metadata) -> bool:
        pass
    
    @abstractmethod
    def reprocess(self, data, **kwargs): ...


class ReprocessFactory:
    _processors = []
    
    @classmethod
    def register_processor(cls, processor):
        cls._processors.append(processor)
    
    @classmethod
    def get_reprocessor(cls, data, metadata):
        for processor in cls._processors:
            if processor.can_process(metadata):
                return processor()
        raise ValueError("No suitable processor found")