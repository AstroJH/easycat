from .core import LightcurveReprocessor

class ZTFReprocessor(LightcurveReprocessor):
    @classmethod
    def can_process(cls, metadata):
        return metadata.get("telescope") == "ZTF"
    
    def reprocess(self, data, **kwargs):
        pass