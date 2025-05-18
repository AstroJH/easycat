from .core import LightcurveReprocessor
import numpy as np

class WiseReprocessor(LightcurveReprocessor):
    @classmethod
    def can_process(cls, metadata):
        return metadata.get("telescope") == "WISE"
    
    def reprocess(self, data, **kwargs):
        pass

    def criteria_basic(self, lcurve):
        na = lcurve["na"]
        nb = lcurve["nb"]
        saa_sep = lcurve["saa_sep"]
        qi_fact = lcurve["qi_fact"]
        qual_frame = lcurve["qual_frame"]

        w1rchi2 = lcurve["w1rchi2"]
        w2rchi2 = lcurve["w2rchi2"]

        cond1 = ((qual_frame > 0) | (qual_frame == -1)) & (qi_fact == 1)
        cond2 = (na == 0) & (nb <= 2)
        cond3 = (saa_sep > 0)
        cond4 = lcurve["moon_masked"].apply(lambda s: s[:2]) == "00"
        cond5 = lcurve["cc_flags"].apply(lambda s: s[:2]) == "00"
        cond6 = (w1rchi2 < 5) & (w2rchi2 < 5)

        return lcurve[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
    

    def filter_missing(self, lcurve,
                       fields=["mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag", "na", "nb"],
                       missing_value=-1):
        """ Filters missing values for specified fields (missing values are represented by -1 by default). """

        mask = np.full(len(lcurve), True)

        for f in fields:
            mask = mask & (lcurve[f] != missing_value)
        
        return lcurve[mask]
    
    def filter_uncertainty(self, lcurve, w1threshold, w2threshold):
        w1sigmag = lcurve["w1sigmag"]
        w2sigmag = lcurve["w2sigmag"]

        mask = (w1sigmag <= w1threshold) & (w2sigmag <= w2threshold)
        
        return lcurve[mask]


    def filter_outliers(self, lcurve):
        pass


    def generate_longterm_lcurve(self, lcurve):
        pass