from easycat.lightcurve.reprocess import register_aggregator
from .wise import WiseAggregator

register_aggregator('wise', WiseAggregator)
