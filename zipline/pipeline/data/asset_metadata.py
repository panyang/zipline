"""
Datasets providing metadata about assets.
"""
from .dataset import Column, DataSet


class AssetMetadata(DataSet):
    """
    Dataset providing point-in-time metadata about assets.
    """
    exchange_id = Column(dtype="S", missing_value="")
    country_id = Column(dtype="S", missing_value="")
    business_country_id = Column(dtype="S", missing_value="")
