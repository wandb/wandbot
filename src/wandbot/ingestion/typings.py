from typing import get_type_hints, Optional, Union, List
from wandbot.ingestion.config import DataSource, DataStoreConfig

# Get the type hints for DataSource and DataStoreConfig
DataSourceDict = get_type_hints(DataSource)
DataStoreConfigDict = get_type_hints(DataStoreConfig)

# Replace the types that are not JSON serializable
DataSourceDict["cache_dir"] = str
DataSourceDict["local_path"] = Optional[str]
DataSourceDict["git_id_file"] = Optional[str]
DataSourceDict["file_pattern"] = Union[str, List[str]]

DataStoreConfigDict["data_source"] = DataSourceDict
DataStoreConfigDict["docstore_dir"] = str

# Add additional fields to the type hints for custom fields
DataSourceDict["dataloader_type"] = str