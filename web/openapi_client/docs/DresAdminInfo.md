# DresAdminInfo


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**version** | **str** |  | 
**start_time** | **int** |  | 
**uptime** | **int** |  | 
**os** | **str** |  | 
**jvm** | **str** |  | 
**args** | **str** |  | 
**cores** | **int** |  | 
**free_memory** | **int** |  | 
**total_memory** | **int** |  | 
**load** | **float** |  | 
**available_sever_threads** | **int** |  | 

## Example

```python
from openapi_client.models.dres_admin_info import DresAdminInfo

# TODO update the JSON string below
json = "{}"
# create an instance of DresAdminInfo from a JSON string
dres_admin_info_instance = DresAdminInfo.from_json(json)
# print the JSON string representation of the object
print DresAdminInfo.to_json()

# convert the object into a dict
dres_admin_info_dict = dres_admin_info_instance.to_dict()
# create an instance of DresAdminInfo from a dict
dres_admin_info_form_dict = dres_admin_info.from_dict(dres_admin_info_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


