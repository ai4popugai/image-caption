# DresInfo


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**version** | **str** |  | 
**start_time** | **int** |  | 
**uptime** | **int** |  | 

## Example

```python
from openapi_client.models.dres_info import DresInfo

# TODO update the JSON string below
json = "{}"
# create an instance of DresInfo from a JSON string
dres_info_instance = DresInfo.from_json(json)
# print the JSON string representation of the object
print DresInfo.to_json()

# convert the object into a dict
dres_info_dict = dres_info_instance.to_dict()
# create an instance of DresInfo from a dict
dres_info_form_dict = dres_info.from_dict(dres_info_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


