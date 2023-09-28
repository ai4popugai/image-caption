# ClientRunInfo


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | 
**name** | **str** |  | 
**description** | **str** |  | [optional] 
**status** | **str** |  | 

## Example

```python
from dres_api.models.client_run_info import ClientRunInfo

# TODO update the JSON string below
json = "{}"
# create an instance of ClientRunInfo from a JSON string
client_run_info_instance = ClientRunInfo.from_json(json)
# print the JSON string representation of the object
print
ClientRunInfo.to_json()

# convert the object into a dict
client_run_info_dict = client_run_info_instance.to_dict()
# create an instance of ClientRunInfo from a dict
client_run_info_form_dict = client_run_info.from_dict(client_run_info_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


