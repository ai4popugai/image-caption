# ClientRunInfoList


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**runs** | [**List[ClientRunInfo]**](ClientRunInfo.md) |  | 

## Example

```python
from dres_api.models.client_run_info_list import ClientRunInfoList

# TODO update the JSON string below
json = "{}"
# create an instance of ClientRunInfoList from a JSON string
client_run_info_list_instance = ClientRunInfoList.from_json(json)
# print the JSON string representation of the object
print
ClientRunInfoList.to_json()

# convert the object into a dict
client_run_info_list_dict = client_run_info_list_instance.to_dict()
# create an instance of ClientRunInfoList from a dict
client_run_info_list_form_dict = client_run_info_list.from_dict(client_run_info_list_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


