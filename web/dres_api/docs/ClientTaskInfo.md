# ClientTaskInfo


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | 
**name** | **str** |  | 
**task_group** | **str** |  | 
**remaining_time** | **int** |  | 
**running** | **bool** |  | 

## Example

```python
from dres_api.models.client_task_info import ClientTaskInfo

# TODO update the JSON string below
json = "{}"
# create an instance of ClientTaskInfo from a JSON string
client_task_info_instance = ClientTaskInfo.from_json(json)
# print the JSON string representation of the object
print
ClientTaskInfo.to_json()

# convert the object into a dict
client_task_info_dict = client_task_info_instance.to_dict()
# create an instance of ClientTaskInfo from a dict
client_task_info_form_dict = client_task_info.from_dict(client_task_info_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


