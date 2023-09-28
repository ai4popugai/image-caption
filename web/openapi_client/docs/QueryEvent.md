# QueryEvent


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**timestamp** | **int** |  | 
**category** | **str** |  | 
**type** | **str** |  | 
**value** | **str** |  | 

## Example

```python
from openapi_client.models.query_event import QueryEvent

# TODO update the JSON string below
json = "{}"
# create an instance of QueryEvent from a JSON string
query_event_instance = QueryEvent.from_json(json)
# print the JSON string representation of the object
print QueryEvent.to_json()

# convert the object into a dict
query_event_dict = query_event_instance.to_dict()
# create an instance of QueryEvent from a dict
query_event_form_dict = query_event.from_dict(query_event_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


