# QueryEventLog


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**timestamp** | **int** |  | 
**events** | [**List[QueryEvent]**](QueryEvent.md) |  | 

## Example

```python
from openapi_client.models.query_event_log import QueryEventLog

# TODO update the JSON string below
json = "{}"
# create an instance of QueryEventLog from a JSON string
query_event_log_instance = QueryEventLog.from_json(json)
# print the JSON string representation of the object
print QueryEventLog.to_json()

# convert the object into a dict
query_event_log_dict = query_event_log_instance.to_dict()
# create an instance of QueryEventLog from a dict
query_event_log_form_dict = query_event_log.from_dict(query_event_log_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


