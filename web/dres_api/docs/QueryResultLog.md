# QueryResultLog


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**timestamp** | **int** |  | 
**sort_type** | **str** |  | 
**result_set_availability** | **str** |  | 
**results** | [**List[QueryResult]**](QueryResult.md) |  | 
**events** | [**List[QueryEvent]**](QueryEvent.md) |  | 

## Example

```python
from dres_api.models.query_result_log import QueryResultLog

# TODO update the JSON string below
json = "{}"
# create an instance of QueryResultLog from a JSON string
query_result_log_instance = QueryResultLog.from_json(json)
# print the JSON string representation of the object
print
QueryResultLog.to_json()

# convert the object into a dict
query_result_log_dict = query_result_log_instance.to_dict()
# create an instance of QueryResultLog from a dict
query_result_log_form_dict = query_result_log.from_dict(query_result_log_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


