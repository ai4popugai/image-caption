# RunResult


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tasks** | [**List[TaskResult]**](TaskResult.md) |  | 
**time_stamp** | **int** |  | 

## Example

```python
from dres_api.models.run_result import RunResult

# TODO update the JSON string below
json = "{}"
# create an instance of RunResult from a JSON string
run_result_instance = RunResult.from_json(json)
# print the JSON string representation of the object
print
RunResult.to_json()

# convert the object into a dict
run_result_dict = run_result_instance.to_dict()
# create an instance of RunResult from a dict
run_result_form_dict = run_result.from_dict(run_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


