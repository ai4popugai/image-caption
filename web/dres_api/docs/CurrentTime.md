# CurrentTime


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**time_stamp** | **int** |  | 

## Example

```python
from dres_api.models.current_time import CurrentTime

# TODO update the JSON string below
json = "{}"
# create an instance of CurrentTime from a JSON string
current_time_instance = CurrentTime.from_json(json)
# print the JSON string representation of the object
print
CurrentTime.to_json()

# convert the object into a dict
current_time_dict = current_time_instance.to_dict()
# create an instance of CurrentTime from a dict
current_time_form_dict = current_time.from_dict(current_time_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


