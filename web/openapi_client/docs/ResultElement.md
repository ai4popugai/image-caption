# ResultElement


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**item** | **str** |  | [optional] 
**text** | **str** |  | [optional] 
**start_time_code** | **str** |  | [optional] 
**end_time_code** | **str** |  | [optional] 
**index** | **int** |  | [optional] 
**rank** | **int** |  | [optional] 
**weight** | **float** |  | [optional] 

## Example

```python
from openapi_client.models.result_element import ResultElement

# TODO update the JSON string below
json = "{}"
# create an instance of ResultElement from a JSON string
result_element_instance = ResultElement.from_json(json)
# print the JSON string representation of the object
print ResultElement.to_json()

# convert the object into a dict
result_element_dict = result_element_instance.to_dict()
# create an instance of ResultElement from a dict
result_element_form_dict = result_element.from_dict(result_element_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


