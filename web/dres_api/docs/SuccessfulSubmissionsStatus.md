# SuccessfulSubmissionsStatus


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**submission** | **str** |  | 
**description** | **str** |  | 
**status** | **bool** |  | 

## Example

```python
from dres_api.models.successful_submissions_status import SuccessfulSubmissionsStatus

# TODO update the JSON string below
json = "{}"
# create an instance of SuccessfulSubmissionsStatus from a JSON string
successful_submissions_status_instance = SuccessfulSubmissionsStatus.from_json(json)
# print the JSON string representation of the object
print
SuccessfulSubmissionsStatus.to_json()

# convert the object into a dict
successful_submissions_status_dict = successful_submissions_status_instance.to_dict()
# create an instance of SuccessfulSubmissionsStatus from a dict
successful_submissions_status_form_dict = successful_submissions_status.from_dict(successful_submissions_status_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


