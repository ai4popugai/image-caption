# openapi_client.SubmissionApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_v1_submit**](SubmissionApi.md#get_api_v1_submit) | **GET** /api/v1/submit | Endpoint to accept submissions


# **get_api_v1_submit**
> SuccessfulSubmissionsStatus get_api_v1_submit(collection=collection, item=item, text=text, frame=frame, shot=shot, timecode=timecode, session=session)

Endpoint to accept submissions

### Example

```python
import time
import os
import dres_api
from dres_api.models.successful_submissions_status import SuccessfulSubmissionsStatus
from dres_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = dres_api.Configuration(
    host="http://localhost"
)

# Enter a context with an instance of the API client
with dres_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = dres_api.SubmissionApi(api_client)
    collection = 'collection_example'  # str | Collection identifier. Optional, in which case the default collection for the run will be considered. (optional)
    item = 'item_example'  # str | Identifier for the actual media object or media file. (optional)
    text = 'text_example'  # str | Text to be submitted. ONLY for tasks with target type TEXT. If this parameter is provided, it superseeds all athers. (optional)
    frame = 56  # int | Frame number for media with temporal progression (e.g. video). (optional)
    shot = 56  # int | Shot number for media with temporal progression (e.g. video). (optional)
    timecode = 'timecode_example'  # str | Timecode for media with temporal progression (e.g. video). (optional)
    session = 'session_example'  # str | Session Token (optional)

    try:
        # Endpoint to accept submissions
        api_response = api_instance.get_api_v1_submit(collection=collection, item=item, text=text, frame=frame,
                                                      shot=shot, timecode=timecode, session=session)
        print("The response of SubmissionApi->get_api_v1_submit:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SubmissionApi->get_api_v1_submit: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **collection** | **str**| Collection identifier. Optional, in which case the default collection for the run will be considered. | [optional] 
 **item** | **str**| Identifier for the actual media object or media file. | [optional] 
 **text** | **str**| Text to be submitted. ONLY for tasks with target type TEXT. If this parameter is provided, it superseeds all athers. | [optional] 
 **frame** | **int**| Frame number for media with temporal progression (e.g. video). | [optional] 
 **shot** | **int**| Shot number for media with temporal progression (e.g. video). | [optional] 
 **timecode** | **str**| Timecode for media with temporal progression (e.g. video). | [optional] 
 **session** | **str**| Session Token | [optional] 

### Return type

[**SuccessfulSubmissionsStatus**](SuccessfulSubmissionsStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**202** | Accepted |  -  |
**400** | Bad Request |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |
**412** | Precondition Failed |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

