# openapi_client.BatchSubmissionApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**post_api_v1_submit_with_runid**](BatchSubmissionApi.md#post_api_v1_submit_with_runid) | **POST** /api/v1/submit/{runId} | Endpoint to accept batch submissions in JSON format


# **post_api_v1_submit_with_runid**
> SuccessStatus post_api_v1_submit_with_runid(run_id, run_result=run_result)

Endpoint to accept batch submissions in JSON format

### Example

```python
import time
import os
import openapi_client
from openapi_client.models.run_result import RunResult
from openapi_client.models.success_status import SuccessStatus
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.BatchSubmissionApi(api_client)
    run_id = 'run_id_example' # str | Competition Run ID
    run_result = openapi_client.RunResult() # RunResult |  (optional)

    try:
        # Endpoint to accept batch submissions in JSON format
        api_response = api_instance.post_api_v1_submit_with_runid(run_id, run_result=run_result)
        print("The response of BatchSubmissionApi->post_api_v1_submit_with_runid:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling BatchSubmissionApi->post_api_v1_submit_with_runid: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| Competition Run ID | 
 **run_result** | [**RunResult**](RunResult.md)|  | [optional] 

### Return type

[**SuccessStatus**](SuccessStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

