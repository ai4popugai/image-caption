# openapi_client.ClientRunInfoApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_v1_client_run_info_currenttask_with_runid**](ClientRunInfoApi.md#get_api_v1_client_run_info_currenttask_with_runid) | **GET** /api/v1/client/run/info/currentTask/{runId} | Returns an overview of the currently active task for a run
[**get_api_v1_client_run_info_list**](ClientRunInfoApi.md#get_api_v1_client_run_info_list) | **GET** /api/v1/client/run/info/list | Lists an overview of all competition runs visible to the current client


# **get_api_v1_client_run_info_currenttask_with_runid**
> ClientTaskInfo get_api_v1_client_run_info_currenttask_with_runid(run_id, session)

Returns an overview of the currently active task for a run

### Example

```python
import time
import os
import dres_api
from dres_api.models.client_task_info import ClientTaskInfo
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
    api_instance = dres_api.ClientRunInfoApi(api_client)
    run_id = 'run_id_example'  # str | 
    session = 'session_example'  # str | Session Token

    try:
        # Returns an overview of the currently active task for a run
        api_response = api_instance.get_api_v1_client_run_info_currenttask_with_runid(run_id, session)
        print("The response of ClientRunInfoApi->get_api_v1_client_run_info_currenttask_with_runid:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ClientRunInfoApi->get_api_v1_client_run_info_currenttask_with_runid: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**|  | 
 **session** | **str**| Session Token | 

### Return type

[**ClientTaskInfo**](ClientTaskInfo.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_v1_client_run_info_list**
> ClientRunInfoList get_api_v1_client_run_info_list(session)

Lists an overview of all competition runs visible to the current client

### Example

```python
import time
import os
import dres_api
from dres_api.models.client_run_info_list import ClientRunInfoList
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
    api_instance = dres_api.ClientRunInfoApi(api_client)
    session = 'session_example'  # str | Session Token

    try:
        # Lists an overview of all competition runs visible to the current client
        api_response = api_instance.get_api_v1_client_run_info_list(session)
        print("The response of ClientRunInfoApi->get_api_v1_client_run_info_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ClientRunInfoApi->get_api_v1_client_run_info_list: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | 

### Return type

[**ClientRunInfoList**](ClientRunInfoList.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

