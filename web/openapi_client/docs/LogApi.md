# openapi_client.LogApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**post_api_v1_log_query**](LogApi.md#post_api_v1_log_query) | **POST** /api/v1/log/query | Accepts query logs from participants
[**post_api_v1_log_result**](LogApi.md#post_api_v1_log_result) | **POST** /api/v1/log/result | Accepts result logs from participants


# **post_api_v1_log_query**
> SuccessStatus post_api_v1_log_query(session, query_event_log=query_event_log)

Accepts query logs from participants

### Example

```python
import time
import os
import openapi_client
from openapi_client.models.query_event_log import QueryEventLog
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
    api_instance = openapi_client.LogApi(api_client)
    session = 'session_example' # str | Session Token
    query_event_log = openapi_client.QueryEventLog() # QueryEventLog |  (optional)

    try:
        # Accepts query logs from participants
        api_response = api_instance.post_api_v1_log_query(session, query_event_log=query_event_log)
        print("The response of LogApi->post_api_v1_log_query:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling LogApi->post_api_v1_log_query: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | 
 **query_event_log** | [**QueryEventLog**](QueryEventLog.md)|  | [optional] 

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

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_api_v1_log_result**
> SuccessStatus post_api_v1_log_result(session, query_result_log=query_result_log)

Accepts result logs from participants

### Example

```python
import time
import os
import openapi_client
from openapi_client.models.query_result_log import QueryResultLog
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
    api_instance = openapi_client.LogApi(api_client)
    session = 'session_example' # str | Session Token
    query_result_log = openapi_client.QueryResultLog() # QueryResultLog |  (optional)

    try:
        # Accepts result logs from participants
        api_response = api_instance.post_api_v1_log_result(session, query_result_log=query_result_log)
        print("The response of LogApi->post_api_v1_log_result:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling LogApi->post_api_v1_log_result: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | 
 **query_result_log** | [**QueryResultLog**](QueryResultLog.md)|  | [optional] 

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

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

