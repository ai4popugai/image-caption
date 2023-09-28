# openapi_client.StatusApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_v1_status_info**](StatusApi.md#get_api_v1_status_info) | **GET** /api/v1/status/info | Returns an overview of the server properties.
[**get_api_v1_status_info_admin**](StatusApi.md#get_api_v1_status_info_admin) | **GET** /api/v1/status/info/admin | Returns an extensive overview of the server properties.
[**get_api_v1_status_time**](StatusApi.md#get_api_v1_status_time) | **GET** /api/v1/status/time | Returns the current time on the server.


# **get_api_v1_status_info**
> DresInfo get_api_v1_status_info()

Returns an overview of the server properties.

### Example

```python
import time
import os
import dres_api
from dres_api.models.dres_info import DresInfo
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
    api_instance = dres_api.StatusApi(api_client)

    try:
        # Returns an overview of the server properties.
        api_response = api_instance.get_api_v1_status_info()
        print("The response of StatusApi->get_api_v1_status_info:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling StatusApi->get_api_v1_status_info: %s\n" % e)
```



### Parameters
This endpoint does not need any parameter.

### Return type

[**DresInfo**](DresInfo.md)

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

# **get_api_v1_status_info_admin**
> DresAdminInfo get_api_v1_status_info_admin()

Returns an extensive overview of the server properties.

### Example

```python
import time
import os
import dres_api
from dres_api.models.dres_admin_info import DresAdminInfo
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
    api_instance = dres_api.StatusApi(api_client)

    try:
        # Returns an extensive overview of the server properties.
        api_response = api_instance.get_api_v1_status_info_admin()
        print("The response of StatusApi->get_api_v1_status_info_admin:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling StatusApi->get_api_v1_status_info_admin: %s\n" % e)
```



### Parameters
This endpoint does not need any parameter.

### Return type

[**DresAdminInfo**](DresAdminInfo.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_v1_status_time**
> CurrentTime get_api_v1_status_time()

Returns the current time on the server.

### Example

```python
import time
import os
import dres_api
from dres_api.models.current_time import CurrentTime
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
    api_instance = dres_api.StatusApi(api_client)

    try:
        # Returns the current time on the server.
        api_response = api_instance.get_api_v1_status_time()
        print("The response of StatusApi->get_api_v1_status_time:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling StatusApi->get_api_v1_status_time: %s\n" % e)
```



### Parameters
This endpoint does not need any parameter.

### Return type

[**CurrentTime**](CurrentTime.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

