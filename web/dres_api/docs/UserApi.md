# openapi_client.UserApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_v1_logout**](UserApi.md#get_api_v1_logout) | **GET** /api/v1/logout | Clears all user roles of the current session.
[**get_api_v1_user**](UserApi.md#get_api_v1_user) | **GET** /api/v1/user | Get information about the current user.
[**post_api_v1_login**](UserApi.md#post_api_v1_login) | **POST** /api/v1/login | Sets roles for session based on user account and returns a session cookie.


# **get_api_v1_logout**
> SuccessStatus get_api_v1_logout(session=session)

Clears all user roles of the current session.

### Example

```python
import time
import os
import dres_api
from dres_api.models.success_status import SuccessStatus
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
    api_instance = dres_api.UserApi(api_client)
    session = 'session_example'  # str | Session Token (optional)

    try:
        # Clears all user roles of the current session.
        api_response = api_instance.get_api_v1_logout(session=session)
        print("The response of UserApi->get_api_v1_logout:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UserApi->get_api_v1_logout: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | [optional] 

### Return type

[**SuccessStatus**](SuccessStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_v1_user**
> UserDetails get_api_v1_user()

Get information about the current user.

### Example

```python
import time
import os
import dres_api
from dres_api.models.user_details import UserDetails
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
    api_instance = dres_api.UserApi(api_client)

    try:
        # Get information about the current user.
        api_response = api_instance.get_api_v1_user()
        print("The response of UserApi->get_api_v1_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UserApi->get_api_v1_user: %s\n" % e)
```



### Parameters
This endpoint does not need any parameter.

### Return type

[**UserDetails**](UserDetails.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**500** | Server Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_api_v1_login**
> UserDetails post_api_v1_login(login_request=login_request)

Sets roles for session based on user account and returns a session cookie.

### Example

```python
import time
import os
import dres_api
from dres_api.models.login_request import LoginRequest
from dres_api.models.user_details import UserDetails
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
    api_instance = dres_api.UserApi(api_client)
    login_request = dres_api.LoginRequest()  # LoginRequest |  (optional)

    try:
        # Sets roles for session based on user account and returns a session cookie.
        api_response = api_instance.post_api_v1_login(login_request=login_request)
        print("The response of UserApi->post_api_v1_login:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UserApi->post_api_v1_login: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **login_request** | [**LoginRequest**](LoginRequest.md)|  | [optional] 

### Return type

[**UserDetails**](UserDetails.md)

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

