# openapi_client.DownloadApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_v1_download_competition_with_competitionid**](DownloadApi.md#get_api_v1_download_competition_with_competitionid) | **GET** /api/v1/download/competition/{competitionId} | Provides a JSON download of the entire competition description structure.
[**get_api_v1_download_run_with_runid**](DownloadApi.md#get_api_v1_download_run_with_runid) | **GET** /api/v1/download/run/{runId} | Provides a JSON download of the entire competition run structure.
[**get_api_v1_download_run_with_runid_scores**](DownloadApi.md#get_api_v1_download_run_with_runid_scores) | **GET** /api/v1/download/run/{runId}/scores | Provides a CSV download with the scores for a given competition run.


# **get_api_v1_download_competition_with_competitionid**
> str get_api_v1_download_competition_with_competitionid(competition_id)

Provides a JSON download of the entire competition description structure.

### Example

```python
import time
import os
import dres_api
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
    api_instance = dres_api.DownloadApi(api_client)
    competition_id = 'competition_id_example'  # str | Competition ID

    try:
        # Provides a JSON download of the entire competition description structure.
        api_response = api_instance.get_api_v1_download_competition_with_competitionid(competition_id)
        print("The response of DownloadApi->get_api_v1_download_competition_with_competitionid:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DownloadApi->get_api_v1_download_competition_with_competitionid: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **competition_id** | **str**| Competition ID | 

### Return type

**str**

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
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_v1_download_run_with_runid**
> str get_api_v1_download_run_with_runid(run_id)

Provides a JSON download of the entire competition run structure.

### Example

```python
import time
import os
import dres_api
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
    api_instance = dres_api.DownloadApi(api_client)
    run_id = 'run_id_example'  # str | Competition run ID

    try:
        # Provides a JSON download of the entire competition run structure.
        api_response = api_instance.get_api_v1_download_run_with_runid(run_id)
        print("The response of DownloadApi->get_api_v1_download_run_with_runid:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DownloadApi->get_api_v1_download_run_with_runid: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| Competition run ID | 

### Return type

**str**

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
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_v1_download_run_with_runid_scores**
> str get_api_v1_download_run_with_runid_scores(run_id)

Provides a CSV download with the scores for a given competition run.

### Example

```python
import time
import os
import dres_api
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
    api_instance = dres_api.DownloadApi(api_client)
    run_id = 'run_id_example'  # str | Competition run ID

    try:
        # Provides a CSV download with the scores for a given competition run.
        api_response = api_instance.get_api_v1_download_run_with_runid_scores(run_id)
        print("The response of DownloadApi->get_api_v1_download_run_with_runid_scores:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DownloadApi->get_api_v1_download_run_with_runid_scores: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| Competition run ID | 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: text/csv, application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

