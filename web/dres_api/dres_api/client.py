import time
from dres_api import UserApi, ClientRunInfoApi, SubmissionApi, LogApi
from dres_api.configuration import Configuration
from dres_api.models import LoginRequest, UserDetails, QueryResult, QueryEvent
from dres_api.exceptions import ApiException
from omegaconf import OmegaConf


class Client:
    def __init__(self):
        config = OmegaConf.load('credentials.yaml')

        # Setup
        user_api = UserApi()
        user_api.api_client.configuration.host = config.host

        try:
            login_request = LoginRequest(username=config.username, password=config.password)
            login = user_api.post_api_v1_login(login_request)
        except ApiException as ex:
            print(f"Could not log in due to exception: {ex.message}")
            return

        # Login successful
        print(f"Successfully logged in.\n"
              f"user: '{login.username}'\n"
              f"role: '{login.role}'\n"
              f"session: '{login.session_id}'")

        # Store session token for future requests
        session_id = login.session_id


if __name__ == '__main__':
    client = Client()
