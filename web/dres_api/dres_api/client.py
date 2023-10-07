import os

import dres_api
from dres_api import UserApi, SubmissionApi
from dres_api.configuration import Configuration
from dres_api.models import LoginRequest
from dres_api.exceptions import ApiException
from omegaconf import OmegaConf


class Client:
    def __init__(self):
        config = OmegaConf.load(f'{ os.path.dirname(os.path.abspath(__file__))}/credentials.yaml')
        self.configuration_instance = Configuration(**config)
        self.session = 'node013jsjozq6rksbv6ua6iq8wy3c0'

        # Enter a context with an instance of the API client
        with dres_api.ApiClient(self.configuration_instance) as api_client:
            # Create an instance of the API class
            api_instance = UserApi(api_client)

            try:
                login_request = LoginRequest(**config)
                login = api_instance.post_api_v1_login(login_request)
            except ApiException as ex:
                print(f"Could not log in due to exception: {ex.message}")
                return

            # Login successful
            print(f"Successfully logged in.\n"
                  f"user: '{login.username}'\n"
                  f"role: '{login.role}'\n")

    def submit(self, item: str, frame: int, timecode: str,):
        """
        Method to submit search result to the DRES server.

        :param item: Identifier for the actual media object or media file.
        :param frame: Frame number for media with temporal progression (e.g. video).
        :param timecode: Timecode for media with temporal progression (e.g. video).
        :return:
        """
        with dres_api.ApiClient(self.configuration_instance) as api_client:
            # Create an instance of the API class
            api_instance = SubmissionApi(api_client)
            collection = None
            text = None
            shot = frame
            session = self.session

            try:
                # Endpoint to accept submissions
                api_response = api_instance.get_api_v1_submit(collection=collection, item=item, text=text, frame=frame,
                                                              shot=shot, timecode=timecode, session=session)
                print("The response of SubmissionApi->get_api_v1_submit:\n")
                print(api_response)
            except Exception as e:
                print("Exception when calling SubmissionApi->get_api_v1_submit: %s\n" % e)


if __name__ == '__main__':
    client = Client()
    client.submit(item='00100.mp4', frame=12, timecode='00:00:00.000')
