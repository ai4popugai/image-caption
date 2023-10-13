import os

import dres_api
from dres_api import UserApi, SubmissionApi, ClientRunInfoApi, ClientRunInfoList, ResultElement, TaskResult, RunResult, \
    BatchSubmissionApi, ClientRunInfo
from dres_api.configuration import Configuration
from dres_api.models import LoginRequest
from dres_api.exceptions import ApiException
from omegaconf import OmegaConf


class Client:
    def __init__(self):
        config = OmegaConf.load(f'{ os.path.dirname(os.path.abspath(__file__))}/credentials.yaml')
        self.configuration_instance = Configuration(**config)
        self.session = 'node0p14zq9j9w771dqyhjpc1kmgd200'

        # Enter a context with an instance of the API client
        with dres_api.ApiClient(self.configuration_instance) as api_client:
            # Create an instance of the API class
            api_user = UserApi(api_client)

            try:
                login_request = LoginRequest(**config)
                login = api_user.post_api_v1_login(login_request)
            except ApiException as ex:
                print(f"Could not log in due to exception: {ex.message}")
                return

            # Login successful
            print(f"Successfully logged in.\n"
                  f"user: '{login.username}'\n"
                  f"role: '{login.role}'\n")

            # see all runs
            api_info = ClientRunInfoApi(api_client)
            run_list: ClientRunInfoList = api_info.get_api_v1_client_run_info_list(self.session)
            run: ClientRunInfo = run_list.runs[0]
            self.run_id = run.id
            print(f'run available: {self.run_id}\n')

    def submit(self, item: str, frame: int,):
        """
        Method to submit search result to the DRES server.

        :param item: Identifier for the actual media object or media file.
        :param frame: Frame number for media with temporal progression (e.g. video).
        :return:
        """
        with dres_api.ApiClient(self.configuration_instance) as api_client:
            # Create an instance of the API class
            api_submission = SubmissionApi(api_client)

            try:
                # Endpoint to accept submissions
                api_response = api_submission.get_api_v1_submit(collection=None, item=item, text=None,
                                                                frame=frame, shot=None,
                                                                timecode=None, session=self.session)
                print("The response of SubmissionApi->get_api_v1_submit:\n")
                print(api_response)
            except Exception as e:
                print("Exception when calling SubmissionApi->get_api_v1_submit: %s\n" % e)

    def batch_submit(self, item: str, frame: int, timestamp: str, ):
        """
        Method to submit search result to the DRES server.

        :param item: Identifier for the actual media object or media file.
        :param frame: Frame number for media with temporal progression (e.g. video).
        :param timestamp: Timecode for media with temporal progression (e.g. video).
        :return:
        """
        with dres_api.ApiClient(self.configuration_instance) as api_client:
            # Create an instance of the API class
            api_submission = BatchSubmissionApi(api_client)
            text = None

            result = ResultElement(item=item, text=text,
                                   start_time_code=timestamp, end_time_code=timestamp,
                                   index=frame)

            task_result = TaskResult(task='video_search', resultName='my_result',
                                     results=[result])
            run_result = RunResult(tasks=[task_result], timeStamp=frame)

            try:
                # Endpoint to accept batch submissions in JSON format
                api_response = api_submission.post_api_v1_submit_with_runid(self.run_id, run_result=run_result)
                print("The response of BatchSubmissionApi->post_api_v1_submit_with_runid:\n")
                print(api_response)
            except Exception as e:
                print("Exception when calling BatchSubmissionApi->post_api_v1_submit_with_runid: %s\n" % e)


if __name__ == '__main__':
    client = Client()
    client.submit(item='00063', frame=1396)
