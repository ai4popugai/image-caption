import time
from dres_api import UserApi, ClientRunInfoApi, SubmissionApi, LogApi
from dres_api.configuration import Configuration
from dres_api.models import LoginRequest, UserDetails, QueryResult, QueryEvent
from dres_api.exceptions import ApiException

class Client:
    @staticmethod
    def run_example():
        config = Configuration()
        config.base_path = Settings.base_path

        # Setup
        user_api = UserApi(config)
        run_info_api = ClientRunInfoApi(config)
        submission_api = SubmissionApi(config)
        log_api = LogApi(config)

        # Handshake
        print(f"Try to log in to {config.base_path} with user {Settings.user}")

        # Login request
        login = None
        try:
            login = user_api.post_api_v1_login(LoginRequest(Settings.user, Settings.password))
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
        time.sleep(1)

        # Example 1: Evaluation Runs Info
        current_runs = run_info_api.get_api_v1_client_run_info_list(session_id)
        print(f"Found {len(current_runs.runs)} ongoing evaluation runs")

        for info in current_runs.runs:
            print(f"{info.name} ({info.id}): {info.status}")
            if info.description is not None:
                print(info.description)
            print()

        # Example 2: Submission
        submission_response = None
        try:
            submission_response = submission_api.get_api_v1_submit(
                session=session_id,
                collection=None,
                item="some_item_name",
                frame=None,
                shot=None,
                timecode="00:00:10:00",
                text=None
            )
        except ApiException as ex:
            if ex.error_code == 401:
                print("There was an authentication error during submission. Check the session id.")
            elif ex.error_code == 404:
                print("There is currently no active task which would accept submissions.")
            else:
                print(f"Something unexpected went wrong during the submission: '{ex.message}'.")
                return

        if submission_response is not None and submission_response.status:
            print("The submission was successfully sent to the server.")

            # Example 3: Log
            log_api.post_api_v1_log_result(
                session=session_id,
                query_result_log=QueryResultLog(
                    timestamp=int(time.time() * 1000),
                    sort_type="list",
                    results=Client.create_result_list(),
                    events=[],
                    result_set_availability=""
                )
            )

        # Graceful logout
        time.sleep(1)
        logout = user_api.get_api_v1_logout(session_id)

        if logout.status:
            print("Successfully logged out")
        else:
            print(f"Error during logout {logout.description}")

    @staticmethod
    def create_result_list():
        result_list = [
            QueryResult("some_item_name", segment=3, score=0.9, rank=1),
            QueryResult("some_item_name", segment=5, score=0.85, rank=2),
            QueryResult("some_other_item_name", segment=12, score=0.76, rank=3)
        ]
        return result_list

    @staticmethod
    def println(msg=""):
        print(msg)