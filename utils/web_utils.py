import os

from dres_api.client import Client


def submit_match(client: Client, match):
    print(f'submitting {match}')
    client.submit(item=match[0],
                  frame=int(os.path.splitext(match[1].split('_')[1])[0]),
                  timestamp=match[2])
