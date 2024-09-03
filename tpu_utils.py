import functools
import os
import subprocess
import threading
import time

import glob
from typing import Literal
import requests
from fabric import Connection
from google.auth import default
import logging
from fabric import Config

config = Config()
config["run"]["warn"] = True

logging.basicConfig(filename="refresh.log", level=logging.WARNING)
# from google.cloud import logging as cloud_logging


def refresh_credentials():
    credentials, project = default()


# @functools.lru_cache()
# def get_fast_bearer():
#     return get_bearer()


def get_bearer():
    return (
        subprocess.check_output("gcloud auth print-access-token", shell=True)
        .decode("utf-8")
        .strip()
    )


@functools.lru_cache()
def get_project():
    return (
        subprocess.check_output(
            "gcloud config list --format 'value(core.project)'", shell=True
        )
        .decode("utf-8")
        .strip()
    )


def create_tpu(
    name,
    zone,
    preemptible,
):
    headers = {
        "Authorization": f"Bearer {get_bearer()}",
        "Content-Type": "application/json",
    }
    params = (("node_id", name),)
    if zone == "us-central1-f":
        type = "v2-8"
    elif zone == "europe-west4-a":
        type = "v3-8"
    elif zone == "us-central2-b":
        type = "v4-8"
    else:
        raise ValueError("Zone not supported")
    data = {
        "accelerator_type": type,
        "runtime_version": "tpu-ubuntu2204-base",
        "network_config": {"enable_external_ips": True},
    }

    if not preemptible:
        assert (
            zone == "us-central2-b" and type == "v4-8"
        ), "Only v4-8 in us-central2-b is supported for non-preemptible TPUs"

    if preemptible:
        data["schedulingConfig"] = {"preemptible": True}
    
    response = requests.post(
        f"https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes",
        headers=headers,
        params=params,
        json=data,
    )
    # print(response.json())
    return response.status_code == 200


def check_tpu(name, zone):
    try:
        headers = {
            "Authorization": f"Bearer {get_bearer()}",
        }
        response = requests.get(
            f"https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}",
            headers=headers,
        )
        ret = response.json()

        if "error" in ret:
            if ret["error"]["status"] == "UNAUTHENTICATED":
                logging.warning("Key expired, refreshing get_fast_bearer.cache")
                # get_fast_bearer.cache_clear()
                get_project.cache_clear()
                return check_tpu(name, zone)
        return ret

    except Exception as e:
        logging.warning(f"Exception in check_tpu: {e}")
        raise e


def delete_tpu(name, zone):
    headers = {
        "Authorization": f"Bearer {get_bearer()}",
    }
    response = requests.delete(
        f"https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}",
        headers=headers,
    )
    return response.json()


def wait_til(name, zone, state, sleep_time=30):
    while True:
        ret = check_tpu(name, zone)
        try:
            print(f'status of {name} = {ret["state"]}')
        except:
            print(f"status of {name} = {ret}")
        matches = True
        for k, expected_v in state.items():
            if k not in ret:
                matches = False
                continue
            if ret[k] != expected_v:
                matches = False
        if "error" in ret:
            return False
        if ret["state"] == "TERMINATED":
            return False
        elif ret["state"] == "PREEMPTED":
            print(f"{name} preempted")
            delete_tpu(name, zone)
            return False
        if matches:
            return True
        time.sleep(sleep_time)


# name = 'test-1'
# zone = 'us-central1-f'
# state = {'state': 'READY', 'health': 'HEALTHY'}


def get_connection(
    name,
    zone,
    retries=3,
):
    # For some reason this makes it work
    try:
        os.system(
            f'gcloud alpha compute tpus tpu-vm ssh {name} --zone {zone} --command="echo hi"'
        )
        info = check_tpu(name, zone)
        outputs = []
        for i in info["networkEndpoints"]:
            outputs.append(
                Connection(
                    i["ipAddress"],
                    config=config,
                    connect_timeout=300,
                    connect_kwargs={
                        "key_filename": os.path.expanduser(
                            "~/.ssh/google_compute_engine"
                        ),
                    },
                )
            )
        return outputs

    except Exception as e:
        logging.warning(f"Exception conneting to {name}: {e}")
        print(f"Exception conneting to {name}: {e}")
        print(f"Retrying {retries} more times")
        if retries > 0:
            time.sleep(10)
            return get_connection(name, zone, retries=retries - 1)
        else:
            raise e


def test_get_connection():
    name = "tuner-75-11"
    zone = "us-central1-f"

    get_connection(name, zone)

    print("done")


def get_tpu_vm_zone():
    result = subprocess.run(
        [
            "curl",
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            "-H",
            "Metadata-Flavor: Google",
        ],
        capture_output=True,
    )
    tpu_vm_zone = result.stdout.decode().strip()
    zone = tpu_vm_zone.split("/")[3]
    return zone


# class RefreshFn(object):
#     def __init__(self):
#         # client = cloud_logging.Client()
#         # client.setup_logging()

#         logging.warning('Initializing RefreshFn')

#     def keep_clearing_cache(self):
#         '''
#         Periodically clears the cache of get_fast_bearer function.
#         '''
#         logging.warning('Clearing get_fast_bearer.cache')
#         get_fast_bearer.cache_clear()
#         self.timer = threading.Timer(self.refresh_period,
#                                      self.keep_clearing_cache)
#         self.timer.start()
#         return

#     def force_refresh(self):
#         '''
#         Forces a refresh of the cache of get_fast_bearer function. To
#         Be called when the refresh period is updated.
#         '''
#         logging.warning('Forcing refresh of get_token.cache')
#         self.keep_clearing_cache()
#         return

#     def get_time_diff(self):
#         '''
#         Helper function to get the time difference between the current
#         time and the last time this function was called.
#         '''
#         self.cur_time = time.time()
#         diff = int(self.cur_time - self.start_time)
#         self.start_time = self.cur_time
#         return diff

#     def update_refresh_period(self):
#         '''
#         Updates the refresh period of the cache of get_fast_bearer function.
#         '''
#         diff = self.get_time_diff()
#         self.refresh_period = diff
#         logging.warning(f'Updating refresh period from {self.refresh_period} to {diff}')
#         self.force_refresh()
#         return

#     def efficient_refresh(self):
#         '''
#         This function checks if the bearer token has been updated.
#         It does so using a inifinite loop that checks if the bearer
#         token is the same as the one returned by get_fast_bearer.
#         If it is not the same, it updates the refresh period
#         and calls the force_refresh function.
#         '''
#         self.start_time = time.time()

#         actual_bearer = get_bearer()
#         fast_bearer = get_fast_bearer()

#         while actual_bearer == fast_bearer:
#             logging.warning(f'Actual bearer == Fast bearer')
#             time.sleep(1)
#             actual_bearer = get_bearer()
#             fast_bearer = get_fast_bearer()

#         self.update_refresh_period()

#         return

#     def cancel(self):
#         self.timer.cancel()
#         return

# def main():
#     refresher = RefreshFn()
#     refresher.efficient_refresh()
#     refresher.cancel()

# if __name__ == '__main__':
#     main()
