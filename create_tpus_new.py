import multiprocessing
from tpu_utils import (
    create_tpu,
    wait_til,
    delete_tpu,
    get_connection,
    check_tpu,
)
import os

from dataclasses import dataclass, field
from absl import app
from etils import eapp


@dataclass
class Args:
    num_tpus: int = 5
    version: str = "tpu-ubuntu2204-base"
    t_sleep: int = 1
    eu: bool = False
    us1f: bool = False
    us2b: bool = True
    preemptible: bool = False

def create_tpu_till_ready_in_loop(name: str, zone: str, preemptible: bool, t_sleep: int) -> None:
    """
    Trys to create a TPU and wait till it is ready.
    If creation fails, it tries again after args.sleep_period seconds.

    Removed the wait till healthy part, as it was causing issues.

    Args:
        name (str)
        zone (str)
        preemptible (bool)
    """
    while True:
        ret = check_tpu(name, zone)
        os.system("GOOGLE_APPLICATION_CREDENTIALS=ai-memory-f8ae2c64a976.json")
        if "error" in ret.keys():
            print(f'status of {name} = {ret["error"]["status"]}')
            if ret["error"]["status"] == "NOT_FOUND":
                create_tpu(name, zone, preemptible)
                if wait_til(
                    name, zone, {"state": "READY"}, sleep_time=t_sleep
                ):
                    print("TPU is ready")
                    return


def main(args: Args):

    if args.eu:
        zone = "europe-west4-a"
        prefix = "pre-eu" if args.preemptible else "eu"
        assert args.preemptible, "Only preemptible TPUs are available in EU"

    elif args.us1f:
        zone = "us-central1-f"
        prefix = "pre-us1f" if args.preemptible else "us1f"
        assert args.preemptible, "Only preemptible TPUs are available in us-central1-f"
        
    elif args.us2b:
        zone = "us-central2-b"
        prefix = "pre-us" if args.preemptible else "us" 

    else:
        raise ValueError("Please specify a zone")

    for i in range(int(args.num_tpus)):
        name = f"{prefix}-node-{i}"
        p = multiprocessing.Process(
            target=create_tpu_till_ready_in_loop, args=(name, zone, args.preemptible, args.t_sleep)
        )
        p.start()


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(Args))  # type: ignore