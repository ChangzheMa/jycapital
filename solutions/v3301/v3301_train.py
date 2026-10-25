"""v3301正式训练入口；必须显式确认，调度由训练session执行。"""
from v3301_run import *
from v3301_preflight import run_device_preflight

if __name__ == "__main__":
    main()
