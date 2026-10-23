"""v2801冻结历史表示与梯度提升的单次正式训练入口。"""
from v2801_run import validate_training, execute, parser, main
from v2801_preflight import run_device_preflight

if __name__ == '__main__':
    main()
