import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes)

try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    setattr(args, 'test', True)
    cfg_path = load_config('configs/weather/ms_5_s6_5_625/MTDAPre.py')
    for k, v in cfg_path.items():
        setattr(args, k, v)

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    default_values = default_parser()
    for attribute in default_values.keys():
        if config.get(attribute) is None:
            config[attribute] = default_values[attribute]

    # set multi-process settings
    setup_multi_processes(config)
    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    #exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)