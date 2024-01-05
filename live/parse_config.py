import os
import logging
from pathlib import Path
from utils import read_json
import dataset as module_dataset


class ConfigParser:
    def __init__(self, config, resume=None, modification=None) -> object:
        self.__config = _update_config(config, modification)
        self.resume = resume

    @property
    def config(self):
        return self.__config

    @classmethod
    def from_args(cls, args, options=""):  # None 대신 ""를 사용
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.json"
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))

        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }

        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])

        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def init_data_loader(self, name, module, *args, **kwargs):
        module_name = self[name]["type"]
        dataset = self.init_obj("dataset", module_dataset)

        module_args = {"dataset": dataset}
        module_args.update(self[name]["args"])

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            pass
    return config
