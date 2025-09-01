# 
# Light weight performance logger to json and (optionally) wandb
# 

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

# hydra logger
elog = logging.getLogger(__name__)


def unnest_dict(nest_dict, flat_dict, prefix=None):
    """Helper method to un-nest a series of nested dictionaries"""
    for k in nest_dict:
        if type(nest_dict[k]) is not dict:
            if prefix is None:
                flat_k = k
            else:
                flat_k = f'{prefix}.{k}'
            flat_v = str(nest_dict[k]) if isinstance(nest_dict[k], list) else nest_dict[k]
            flat_dict[flat_k] = flat_v
        else:
            if prefix is None:
                new_prefix = k
            else:
                new_prefix = f'{prefix}.{k}'
            unnest_dict(nest_dict[k], flat_dict, new_prefix)
    return flat_dict


def np2primitive(obj):
    """Convert numpy objets to primitive types, in particular because not all 
    numpy type are supported by default JSON encoder during write"""

    if type(obj) == torch.Tensor:
        obj = obj.item()

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj  # give up for non-numpy


def table_to_json_serializable_dict(table):
    def serialize_val(val):
        if hasattr(val, "_asdict"):  # for W&B media objects
            return val._asdict()
        elif hasattr(val, "to_json"):  # fallback for custom types
            return val.to_json()
        elif isinstance(val, (int, float, str, bool)) or val is None:
            return val
        else:
            # Convert NumPy scalars or other types
            try:
                return val.item()
            except:
                return str(val)  # fallback to string

    # Initialize dict
    result = {col: [] for col in table.columns}
    for row in table.data:
        for col, val in zip(table.columns, row):
            result[col].append(serialize_val(val))

    return result


class Logger(object):
    """
    Simple logger that wraps over wandb to log to both json and wandb
    """
    def __init__(self, log_dir: Path, use_wb: bool = False, 
                 wb_kwargs: Dict = {}, wb_name: Optional[str] = None,
                 wb_cfg: Optional[DictConfig] = None):
        self._log_dir = log_dir
        self._json_path = log_dir / 'metrics.json'
        # self._meters = defaultdict(RunningMeter)

        self.use_wb = use_wb
        if use_wb:
            self.wb_kwargs = wb_kwargs
            self._init_wandb(run_name=wb_name, wandb_kwargs=wb_kwargs, cfg=wb_cfg)

    def _init_wandb(self, run_name: str, wandb_kwargs: DictConfig, cfg: DictConfig):
        """If using WandB"""
        cfg_dict = OmegaConf.to_container(cfg)  # convert to dictionary
        unnest_cfg = unnest_dict(cfg_dict, {}, prefix='cfg')
        
        try:
            elog.info("Initializing wandb...")
            wandb.login()
            # Init using fork for wandb version > 0.13.0, per:
            # https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
            wandb.init(
                **wandb_kwargs,
                name=run_name,
                config=unnest_cfg,
                settings=wandb.Settings(start_method="fork"),
            )
            elog.info("wandb initialized.")
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            elog.warn(f"Wandb failed to initialize in logger.py, setting use_wb to False. {e}")
            self.use_wb = False
            pass

    def _dump_to_json(self, data: Dict[str, float]):
        # Convert dictionary to JSON string; adding a newline for readability
        # since the file will store multiple entries
        json_str = json.dumps(data) + "\n"

        # Write to file
        with open(self._json_path, 'a') as file:
            file.write(json_str)

    def _dump_to_console(self, data: Dict[str, float]):
        out_str_list = []
        for k in data:
            if isinstance(data[k], dict):
                continue  # skip wandb tables
            out_str_list.append(f'{k}: {data[k]:.03f}')
        
        print(' | '.join(out_str_list))

    def _dump_to_wandb(self, data: Dict[str, float], step: int):
        wandb.log(data, step=step)

    def dump(self, step: int, mode: Optional[str] = None):
        """Write out and dump currently aggregated metrics"""
        # Aggregate data from all meters
        data = {}
        for k in self._meters:
            cur_stats = self._meters[k].get_stats()  # Dict 
            if mode is None or mode == "mean_only_simple":
                data[f'{k}'] = np2primitive(cur_stats["mean"])  # don't add "-mean" in the end
            else:
                for s in cur_stats:
                    if mode is not None and mode == "mean_only" \
                        and s != "mean":
                        continue
                    if mode is not None and mode == "mean_std" \
                        and s not in ["mean", "std"]:
                        continue
                    data[f'{k}-{s}'] = np2primitive(cur_stats[s])

        data['step'] = step  # TODO: necessary or not?

        # Dump and clear existing meters
        self._dump_to_json(data)
        self._dump_to_console(data)
        if self.use_wb:
            self._dump_to_wandb(data, step)

        self._meters.clear()

    def log(self, metrics: Dict[str, Any], step: int):
        """Logs a dictionary of values"""
        metrics['step'] = step  # NOTE: not sure if necessary

        wandb_data = metrics  # assume by default it is wandb compatible
        json_data = {}

        for key, value in wandb_data.items():
            if isinstance(value, wandb.Table):
                # json_value = table_to_json_serializable_dict(value) 
                continue
            else:
                json_value = np2primitive(value)  # convert to primitive type

            json_data[key] = json_value

        self._dump_to_json(json_data)
        self._dump_to_console(json_data)
        if self.use_wb:
            self._dump_to_wandb(wandb_data, step)

    def log_histograms(self, metrics: Dict[str, Any], step: int):
        """Save all list-of-floats in metrics as lists to local JSON, and as wandb histograms if enabled."""

        # TODO: should instead: iterate over each metric, if it is scalar then log
        # normally, if it is a list then log has histogram (wandb) or list (json) or mean (console)
        
        # Always save to local JSON as lists
        json_data = {}
        for key, value in metrics.items():
            if isinstance(value, list) and all(isinstance(x, (float, int, np.floating, np.integer)) for x in value):
                json_data[key] = [np2primitive(x) for x in value]
        json_data['step'] = step

        if json_data:
            self._dump_to_json(json_data)
            console_data = {k: np.mean(v) for k, v in json_data.items()}
            self._dump_to_console(console_data)
        # Log to wandb as histograms if enabled
        if self.use_wb:
            wandb_hist_data = {}
            for key, value in metrics.items():
                if isinstance(value, list) and all(isinstance(x, (float, int, np.floating, np.integer)) for x in value):
                    wandb_hist_data[key + "_hist"] = wandb.Histogram(value)
            if wandb_hist_data:
                wandb.log(wandb_hist_data, step=step)

    def finish(self):
        if self.use_wb:
            wandb.finish()  # TODO: change to join?
            #wandb.join()  # work-around with wandb 0.9.7 to work with multiple
                          # parallel job submission
