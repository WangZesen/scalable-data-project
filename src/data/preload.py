import os
import subprocess
import time
import torch.distributed as dist
from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..conf import Config

def preload_to_local(cfg: 'Config'):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    tmp_dir = os.environ["TMPDIR"]

    if local_rank == 0:
        logger.info(f'Preloading data to local storage [rank {rank}]')
        subprocess.call(f'cp {cfg.data.sharded_data_dir}/{cfg.data.train_prefix}.zip {tmp_dir}', shell=True)
        subprocess.call(f'cp {cfg.data.sharded_data_dir}/{cfg.data.valid_prefix}.zip {tmp_dir}', shell=True)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
    
    if rank == 0:
        logger.info(f'Unzipping data [rank {rank}]')
    
    start_time = time.time()
    if local_rank == 0:
        subprocess.check_call(f"unzip -q {tmp_dir}/{cfg.data.train_prefix}.zip -d {tmp_dir}", shell=True)
        subprocess.check_call(f"mv {tmp_dir}/{cfg.data.train_prefix} {tmp_dir}/train", shell=True)
        subprocess.check_call(f"unzip -q {tmp_dir}/{cfg.data.valid_prefix}.zip -d {tmp_dir}", shell=True)
        subprocess.check_call(f"mv {tmp_dir}/{cfg.data.valid_prefix} {tmp_dir}/val", shell=True)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
    if rank == 0:
        logger.info(f'Done. Elapsed time: {time.time() - start_time:.2f} s')
    if rank == 0:
        logger.info(f'Preload Complete [rank {rank}]. "data_dir" is changed to: {tmp_dir}')

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    cfg.data.data_dir = tmp_dir

