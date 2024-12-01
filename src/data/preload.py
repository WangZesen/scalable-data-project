import os
import subprocess
import time
import shutil
import numpy as np
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
    if rank == 0:
        logger.info(f'Done. Elapsed time: {time.time() - start_time:.2f} s')
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    if cfg.data.alpha > 0:
        
        if local_rank == 0:
            rng = np.random.RandomState(cfg.train.reproduce.seed)

            # get classes
            classes = sorted(os.listdir(f'{tmp_dir}/train'))
            # get proportion of classes
            proportions = rng.dirichlet([cfg.data.alpha] * world_size, len(classes))
            # number of files per class per worker
            num_files = {}

            # move files to new directories
            for i in range(len(classes)):
                num_files[classes[i]] = []
                files = sorted(os.listdir(f'{tmp_dir}/train/{classes[i]}'))
                rng.shuffle(files)
                for j in range(world_size):
                    # create new directory
                    os.makedirs(f'{tmp_dir}/train_{j}/{classes[i]}', exist_ok=True)

                    # move files
                    s = round(np.sum(proportions[i][:j]) * len(files))
                    t = round(np.sum(proportions[i][:j+1]) * len(files))
                    if (t - s == 0) and (rank == 0):
                        logger.warning(f'No files for worker {j} in class {classes[i]}')
                    for k in range(s, t):
                        shutil.move(f'{tmp_dir}/train/{classes[i]}/{files[k]}', f'{tmp_dir}/train_{j}/{classes[i]}/')
                    
                    num_files[classes[i]].append(t - s)
        
            if rank == 0:
                logger.info('---- Number of samples per worker ----')
                for _class in classes:
                    logger.info(f'{_class:<25s}: {num_files[_class]}')
                # compute number of samples per worker
                totals = np.sum(list(num_files.values()), axis=0)
                logger.info(f'Total samples per worker: {totals}')
                logger.info('--------------------------------------')
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    cfg.data.data_dir = tmp_dir
    logger.info(f'Preload Complete [rank {rank}]. "data_dir" is changed to: {cfg.data.data_dir}')
