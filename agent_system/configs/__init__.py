import os
from pathlib import Path
from omegaconf import OmegaConf

configs = [OmegaConf.load(os.environ.get('AOTD_AGENT_CONFIG', str(Path(__file__).with_name('base_config.yaml'))))]

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)
