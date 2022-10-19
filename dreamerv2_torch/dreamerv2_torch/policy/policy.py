from typing import Protocol,Dict

from torch import Tensor


from dreamerv2_torch.world_model import WorldModel

class Policy(Protocol):

    def policy(self, state: Dict[str, Tensor], sample: bool):
        ...

    def reset(self):
        ...

class TrainablePolicy(Policy):
    def initialize_optimizer(self):
        ...

    def train_batch(self, world_model: WorldModel, start, is_terminal, reward_fn):
        ...