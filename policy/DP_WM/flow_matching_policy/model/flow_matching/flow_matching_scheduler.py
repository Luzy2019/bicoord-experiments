from types import SimpleNamespace


class FlowMatchingSchedulerConfig:
    """Small Hydra-instantiated config holder for FM time discretization.

    Existing policies historically received a DDPM scheduler object with a
    `.config.num_train_timesteps` attribute. Keeping that shape avoids touching
    workspace/checkpoint plumbing while removing DDPM sampling semantics.
    """

    def __init__(self, num_train_timesteps: int = 100, path: str = "rectified"):
        self.config = SimpleNamespace(
            num_train_timesteps=int(num_train_timesteps),
            path=path,
        )
