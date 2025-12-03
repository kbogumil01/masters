import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional

from .dense import DenseNet
from .res import ResNet
from .conv import ConvNet
from ..config import EnhancerConfig, NetworkImplementation


class MetadataEncoder(nn.Module):
    """Encoder of metadata"""
    @validate_arguments
    def __init__(self, metadata_size: int = 6, metadata_features: int = 64, size: int = 132, num_of_blocks: int = 2) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor, size: None | int | tuple[int, int] = None) -> Tensor:
        if size is None:
            size = self.size
        x = torch.nn.functional.interpolate(x, size=size)
        return x


class Enhancer(nn.Module):
    """Enhancer network"""
    @validate_arguments
    def __init__(self, config: EnhancerConfig) -> None:
        super().__init__()
        self.with_mask = config.with_mask
        self.metadata_encoder = MetadataEncoder(
            metadata_size=config.metadata_size,
            metadata_features=config.metadata_features,
            size=config.input_shape[0],
        )

        # BASELINE: 3 (RGB) + metadata_features (4)
        num_features = config.metadata_features + config.input_shape[2]

        # TODO: Odkomentuj dla Eksperymentu 2 i 3
        # if getattr(config, "use_vvc_features", False):
        #     num_features += 6

        self.model = {
            NetworkImplementation.DENSE: DenseNet,
            NetworkImplementation.RES: ResNet,
            NetworkImplementation.CONV: ConvNet,
        }[config.implementation](
            config,
            initial_features=num_features,
        )

    def forward(self, input_: Tensor, metadata: Tensor, vvc_features: Optional[Tensor] = None) -> Tensor:
        shape = input_.shape[2:]
        encoded_metadata = self.metadata_encoder(metadata, shape)
        
        # BASELINE: Łączymy tylko obraz i metadane
        data = torch.cat((input_, encoded_metadata), 1)

        # TODO: Odkomentuj dla Eksperymentu 2 i 3
        # if vvc_features is not None:
        #     data = torch.cat((data, vvc_features), 1)

        result = self.model(data)

        if self.with_mask:
            with_mask = torch.add(input_, result)
            return with_mask

        return result