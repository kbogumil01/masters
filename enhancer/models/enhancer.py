import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments

from .dense import DenseNet
from .res import ResNet
from .conv import ConvNet
from ..config import EnhancerConfig, NetworkImplementation


class MetadataEncoder(nn.Module):
    """
    Encoder of metadata
    """

    @validate_arguments
    def __init__(
        self,
        metadata_size: int = 6,
        metadata_features: int = 64,
        size: int = 132,
        num_of_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.size = size

    def forward(self, x: Tensor, size: None | int | tuple[int, int] = None) -> Tensor:
        if size is None:
            size = self.size

        x = torch.nn.functional.interpolate(x, size=size)
        return x


class VVCFeatureEncoder(nn.Module):
    """
    Encoder for VVC intelligence features (13 channels)
    Processes dequant coefficients, boundaries, and enhanced features
    """
    
    def __init__(self, in_channels: int = 13, out_channels: int = 16):
        super().__init__()
        
        # Lightweight encoder to compress VVC features
        self.encoder = nn.Sequential(
            # First conv: spatial analysis of VVC features
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            
            # Second conv: feature refinement  
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: VVC features (B, 13, H, W)
        Returns:
            Processed VVC features (B, 16, H, W)
        """
        return self.encoder(x)


class Enhancer(nn.Module):
    """
    Enhancer network
    """

    @validate_arguments
    def __init__(
        self,
        config: EnhancerConfig,
    ) -> None:
        super().__init__()

        self.with_mask = config.with_mask

        self.metadata_encoder = MetadataEncoder(
            metadata_size=config.metadata_size,
            metadata_features=config.metadata_features,
            size=config.input_shape[0],
        )

        # NEW: VVC Feature Encoder
        self.vvc_encoder = VVCFeatureEncoder(
            in_channels=13,   # 13 VVC intelligence channels
            out_channels=16,  # Compressed VVC features
        )

        # Total input features: RGB(3) + metadata(4) + VVC(16) = 23 channels
        num_features = config.metadata_features + config.input_shape[2] + 16

        self.model = {
            NetworkImplementation.DENSE: DenseNet,
            NetworkImplementation.RES: ResNet,
            NetworkImplementation.CONV: ConvNet,
        }[config.implementation](
            config,
            initial_features=num_features,
        )

    def forward(self, input_: Tensor, metadata: Tensor, vvc_features: Tensor = None) -> Tensor:
        shape = input_.shape[2:]
        encoded_metadata = self.metadata_encoder(metadata, shape)
        
        # Process VVC features if available
        if vvc_features is not None and vvc_features.numel() > 0:
            encoded_vvc = self.vvc_encoder(vvc_features)
            # Concatenate: RGB(3) + metadata(4) + VVC(16) = 23 channels
            data = torch.cat((input_, encoded_metadata, encoded_vvc), 1)
        else:
            # Fallback: create zero VVC features for backward compatibility
            batch_size = input_.size(0)
            zero_vvc = torch.zeros(batch_size, 16, *shape, device=input_.device, dtype=input_.dtype)
            data = torch.cat((input_, encoded_metadata, zero_vvc), 1)
        
        result = self.model(data)

        if self.with_mask:
            with_mask = torch.add(input_, result)
            return with_mask

        return result


if __name__ == "__main__":
    from torchsummary import summary
    import sys
    from ..config import Config

    config = Config.load(sys.argv[1])

    g = Enhancer(config.enhancer)
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g, [(3, 132, 132), (6, 1, 1)], device="cpu", depth=10)
    # summary(g, [(3, 1920, 1080), (6, 1, 1)], device="cpu", depth=10)
