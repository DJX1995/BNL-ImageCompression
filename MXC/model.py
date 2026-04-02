import torch
import torch.nn as nn
from typing import Optional, Tuple


def to_grayscale_model(model: nn.Module, weight_transfer: str = "average") -> nn.Module:
    """
    Convert a model from RGB (3-channel) to grayscale (1-channel) input/output.
    Handles skip connections in ResidualBlockWithStride and similar blocks.
    
    Args:
        model: The model to convert
        weight_transfer: Method for transferring weights from RGB to grayscale
                       - "average": Average RGB channels for first layer
                       - "sum": Sum RGB channels for first layer  
                       - "first": Use only first RGB channel
                       - "random": Random initialization
    
    Returns:
        Modified model with grayscale input/output
    """
    # Find all Conv2d layers that need conversion
    layers_to_convert = []
    skip_connections = []
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Check if this is a layer that takes RGB input (3 channels) OR outputs RGB (3 channels)
            if m.in_channels == 3 or m.out_channels == 3:
                layers_to_convert.append(m)
        # Check for ResidualBlockWithStride or similar blocks with skip connections
        elif hasattr(m, 'skip') and m.skip is not None and isinstance(m.skip, nn.Conv2d):
            if m.skip.in_channels == 3:
                skip_connections.append(m.skip)
        # Check for subpel_conv3x3 layers that output RGB
        elif isinstance(m, nn.Sequential) and len(m) == 2:
            # Check if this is a subpel_conv3x3: Conv2d + PixelShuffle
            if (isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.PixelShuffle) and 
                m[0].out_channels == 3 * m[1].upscale_factor**2):
                layers_to_convert.append(m)
    
    print(f"Found {len(layers_to_convert)} layers with RGB input/output to convert")
    print(f"Found {len(skip_connections)} skip connections with RGB input to convert")
    
    # Convert all RGB input/output layers
    replacements = []
    for conv_layer in layers_to_convert:
        if isinstance(conv_layer, nn.Conv2d):
            if conv_layer.in_channels == 3:  # Input layer (RGB -> grayscale)
                new_layer = nn.Conv2d(
                    in_channels=1,
                    out_channels=conv_layer.out_channels,
                    kernel_size=conv_layer.kernel_size,
                    stride=conv_layer.stride,
                    padding=conv_layer.padding,
                    dilation=conv_layer.dilation,
                    groups=conv_layer.groups,
                    bias=(conv_layer.bias is not None),
                    padding_mode=conv_layer.padding_mode,
                )
                _transfer_first_layer_weights(conv_layer, new_layer, weight_transfer)
                replacements.append((conv_layer, new_layer))
                print(f"Prepared INPUT conversion: {conv_layer.in_channels}->{conv_layer.out_channels} -> {new_layer.in_channels}->{new_layer.out_channels}")
                
            elif conv_layer.out_channels == 3:  # Output layer (grayscale -> RGB)
                new_layer = nn.Conv2d(
                    in_channels=conv_layer.in_channels,
                    out_channels=1,
                    kernel_size=conv_layer.kernel_size,
                    stride=conv_layer.stride,
                    padding=conv_layer.padding,
                    dilation=conv_layer.dilation,
                    groups=conv_layer.groups,
                    bias=(conv_layer.bias is not None),
                    padding_mode=conv_layer.padding_mode,
                )
                _transfer_last_layer_weights(conv_layer, new_layer, weight_transfer)
                replacements.append((conv_layer, new_layer))
                print(f"Prepared OUTPUT conversion: {conv_layer.in_channels}->{conv_layer.out_channels} -> {new_layer.in_channels}->{new_layer.out_channels}")
        
        elif isinstance(conv_layer, nn.Sequential):  # subpel_conv3x3 case
            # This is a subpel_conv3x3: Conv2d + PixelShuffle
            conv_part = conv_layer[0]  # The Conv2d part
            pixel_shuffle = conv_layer[1]  # The PixelShuffle part
            
            # Create new Conv2d that outputs 1 * upscale_factor**2 channels
            new_conv = nn.Conv2d(
                in_channels=conv_part.in_channels,
                out_channels=1 * pixel_shuffle.upscale_factor**2,
                kernel_size=conv_part.kernel_size,
                stride=conv_part.stride,
                padding=conv_part.padding,
                dilation=conv_part.dilation,
                groups=conv_part.groups,
                bias=(conv_part.bias is not None),  # Match the original bias setting
                padding_mode=conv_part.padding_mode,
            )
            
            # Transfer weights from RGB to grayscale
            _transfer_last_layer_weights(conv_part, new_conv, weight_transfer)
            
            # Create new Sequential with Conv2d + PixelShuffle
            new_layer = nn.Sequential(new_conv, pixel_shuffle)
            replacements.append((conv_layer, new_layer))
            print(f"Prepared SUBPEL conversion: {conv_part.in_channels}->{conv_part.out_channels} -> {new_conv.in_channels}->{new_conv.out_channels}")
    
    # Convert skip connections
    for skip_conv in skip_connections:
        new_skip = nn.Conv2d(
            in_channels=1,
            out_channels=skip_conv.out_channels,
            kernel_size=skip_conv.kernel_size,
            stride=skip_conv.stride,
            padding=skip_conv.padding,
            dilation=skip_conv.dilation,
            groups=skip_conv.groups,
            bias=(skip_conv.bias is not None),
            padding_mode=skip_conv.padding_mode,
        )
        _transfer_first_layer_weights(skip_conv, new_skip, weight_transfer)
        replacements.append((skip_conv, new_skip))
        print(f"Prepared skip conversion: {skip_conv.in_channels}->{skip_conv.out_channels} -> {new_skip.in_channels}->{new_skip.out_channels}")
    
    # Apply all replacements
    for old_layer, new_layer in replacements:
        success = _replace_layer_in_model(model, old_layer, new_layer)
        if not success:
            print(f"Warning: Failed to replace layer - this might cause issues")
    
    return model


def _transfer_first_layer_weights(old_layer: nn.Conv2d, new_layer: nn.Conv2d, method: str):
    """Transfer weights from RGB first layer to grayscale first layer."""
    with torch.no_grad():
        if method == "average":
            # Average across RGB channels: (out_channels, 3, H, W) -> (out_channels, 1, H, W)
            new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)
        elif method == "sum":
            # Sum across RGB channels
            new_layer.weight.data = old_layer.weight.data.sum(dim=1, keepdim=True)
        elif method == "first":
            # Use only the first RGB channel
            new_layer.weight.data = old_layer.weight.data[:, 0:1, :, :]
        elif method == "random":
            # Random initialization
            nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown weight transfer method: {method}")
        
        # Handle bias correctly - only copy if both layers have bias
        if new_layer.bias is not None and old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data.clone()
        elif new_layer.bias is not None and old_layer.bias is None:
            # New layer has bias but old doesn't - initialize to zero
            nn.init.zeros_(new_layer.bias)
        # If new_layer.bias is None, we don't need to do anything


def _transfer_last_layer_weights(old_layer: nn.Conv2d, new_layer: nn.Conv2d, method: str):
    """Transfer weights from RGB last layer to grayscale last layer."""
    with torch.no_grad():
        if method == "average":
            # Average across output channels: (3, in_channels, H, W) -> (1, in_channels, H, W)
            new_layer.weight.data = old_layer.weight.data.mean(dim=0, keepdim=True)
        elif method == "sum":
            # Sum across output channels
            new_layer.weight.data = old_layer.weight.data.sum(dim=0, keepdim=True)
        elif method == "first":
            # Use only the first output channel
            new_layer.weight.data = old_layer.weight.data[0:1, :, :, :]
        elif method == "random":
            # Random initialization
            nn.init.zeros_(new_layer.weight)
        else:
            raise ValueError(f"Unknown weight transfer method: {method}")
        
        # Handle bias correctly - only copy if both layers have bias
        if new_layer.bias is not None and old_layer.bias is not None:
            if method == "first":
                new_layer.bias.data = old_layer.bias.data[0:1]
            else:
                # For subpel_conv3x3, we need to handle the case where output channels change
                if old_layer.bias.shape[0] != new_layer.bias.shape[0]:
                    # For subpel_conv3x3: 12 -> 4 channels, we need to average in groups
                    # 12 channels -> 4 channels means we average every 3 channels
                    old_bias = old_layer.bias.data
                    new_bias = old_bias.view(4, 3).mean(dim=1)  # Reshape and average
                    new_layer.bias.data = new_bias
                else:
                    new_layer.bias.data = old_layer.bias.data.clone()
        elif new_layer.bias is not None and old_layer.bias is None:
            # New layer has bias but old doesn't - initialize to zero
            nn.init.zeros_(new_layer.bias)
        # If new_layer.bias is None, we don't need to do anything


def _replace_layer_in_model(model: nn.Module, old_layer: nn.Module, new_layer: nn.Module) -> bool:
    """Replace a layer in the model hierarchy."""
    for name, child in model.named_children():
        if child is old_layer:
            setattr(model, name, new_layer)
            return True
        elif _replace_layer_in_model(child, old_layer, new_layer):
            return True
    return False

