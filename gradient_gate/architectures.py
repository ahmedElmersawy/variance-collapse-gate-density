"""Factory functions for REAL, off-the-shelf architectures that
GateInstrumentor can analyze with zero modification — contrast with
run_experiments.py Phase 3A's AlphaResNet18/AlphaVGG11, which had to replace
every ReLU with a custom AlphaSigmoid to get an analyzable gate (see that
file's "MODELING DECISION" comment). torchvision models are built with
weights=None (random init): gate-collapse / trainability questions are about
the architecture+activation landscape at/near initialization, not about a
specific pretrained checkpoint, and random init avoids a network download
dependency for every experiment run.
"""
import torch.nn as nn
import torchvision

SUPPORTED = ("resnet18", "resnet50", "vgg11", "vit_b_16", "convnext_tiny")


def _disable_inplace(model: nn.Module) -> nn.Module:
    """Inplace activations (torchvision's default) corrupt the
    grad_input/grad_output chain-rule trick GateInstrumentor relies on: the
    input tensor is overwritten before autograd can attribute a clean
    grad_input to it. Force out-of-place everywhere instrumentation is used."""
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def build_model(name: str, num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"unknown architecture '{name}', expected one of {SUPPORTED}")
    ctor = getattr(torchvision.models, name)
    model = ctor(weights=None, num_classes=num_classes)
    if in_channels != 3:
        if name.startswith("resnet"):
            model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=7,
                                     stride=2, padding=3, bias=False)
        elif name == "vgg11":
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    return _disable_inplace(model)


def input_size_for(name: str) -> int:
    """ResNet/VGG tolerate small inputs, but ViT-B/16 hardcodes a 14x14 patch
    grid for 224x224 input (patch size 16) and ConvNeXt's stem/downsampling
    stages are tuned for ImageNet-scale inputs; use 224 for those, 64 for the
    conv nets (big enough to exercise every stage's stride)."""
    if name in ("vit_b_16", "convnext_tiny"):
        return 224
    return 64
