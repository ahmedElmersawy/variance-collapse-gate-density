"""CIFAR-native (32x32) architecture variants with REAL activations (nn.ReLU)
— for the training-dynamics study (does gate collapse emerge during NORMAL
supervised training?), using an artificially-injected AlphaSigmoid gate (as
in run_experiments.py Phase 3A) would defeat the purpose: we need the
network's own, unmodified nonlinearity.

torchvision's resnet18/resnet50/vgg11 (gradient_gate.architectures) use an
ImageNet-shaped stem (7x7 stride-2 conv + maxpool for ResNet; 5 maxpools
sized for a 224x224 input for VGG) that collapses a 32x32 CIFAR image's
spatial resolution to ~1x1 within the first couple of layers — a real
training run on that stem would conflate "gate collapse" with "the stem
destroyed the image," which is not the phenomenon under study. The standard
fix in the CIFAR-ResNet/VGG literature (used here) is a 3x3 stride-1 stem
with no initial maxpool, keeping the rest of the topology (block counts,
channel widths, stage strides) identical to the canonical architecture.

ViT-B/16 and ConvNeXt-Tiny are NOT reimplemented here — they keep their
canonical torchvision form and CIFAR images are resized to 224 for them
(see run_training_dynamics.py), since shrinking patch_size/stem to fit 32x32
would no longer be "ViT-B/16"/"ConvNeXt-Tiny" by name.

ACTIVATION ABLATION (architecture-fixed, activation-varied): every
constructor below also accepts `act_layer`, a zero-arg callable returning a
fresh activation module (default nn.ReLU). This exists specifically to
separate activation effects from architecture effects: ResNet-18/VGG-11
with GELU/SiLU/Mish swapped in, topology otherwise identical, lets us ask
whether the ReLU-vs-ViT-B/16 gate-density contrast tracks the activation
function or the CNN-vs-Transformer architecture family.
"""
import torch.nn as nn

CIFAR_NATIVE_ARCHS = ("resnet18", "resnet50", "vgg11")
ABLATION_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
}


def _default_norm(num_channels):
    return nn.BatchNorm2d(num_channels)


def groupnorm_layer(num_channels):
    """Standard, well-behaved BatchNorm replacement for the BN-necessity
    ablation: num_groups capped at 32 (or num_channels if smaller, e.g. the
    64-channel stem still gets 32 groups of 2 channels each)."""
    return nn.GroupNorm(min(32, num_channels), num_channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, act_layer=nn.ReLU, norm_layer=_default_norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), norm_layer(out_ch))
        self.act2 = act_layer()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act2(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, act_layer=nn.ReLU, norm_layer=_default_norm):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = norm_layer(mid_ch)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(mid_ch)
        self.act2 = act_layer()
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = norm_layer(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), norm_layer(out_ch))
        self.act3 = act_layer()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.act3(out)


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, widths=(64, 128, 256, 512), strides=(1, 2, 2, 2),
                 in_channels=3, num_classes=10, act_layer=nn.ReLU, norm_layer=_default_norm):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.stem_bn = norm_layer(64)
        self.stem_act = act_layer()
        layers = []
        in_ch = 64
        for w, s, n in zip(widths, strides, num_blocks):
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(in_ch, w, stride=stride, act_layer=act_layer, norm_layer=norm_layer))
                in_ch = w * block.expansion
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def cifar_resnet18(num_classes=10, in_channels=3, act_layer=nn.ReLU, norm_layer=_default_norm):
    return CifarResNet(BasicBlock, num_blocks=(2, 2, 2, 2), in_channels=in_channels,
                        num_classes=num_classes, act_layer=act_layer, norm_layer=norm_layer)


def cifar_resnet50(num_classes=10, in_channels=3, act_layer=nn.ReLU):
    return CifarResNet(Bottleneck, num_blocks=(3, 4, 6, 3), in_channels=in_channels,
                        num_classes=num_classes, act_layer=act_layer)


class CifarVGG11(nn.Module):
    """VGG-11 ('A') topology, BN + activation, sized for 32x32 (5 maxpools ->
    1x1 spatial), matching run_experiments.py's AlphaVGG11 topology exactly
    but with a real, configurable activation instead of the alpha-sigmoid gate."""
    CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self, num_classes=10, in_channels=3, act_layer=nn.ReLU):
        super().__init__()
        layers = []
        in_ch = in_channels
        for v in self.CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_ch, v, 3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(v))
                layers.append(act_layer())
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512, 256), act_layer(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_cifar_model(name, num_classes=10, in_channels=3, activation="relu", norm="batchnorm"):
    act_layer = ABLATION_ACTIVATIONS[activation]
    norm_layer = groupnorm_layer if norm == "groupnorm" else _default_norm
    if name == "resnet18":
        return cifar_resnet18(num_classes, in_channels, act_layer=act_layer, norm_layer=norm_layer)
    if name == "resnet50":
        return cifar_resnet50(num_classes, in_channels, act_layer=act_layer)
    if name == "vgg11":
        return CifarVGG11(num_classes, in_channels, act_layer=act_layer)
    raise ValueError(f"no CIFAR-native variant for '{name}'")
