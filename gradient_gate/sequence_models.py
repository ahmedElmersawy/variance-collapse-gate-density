"""P4 -- broader architecture coverage: is the smooth-activation gate-density
rise CNN-specific, or does it also show up in non-convolutional
architectures? Adds two CIFAR-native (32x32), activation-configurable
architectures outside the CNN family: a small MLP-Mixer and a small
Transformer-Encoder.

Both are deliberately NOT torchvision/timm/nn.TransformerEncoderLayer
wrappers: nn.TransformerEncoderLayer's MLP activation, when given a callable,
is stored as a plain function reference called inside forward() -- it is not
a registered nn.Module submodule, so it never appears in named_modules() and
GateInstrumentor's hook-based gate recovery (instrumentation.py) cannot see
it. Every activation here is an explicit act_layer() submodule (the same
pattern cifar_models.py uses for the CNNs), so the existing instrumentation,
dataloaders, and seed-level statistics methodology all apply unchanged.
"""
import torch
import torch.nn as nn

SEQUENCE_NATIVE_ARCHS = ("mlp_mixer", "transformer_encoder")


class MlpMixerBlock(nn.Module):
    """Standard MLP-Mixer block (Tolstikhin et al. 2021): a token-mixing MLP
    (operates across the patch/token axis) and a channel-mixing MLP (operates
    across the hidden-dim axis), each pre-normed with a residual connection."""

    def __init__(self, num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim, act_layer):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_fc1 = nn.Linear(num_tokens, token_mlp_dim)
        self.token_act = act_layer()
        self.token_fc2 = nn.Linear(token_mlp_dim, num_tokens)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_fc1 = nn.Linear(hidden_dim, channel_mlp_dim)
        self.channel_act = act_layer()
        self.channel_fc2 = nn.Linear(channel_mlp_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, num_tokens, hidden_dim)
        y = self.norm1(x).transpose(1, 2)               # (batch, hidden_dim, num_tokens)
        y = self.token_fc2(self.token_act(self.token_fc1(y)))
        x = x + y.transpose(1, 2)

        y = self.norm2(x)
        y = self.channel_fc2(self.channel_act(self.channel_fc1(y)))
        return x + y


class CifarMlpMixer(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, depth=8, token_mlp_dim=64, channel_mlp_dim=512, act_layer=nn.GELU):
        super().__init__()
        assert img_size % patch_size == 0
        num_tokens = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            MlpMixerBlock(num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim, act_layer)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (batch, num_tokens, hidden_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm transformer encoder block: nn.MultiheadAttention
    (attention's softmax is intentionally NOT instrumented as a 'gate' --
    out of scope by this project's own definition, see instrumentation.py's
    module docstring: Gamma is a per-unit elementwise quantity, not a
    non-elementwise mixing op) followed by a 2-layer FFN with an explicit,
    instrumentable act_layer() submodule."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, act_layer):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        y = self.norm2(x)
        x = x + self.fc2(self.act(self.fc1(y)))
        return x


class CifarTransformerEncoder(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, depth=6, num_heads=4, mlp_dim=256, act_layer=nn.GELU):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_dim, act_layer)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (batch, num_patches, hidden_dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)[:, 0]
        return self.head(x)


def build_sequence_model(name, num_classes=10, in_channels=3, activation="relu"):
    from gradient_gate.cifar_models import ABLATION_ACTIVATIONS
    act_layer = ABLATION_ACTIVATIONS[activation]
    if name == "mlp_mixer":
        return CifarMlpMixer(num_classes=num_classes, in_channels=in_channels, act_layer=act_layer)
    if name == "transformer_encoder":
        return CifarTransformerEncoder(num_classes=num_classes, in_channels=in_channels, act_layer=act_layer)
    raise ValueError(f"no sequence-model variant for '{name}'")
