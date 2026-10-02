"""Models: pseudo-Siamese multitask ResNet-18 classifier and a Siamese ResNet-18 U-Net for damage segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as tv_resnet18

from . import config as C

RESNET_STAGES = ["layer1", "layer2", "layer3", "layer4"]


def resnet18(pretrained=True):
    """torchvision ResNet-18, with ImageNet weights from the timm GitHub release (SHA-256 checked)."""
    model = tv_resnet18(weights=None)
    if pretrained:
        state = torch.hub.load_state_dict_from_url(C.RESNET18_WEIGHTS_URL, map_location="cpu", check_hash=True)
        model.load_state_dict(state)
    return model


class ResNetEncoder(nn.Module):
    """ResNet-18 without its classifier. Returns the feature maps of the stem and the four stages."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = resnet18(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # stride 2, 64 ch
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [s0, s1, s2, s3, s4]


def _freeze_bn_of_frozen(module):
    """Keep pretrained BatchNorm statistics in layers whose parameters are frozen."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


class MultitaskSiameseResNet18(nn.Module):
    """Two separate (pseudo-Siamese) ResNet-18 encoders for the pre- and post-event images.

    Global-average-pooled features f_pre, f_post and their difference are concatenated and fed to a
    shared dropout-regularised layer, then to two softmax heads: disaster type and severity.
    """

    # Progressive fine-tuning: (first epoch, trainable encoder stages). Earlier stages stay frozen longer.
    UNFREEZE_SCHEDULE = [
        (1, []),
        (4, ["layer4"]),
        (8, ["layer3", "layer4"]),
        (12, ["layer2", "layer3", "layer4"]),
        (16, ["stem", "layer1", "layer2", "layer3", "layer4"]),
    ]

    def __init__(self, pretrained=True, dropout=0.5, hidden=256):
        super().__init__()
        self.encoder_pre = ResNetEncoder(pretrained)
        self.encoder_post = ResNetEncoder(pretrained)
        feat = self.encoder_pre.channels[-1]  # 512
        self.shared = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(3 * feat, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.type_head = nn.Linear(hidden, len(C.DISASTER_TYPES))
        self.severity_head = nn.Linear(hidden, len(C.SEVERITIES))
        self.trainable_stages = None
        self.set_trainable([])

    def encode(self, encoder, x):
        return F.adaptive_avg_pool2d(encoder(x)[-1], 1).flatten(1)

    def forward(self, pre, post):
        f_pre, f_post = self.encode(self.encoder_pre, pre), self.encode(self.encoder_post, post)
        h = self.shared(torch.cat([f_pre, f_post, f_post - f_pre], dim=1))
        return self.type_head(h), self.severity_head(h)

    def backbone_parameters(self):
        return list(self.encoder_pre.parameters()) + list(self.encoder_post.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder_")]

    def set_trainable(self, stages):
        """Freeze both encoders, then unfreeze the named stages ('stem' also covers the first BatchNorm)."""
        self.trainable_stages = list(stages)
        for enc in (self.encoder_pre, self.encoder_post):
            for p in enc.parameters():
                p.requires_grad = False
            for name in stages:
                for p in getattr(enc, name).parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            _freeze_bn_of_frozen(self.encoder_pre)
            _freeze_bn_of_frozen(self.encoder_post)
        return self

    def unfreeze_for_epoch(self, epoch):
        """Apply the progressive schedule; returns a description if the trainable set changed."""
        target = []
        for start, stages in self.UNFREEZE_SCHEDULE:
            if epoch >= start:
                target = stages
        if target != self.trainable_stages:
            self.set_trainable(target)
            return "all encoder layers" if "stem" in target else "encoder " + ", ".join(target)
        return None

    @property
    def unfrozen_level(self):
        """Number of trainable encoder stages (0 = frozen, 5 = everything), for logging and plots."""
        return len(self.trainable_stages)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseResNetUNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) with a shared-weight ResNet-18 encoder applied to pre and post images.

    At every scale the pre and post feature maps and their absolute difference are fused by a 1x1 conv and used
    as the skip connection. Outputs per-pixel logits for background / intact building / damaged building.
    """

    def __init__(self, pretrained=True, n_classes=3, dropout=0.2):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained)
        ch = self.encoder.channels  # 64, 64, 128, 256, 512
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * c, c, 1, bias=False), nn.BatchNorm2d(c),
                                                 nn.ReLU(inplace=True)) for c in ch])
        dec = [256, 128, 64, 64, 32]
        self.decoder = nn.ModuleList([
            DecoderBlock(ch[4], ch[3], dec[0], dropout),  # 1/32 -> 1/16
            DecoderBlock(dec[0], ch[2], dec[1], dropout),  # -> 1/8
            DecoderBlock(dec[1], ch[1], dec[2]),           # -> 1/4
            DecoderBlock(dec[2], ch[0], dec[3]),           # -> 1/2
            DecoderBlock(dec[3], 0, dec[4]),               # -> full resolution
        ])
        self.out = nn.Conv2d(dec[4], n_classes, 1)

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def decoder_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, pre, post):
        fp, fq = self.encoder(pre), self.encoder(post)
        feats = [fuse(torch.cat([a, b, (b - a).abs()], dim=1)) for fuse, a, b in zip(self.fuse, fp, fq)]
        x = feats[4]
        for block, skip in zip(self.decoder, [feats[3], feats[2], feats[1], feats[0], None]):
            x = block(x, skip)
        return self.out(x)
