# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from . import resnet_backbone


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)
    

class fixedActionGeneratorAngle(nn.Module):

    def __init__(self):
        super(fixedActionGeneratorAngle, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class BasicBlockResNet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mid_planes=None, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockResNet, self).__init__()
        if not mid_planes:
            mid_planes = planes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, mid_planes, stride)
        self.bn1 = norm_layer(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample =  nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm_layer(planes),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UpResNet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlockResNet(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BasicBlockResNet(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class fixedBranchedCIMNetWithDepthAngle(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(fixedBranchedCIMNetWithDepthAngle, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        # self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = fixedActionGeneratorAngle()
        self.actionGenerator_left = fixedActionGeneratorAngle()
        self.actionGenerator_down = fixedActionGeneratorAngle()
        self.actionGenerator_right = fixedActionGeneratorAngle()
        self.actionGenerator_straight = fixedActionGeneratorAngle()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        # feature_command = self.commandFeatureExtractor(x2)
        # feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_rgb[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        output = F.tanh(output)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output, output_depth


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = torch.mean(x, dim=4, keepdim=False)
    # x = x.permute(0, 1, 4, 2, 3)
    # x = torch.flatten(x, start_dim=1, end_dim=2)
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class myDQN(nn.Module):
  def __init__(self, args, action_space):
    super(myDQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
      self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
      self.fc = nn.Linear(self.conv_output_size * 2, self.conv_output_size)
    elif args.architecture == 'high-level-policy':
      self.convs = resnet_backbone.resnet34(pretrained=True)
      self.convs.fc = nn.Linear(512, 256)
      self.conv_output_size = 256
      self.fc = nn.Linear(self.conv_output_size * 3, self.conv_output_size)
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x_image = x[..., 0]
    x_ref = x[..., 1]
    x_cond = x[..., 2]
    # x_ref = x.clone()
    x_image = self.convs(x_image)
    x_ref = self.convs(x_ref)
    x_cond = self.convs(x_cond)
    x_image = x_image.view(-1, self.conv_output_size)
    x_ref = x_ref.view(-1, self.conv_output_size)
    x_cond = x_cond.view(-1, self.conv_output_size)
    x = torch.cat([x_image, x_ref, x_cond], dim=-1)
    x = self.fc(F.relu(x))
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class CommandGenerator(nn.Module):

    def __init__(self):
        super(CommandGenerator, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output

class HighLevelCIL(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(HighLevelCIL, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.rgbFeatureExtractor2 = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor2.fc = nn.Linear(512, 512)
        # self.commandFeatureExtractor = onehotFeatureExtractor()
        self.commandGenerator = CommandGenerator()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.depthDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)
        self.targetsDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.targetsDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.targetsDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.targetsDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.targetsDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.targetsDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2, x3):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        feature_ref_rgb, _, _, _, _, _ = self.rgbFeatureExtractor(x2)
        # x_fusion = torch.cat([x2[:, 0, :, :].unsqueeze(1), x2[:, 1, :, :].unsqueeze(1), torch.mean(x3, dim=1, keepdim=True)], dim=1)
        feature_condition, _, _, _, _, _ = self.rgbFeatureExtractor2(x3)
        # feature_condition, _, _, _, _, _ = self.rgbFeatureExtractor(x3)
        # x_fusion = torch.cat([torch.mean(x1, dim=1, keepdim=True), torch.mean(x2, dim=1, keepdim=True), torch.mean(x3, dim=1, keepdim=True)], dim=1)
        # feature_fusion, f1_fusion, f2_fusion, f3_fusion, f4_fusion, f5_fusion = self.rgbFeatureExtractor(x_fusion)
        output_command = self.commandGenerator(torch.cat([feature_rgb, feature_ref_rgb, feature_condition], dim=-1))
        # output_command = self.commandGenerator(torch.cat([feature_rgb, feature_condition], dim=-1))
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.depthDecoder_up5(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)

        # output_targets = self.targetsDecoder_up1(f5_fusion, f4_fusion)
        # output_targets = self.targetsDecoder_up2(output_targets, f3_fusion)
        # output_targets = self.targetsDecoder_up3(output_targets, f2_fusion)
        # output_targets = self.targetsDecoder_up4(output_targets, f1_fusion)
        # output_targets = self.targetsDecoder_up5(output_targets)
        # output_targets = self.targetsDecoder_outc(output_targets)

        output_targets = output_depth.clone()
        return output_command, output_depth, output_targets


class actionGeneratorAngle(nn.Module):

    def __init__(self):
        super(actionGeneratorAngle, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output
    

class SingleLevelCIL(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SingleLevelCIL, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.rgbFeatureExtractor2 = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor2.fc = nn.Linear(512, 512)
        self.actionGenerator1 = actionGeneratorAngle()
        self.actionGenerator2 = actionGeneratorAngle()
        self.commandGenerator = CommandGenerator()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.depthDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)
        self.targetsDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.targetsDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.targetsDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.targetsDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.targetsDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.targetsDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2, x3):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        feature_condition, _, _, _, _, _ = self.rgbFeatureExtractor2(x3)
        output_angle = self.actionGenerator1(torch.cat([feature_rgb, feature_condition], dim=-1))
        output_angle = F.tanh(output_angle)
        output_stop = self.actionGenerator2(torch.cat([feature_rgb, feature_condition], dim=-1))

        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.depthDecoder_up5(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)

        output_targets = output_depth.clone()
        return output_angle, output_stop, output_depth, output_targets
    

# class SingleLevelCILHigh(nn.Module):

#     def __init__(self, norm_layer=nn.BatchNorm2d):
#         super(SingleLevelCILHigh, self).__init__()
#         self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
#         self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
#         self.actionGenerator = actionGeneratorAngle()

#     def forward(self, x1, x2, x3):
#         feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
#         feature_ref, _, _, _, _, _ = self.rgbFeatureExtractor(x2)
#         output_stop = self.actionGenerator(torch.cat([feature_rgb, feature_rgb - feature_ref], dim=-1))

#         return output_stop


class actionGeneratorAngleLong(nn.Module):

    def __init__(self):
        super(actionGeneratorAngleLong, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class SingleLevelCILHigh(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SingleLevelCILHigh, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.actionGenerator = actionGeneratorAngleLong()

    def forward(self, x1, x2, x3):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        feature_ref, _, _, _, _, _ = self.rgbFeatureExtractor(x2)
        feature_condition, _, _, _, _, _ = self.rgbFeatureExtractor(x3)
        output_stop = self.actionGenerator(torch.cat([feature_rgb, feature_rgb - feature_ref, feature_condition], dim=-1))

        return output_stop
    

class targetsDetectionNet(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(targetsDetectionNet, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.depthDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)
        self.targetsDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.targetsDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.targetsDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.targetsDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.targetsDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.targetsDecoder_outc = OutConv(32, 1)

    def forward(self, x):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x)

        output_targets = self.targetsDecoder_up1(f5, f4)
        output_targets = self.targetsDecoder_up2(output_targets, f3)
        output_targets = self.targetsDecoder_up3(output_targets, f2)
        output_targets = self.targetsDecoder_up4(output_targets, f1)
        output_targets = self.targetsDecoder_up5(output_targets)
        output_targets = self.targetsDecoder_outc(output_targets)

        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.depthDecoder_up5(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)

        return output_targets, output_depth


class myDQN(nn.Module):
  def __init__(self, args, action_space):
    super(myDQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.convs2 = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
      self.fc = NoisyLinear(self.conv_output_size * 4, self.conv_output_size, std_init=args.noisy_std)
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
      self.fc = nn.Linear(self.conv_output_size * 2, self.conv_output_size)
    elif args.architecture == 'high-level-policy':
      self.convs = resnet_backbone.resnet34(input_channels=args.history_length, pretrained=True)
      self.convs.fc = nn.Linear(512, 512)
      self.conv_output_size = 512
      self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    elif args.architecture == 'mobilenet':
      self.convs = torchvision.models.mobilenet_v3_large(input_channels=args.history_length, pretrained=True)
    #   self.convs.fc = nn.Linear(512, 512)
      self.conv_output_size = 1000
      self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    # x_image = x[..., 0]
    # x_ref = x[..., 1]
    # x_cond = x[..., 2]
    x_image = x[:, -4:, :, :, 0]
    x_ref = x[:, -4:, :, :, 1]
    x_cond = x[:, -4:, :, :, 2]
    x_image_hist = x[..., 0]
    # x_ref = x.clone()
    x_image = self.convs(x_image)
    x_ref = self.convs(x_ref)
    x_cond = self.convs(x_cond)
    x_image_hist = self.convs2(x_image_hist)
    x_image = x_image.view(-1, self.conv_output_size)
    x_ref = x_ref.view(-1, self.conv_output_size)
    x_cond = x_cond.view(-1, self.conv_output_size)
    x_image_hist = x_image_hist.view(-1, self.conv_output_size)
    x = torch.cat([x_image, x_ref, x_cond, x_image_hist], dim=-1)
    x = self.fc(F.relu(x))
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class newDQN(nn.Module):
  def __init__(self, args, action_space):
    super(newDQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
        self.convs = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                    nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
    #   self.convs2 = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
    #                              nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
    #                              nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
    #   self.convs = nn.Sequential(nn.Conv3d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
    #                             nn.Conv3d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
    #                             nn.Conv3d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        self.conv_output_size = 3136
        self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    elif args.architecture == 'data-efficient':
        self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                    nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
        self.conv_output_size = 576
        self.fc = nn.Linear(self.conv_output_size * 2, self.conv_output_size)
    elif args.architecture == 'high-level-policy':
        self.convs = resnet_backbone.resnet34(input_channels=args.history_length, pretrained=True)
        self.convs.fc = nn.Linear(512, 512)
        self.conv_output_size = 512
        self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    elif args.architecture == 'mobilenet':
        self.convs = torchvision.models.mobilenet_v3_large(input_channels=args.history_length, pretrained=True)
    #   self.convs.fc = nn.Linear(512, 512)
        self.conv_output_size = 1000
        self.fc = NoisyLinear(self.conv_output_size * 3, self.conv_output_size, std_init=args.noisy_std)
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x_image = x[..., 0]
    # plt.imshow(x_image.cpu().data.numpy()[0, 0, :, :])
    # plt.show()
    x_ref = x[..., 1]
    # plt.imshow(x_ref.cpu().data.numpy()[0, 0, :, :])
    # plt.show()
    x_cond = x[..., 2]
    # plt.imshow(x_cond.cpu().data.numpy()[0, 0, :, :])
    # plt.show()
    # x_image = x[..., :3]
    # x_ref = x[..., 3 : 6]
    # x_cond = x[..., 6 : 9]
    # x_image = x_image.permute(0, 1, 4, 3, 2)
    # x_ref = x_ref.permute(0, 1, 4, 3, 2)
    # x_cond = x_cond.permute(0, 1, 4, 3, 2)
    # x_ref = x.clone()
    x_image = self.convs(x_image)
    x_ref = self.convs(x_ref)
    x_cond = self.convs(x_cond)
    x_image = x_image.view(-1, self.conv_output_size)
    x_ref = x_ref.view(-1, self.conv_output_size)
    x_cond = x_cond.view(-1, self.conv_output_size)
    # print("Distance:", torch.dot(x_ref.view(-1), x_image.view(-1)) / (torch.norm(x_ref) * torch.norm(x_image)))
    print("Distance:", torch.norm(x_ref - x_image))
    x = torch.cat([x_image, x_ref - x_image, x_ref * x_cond], dim=-1)
    x = self.fc(F.relu(x))
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
        q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
        if 'fc' in name:
            module.reset_noise()


class actionGeneratorAngleMultiFrame(nn.Module):

    def __init__(self):
        super(actionGeneratorAngleMultiFrame, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class SingleLevelCILMultiFrame(nn.Module):

    def __init__(self, history_length, norm_layer=nn.BatchNorm2d):
        super(SingleLevelCILMultiFrame, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(input_channels=history_length, norm_layer=nn.InstanceNorm2d, pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.actionGenerator1 = actionGeneratorAngleMultiFrame()
        self.actionGenerator2 = actionGeneratorAngleMultiFrame()

    def forward(self, x):
        x_img = x[..., 0]
        x_ref = x[..., 1]
        x_cond = x[..., 2]
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x_img)
        feature_ref, _, _, _, _, _ = self.rgbFeatureExtractor(x_ref)
        feature_cond, _, _, _, _, _ = self.rgbFeatureExtractor(x_cond)
        output_angle = self.actionGenerator1(torch.cat([feature_rgb, feature_ref - feature_rgb, feature_ref * feature_cond], dim=-1))
        output_angle = F.tanh(output_angle)
        output_stop = self.actionGenerator2(torch.cat([feature_rgb, feature_ref - feature_rgb, feature_ref * feature_cond], dim=-1))

        return output_angle, output_stop


  
class SingleLevelCILMultiFrameEnd2end(nn.Module):

    def __init__(self, input_channels):
        super(SingleLevelCILMultiFrameEnd2end, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(input_channels=input_channels, norm_layer=nn.InstanceNorm2d, pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.actionGenerator1 = actionGeneratorAngleMultiFrame()
        self.actionGenerator2 = actionGeneratorAngleMultiFrame()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, input_channels)
        self.targetsDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.targetsDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.targetsDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.targetsDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.targetsDecoder_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.targetsDecoder_outc = OutConv(32, 1)

    def forward(self, x):
        x_img = x[..., 0]
        x_ref = x[..., 1]
        x_cond = x[..., 2]
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x_img)
        feature_ref, _, _, _, _, _ = self.rgbFeatureExtractor(x_ref)
        feature_cond, _, _, _, _, _ = self.rgbFeatureExtractor(x_cond)
        output_angle = self.actionGenerator1(torch.cat([feature_rgb, feature_ref - feature_rgb, feature_ref * feature_cond], dim=-1))
        output_angle = F.tanh(output_angle)
        output_stop = self.actionGenerator2(torch.cat([feature_rgb, feature_ref - feature_rgb, feature_ref * feature_cond], dim=-1))
        f6 = self.depthDecoder_up1(f5, f4)
        f7 = self.depthDecoder_up2(f6, f3)
        f8 = self.depthDecoder_up3(f7, f2)
        f9 = self.depthDecoder_up4(f8, f1)
        f10 = self.up(f9)
        output_depth = self.depthDecoder_outc(f10)
        f6_targets = self.targetsDecoder_up1(f5, f4)
        f7_targets = self.targetsDecoder_up2(f6_targets, f3)
        f8_targets = self.targetsDecoder_up3(f7_targets, f2)
        f9_targets = self.targetsDecoder_up4(f8_targets, f1)
        f10_targets = self.targetsDecoder_up(f9_targets)
        output_targets = self.targetsDecoder_outc(f10_targets)

        return output_angle, output_stop, output_depth, output_targets


class ForeignBodyNet(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ForeignBodyNet, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(input_channels=3, num_classes=2, pretrained=True)
        # self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        # self.fc = nn.Linear(512, 512)
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.depthDecoder_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x):
        output_cls, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x)

        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.depthDecoder_up5(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)

        return output_cls, output_depth

