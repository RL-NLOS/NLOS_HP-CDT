# HP-Net
import torch
import torch.nn as nn
import Self_attention
import CTF


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def upconv1x1(in_planes, out_planes, stride=2, output_padding=1):
    """2x2 upconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, output_padding=output_padding, bias=False)


def upconv(in_planes, out_planes, kernel_size=2, stride=2, padding=0):
    """2x2 upconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def upconv_nxn(in_planes, out_planes, kernel_size, stride):
    """nxn upconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False)


class fft_bench_complex_mlp_flops(nn.Module):
    def __init__(self, in_dim, out_dim, dw=1, norm='backward', window_size=0, bias=False):
        super(fft_bench_complex_mlp_flops, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.window_size = window_size
        hid_dim = out_dim * dw
        self.complex_weight1 = nn.Conv2d(in_dim*2, hid_dim*2, kernel_size=1, groups=2, bias=bias)
        self.complex_weight2 = nn.Conv2d(hid_dim*2, out_dim*2, kernel_size=1, groups=2, bias=bias)
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        y = torch.cat([y.real, y.imag], dim=1)
        y = self.complex_weight1(y)
        y = self.act_fft(y)
        y = self.complex_weight2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


# DB
class DownBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DownBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.In1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.In2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.In1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.In2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# UB
class UpBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(UpBlock, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(inplanes, planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes)
        )
        self.fft = fft_bench_complex_mlp_flops(inplanes, planes)
        self.In = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.residual = nn.Sequential(
            conv1x1(inplanes, planes)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample(x)
        identify = self.residual(x)
        c_out = self.conv(x)
        f_out = self.fft(x)
        out = self.relu(self.In(c_out + f_out + identify))
        return out


class FIU(nn.Module):
    def __init__(self, in_C, out_C):
        super(FIU, self).__init__()
        self.f_p = BasicConv(in_C, out_C, 3, 1, 1)
        self.f_l = BasicConv(in_C, out_C, 3, 1, 1)
        self.f_fusion = Fusion_module(channels=out_C)

    def forward(self, f_p, f_l):
        f_p_out = self.f_p(f_p)
        f_l_out = self.f_l(f_l)
        out = self.f_fusion(f_p_out, f_l_out)
        return out


class Fusion_module(nn.Module):
    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Channel_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.Sp_att1 = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.Sp_att2 = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.Channel_reduc = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.Sp_att3 = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.Channel_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        sp = torch.cat([self.Sp_att1(x1) * x1, self.Sp_att2(x2) * x2], dim=1)
        ca1_w = self.Channel_att1(input)
        ch = ca1_w * input
        fu = ch + sp
        # x1, x2 = torch.split(fu, c, dim=1)
        # xo = self.Channel_reduc(torch.cat([x1, x2], dim=1))
        xo = self.Channel_reduc(fu)
        sp3_w = self.Sp_att3(xo)
        ca2_w = self.Channel_att2(xo)
        return sp3_w * xo + ca2_w * xo


class BasicConv(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.InstanceNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class Fusion(nn.Module):
    def __init__(self, inplanes, planes, H, W, num_heads):

        super(Fusion, self).__init__()
        self.H = H
        self.W = W

        # channel expand
        self.conv1 = conv3x3(3, planes)
        self.In1 = nn.InstanceNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # HPFI fusion
        self.FIU = FIU(planes, inplanes)

        self.ctf = CTF.BasicLayer(dim=inplanes, depth=2, num_heads=num_heads, window_size=8,
                                                    mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.)

    def forward(self, x, y, z):

        identify_y = y

        # channel expand
        x = self.conv1(x)
        x = self.In1(x)
        x = self.relu1(x)

        # # HPFI fusion
        y = self.FIU(x, y)
        y = y + identify_y

        # CTF
        fusion = self.ctf(y) + z

        return y, fusion


class SimpleFusion(nn.Module):
    def __init__(self, inplanes, planes, H, W):

        super(SimpleFusion, self).__init__()
        self.H = H
        self.W = W

        # channel expand
        self.conv1 = conv3x3(3, planes)
        self.In1 = nn.InstanceNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # HPFI fusion
        self.In3 = nn.InstanceNorm2d(inplanes)
        self.FIU = FIU(256, 256)

    def forward(self, x, y, z):

        identify_y = y

        # channel expand
        x = self.conv1(x)
        x = self.In1(x)
        x = self.relu1(x)

        # HPFI fusion
        y = self.FIU(x, y)
        y = y + identify_y

        # Transformer output
        z = z.transpose(1, 2)
        z = z.view(z.shape[0], z.shape[1], self.H, self.W)

        fusion = y + self.In3(z)
        return fusion


class Encoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2],
                 num_heads=[4, 8, 16],
                 down_block = DownBlock, down_layers = [2, 2, 2],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):

        super(Encoder, self).__init__()
        # CNN branch
        self.inplanes_down = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.In1 = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Downsampling
        self.downlayer1 = self._make_downlayer(down_block, 64,  H=32, W=32, blocks=down_layers[0], stride=2)
        self.downlayer2 = self._make_downlayer(down_block, 128, H=16, W=16, blocks=down_layers[1], stride=2)
        self.downlayer3 = self._make_downlayer(down_block, 256, H=8,  W=8,  blocks=down_layers[2], stride=2)

        # Avgpool branch
        self.avgpool0 = nn.AvgPool2d(4)
        self.avgpool1 = nn.AvgPool2d(2)
        self.avgpool2 = nn.AvgPool2d(2)
        self.avgpool3 = nn.AvgPool2d(2)

        # Transformer branch
        self.num_layers = len(depths)
        self.patch_embed = Self_attention.PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Self_attention.BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=Self_attention.PatchMerging,
                               # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               )
            self.layers_down.append(layer)

        # fusion
        self.fusion0 = Fusion(inplanes=32,  planes=32,  H=64, W=64, num_heads=4)
        self.fusion1 = Fusion(inplanes=64,  planes=64,  H=32, W=32, num_heads=8)
        self.fusion2 = Fusion(inplanes=128, planes=128, H=16, W=16, num_heads=16)
        self.fusion3 = SimpleFusion(inplanes=256, planes=256, H=8, W=8)

    def _make_downlayer(self, block, planes, H, W, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes_down != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_down, planes * block.expansion, stride),
                nn.InstanceNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes_down, planes, H, W, stride, downsample))
        self.inplanes_down = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_down, planes, H, W))

        return nn.Sequential(*layers)

    def forward(self, x):

        # encoder
        a0 = self.avgpool0(x)
        d0 = self.maxpool(self.relu(self.In1(self.conv1(x))))
        t0, H, W = self.patch_embed(x)
        t0 = self.pos_drop(t0)
        c0, f0 = self.fusion0(a0, d0, t0)

        a1 = self.avgpool1(a0)
        d1 = self.downlayer1(c0)
        t1, H, W = self.layers_down[0](f0, H, W)
        c1, f1 = self.fusion1(a1, d1, t1)

        a2 = self.avgpool2(a1)
        d2 = self.downlayer2(c1)
        t2, H, W = self.layers_down[1](f1, H, W)
        c2, f2 = self.fusion2(a2, d2, t2)

        a3 = self.avgpool3(a2)
        d3 = self.downlayer3(c2)
        t3, H, W = self.layers_down[2](f2, H, W)
        f3 = self.fusion3(a3, d3, t3)

        return f3


class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()

        # Upsampling
        self.uplayer1 = UpBlock(256, 256)
        self.uplayer2 = UpBlock(256, 128)
        self.uplayer3 = UpBlock(128, 64)
        self.uplayer4 = UpBlock(64, 32)
        self.uplayer5 = UpBlock(32, 32)

        # refine
        self.conv_final = conv1x1(32, 3)
        self.tanh = nn.Tanh()


    def forward(self, x):

        # decoder
        u1 = self.uplayer1(x)
        u2 = self.uplayer2(u1)
        u3 = self.uplayer3(u2)
        u4 = self.uplayer4(u3)
        u5 = self.uplayer5(u4)

        cf = self.conv_final(u5)
        out = self.tanh(cf)

        return out


class HP_Net(nn.Module):
    def __init__(self):
        super(HP_Net, self).__init__()

        self.encoder = Encoder(down_block = DownBlock, down_layers = [2, 2, 2])
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y