from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arch_utils import LayerNorm2d, MySequential

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(LayerNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)





class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()


        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        return y
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.relu3 = SimpleGate()
        self.relu5 = SimpleGate()
        self.dwconv3x3_1 = nn.Conv2d(hidden_features//2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features//2 , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features//2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features//2 , bias=bias)

        self.relu3_1 = SimpleGate()
        self.relu5_1 = SimpleGate()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class NAFBlock2(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        self.layers = SCABlock(c)
        
        self.ffn = FeedForward(c, FFN_Expand)
       

        self.norm2 = LayerNorm2d(c)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        y = self.layers(inp)
        x = self.ffn(self.norm2(y))
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFBlock(nn.Module):
    def __init__(self, c, num_blk, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        sca = [SCABlock(c) for _ in range(num_blk)]
        self.sca = nn.Sequential(*sca)
        # SimpleGate
        self.sg = SimpleGate()

        self.ffn = FeedForward(c, FFN_Expand)
      

        self.norm2 = LayerNorm2d(c)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
       

    def forward(self, inp):

        y = self.sca(inp)
        x = self.ffn(self.norm2(y))
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFBlock3(nn.Module):
    def __init__(self, c, num_blk):
        super().__init__()
        layers = [NAFBlock2(c) for _ in range(num_blk)]
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        return self.layers(inp)
    


class MSBlcok(nn.Module):
    def __init__(self, c, num_blk, j_alfa=1):
        super().__init__()
        self.down = nn.Conv2d(c, c//2, kernel_size=2, stride=2, groups=c//2)
        self.down2 = nn.Conv2d(c//2, c//4, kernel_size=2, stride=2, groups=c//4)
        self.down3 = nn.Conv2d(c//2, c//4, kernel_size=2, stride=2, groups=c//4)
        self.n1 = NAFBlock(c, num_blk)
        # self.n1 = nn.Sequential(*n1)
        self.n2 = NAFBlock(c//2,num_blk)
        # self.n2 = nn.Sequential(*n2)
        self.n3 = NAFBlock(c//4,num_blk)
        # self.n3 = nn.Sequential(*n3)
        self.up1 = nn.Sequential(
                    DySample(c//2,2),
                    nn.Conv2d(c//2,c,1)
                )
        self.up2 = nn.Sequential(
                    DySample(c//4,4),
                    nn.Conv2d(c//4,c,1)
                )
        self.j_alfa = j_alfa
        
 
     
    def forward(self, inp):  
        x_1 = inp
        x_2 = self.down(inp)
        x_3 = self.down2(x_2)
        out_1 = self.n1(x_1)
        out_2 = self.n2(x_2)
        out_2d = self.down3(out_2)
        out_3 = self.n3(x_3+out_2d)
        out_2 = self.up1(out_2)
        out_3 = self.up2(out_3)
        return out_1 + out_2 + out_3 * self.j_alfa




class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
       

    def forward(self, x):

        return x

class sca_layer(nn.Module):
    def __init__(self, channel):
        super().__init__()


        
    def forward(self, x):

        return x 


class DAU(nn.Module):
    def __init__(self, n_feat):
        super(DAU, self).__init__()
        ## Spatial Attention
        self.SA = SpatialGate(n_feat)
        ## Channel Attention        
        self.CA = sca_layer(n_feat)
        self.inside_all = nn.Parameter(torch.zeros(n_feat,1,1), requires_grad=True)

    def forward(self, x):
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        res = ca_branch * self.inside_all + sa_branch
        return res

class HighF(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(HighF, self).__init__()
    

    def forward(self, x):
     
        return x

class SFM(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
      

    def forward(self, x):
        return x






class SFSM(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        self.dau = DAU(channel)
        self.sfm = SFM(channel)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))
        self.c = nn.Parameter(torch.zeros(channel,1,1))
        self.d = nn.Parameter(torch.ones(channel,1,1))

        self.inside_all = nn.Parameter(torch.zeros(channel,1,1), requires_grad=True)
    def forward(self, x):
        
        # save_feature_maps(x,'e_i')
        out = self.dau(x)
        # save_feature_maps(out,'e_dau')
        out = self.sfm(out)
        # save_feature_maps(out,'e_o')
        out2 = self.sfm(x)
        # save_feature_maps(out2,'e_sfm')
        out2 = self.dau(out2)
        # save_feature_maps(out2,'e_o2')


        out_1 = self.a*out + self.b*x
        out_2 = self.c*out2 + self.d*x

        # save_feature_maps(out + out2,'e_o5')
        return out_1 + out_2 * self.inside_all








class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    




class ECFNet(nn.Module):
    def __init__(self, base_channel = 32, num_res=8, alfa=1):
        super(ECFNet, self).__init__()

        
   
        self.Encoder = nn.ModuleList([
            MSBlcok(base_channel, num_res,alfa),
            NAFBlock3(base_channel*2,num_res),
            NAFBlock3(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            NAFBlock3(base_channel * 4, num_res),
            NAFBlock3(base_channel * 2, num_res),
            MSBlcok(base_channel, num_res,alfa)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        pyramid_attention = []
        for _ in range(1):
            pyramid_attention.append(SFSM(base_channel * 4))
        self.pyramid_attentions = nn.Sequential(*pyramid_attention)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
    
   
        z = self.pyramid_attentions(z)
   
        
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size).cuda()

            if m.output_size == 1:
                setattr(model, n, pool)
            # assert m.output_size == 1
            

        # if isinstance(m, Attention):
        #     attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, is_prompt=m.is_prompt, bias=True, base_size=base_size, fast_imp=False,
        #                           train_size=train_size)
        #     setattr(model, n, attn)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size).cuda()
        with torch.no_grad():
            self.forward(imgs)

class ECFNetLocal(Local_Base, ECFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ECFNet.__init__(self, *args, **kwargs)
        self.cuda()

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
def build_net():
    return ECFNet()

if __name__ == "__main__":
    import time
    # start = time.time()
    net = ECFNet()
    x = torch.randn((2, 3, 256, 256))
    print("Total number of param  is ", sum(i.numel() for i in net.parameters()))
    t=net(x)
    print(t[0].shape)
    inp_shape = (3, 256, 256)
    from ptflops import get_model_complexity_info
    FLOPS = 0

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # # print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9



    print('mac', macs, params)
