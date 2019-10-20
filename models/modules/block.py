from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

from . import spectral_norm as SN
import logging
logger = logging.getLogger('blocks')
####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class MergeBlock(nn.Module):
    def __init__(self, stages, merger):
        
        super(MergeBlock, self).__init__()
        self.stages = stages
        
        self.merger = merger
        for idx, stage in enumerate(self.stages):
            # registering submodules
            self.add_module("stage_{:02d}".format(idx+1),stage)

    def forward(self, x):
        OUS = [x,]
        t = x
        for stage in self.stages:
            t = stage(t)
            OUS.append(t)
        output = self.merger(torch.cat(OUS, 1))
        return output
    
    
    def __repr__(self):
        modstr = ['\n|']
        for idx, stage in enumerate(self.stages):
            modstr.append(stage.__repr__().replace('\n', '\n'+'|'*(idx+1)))
            modstr.append('\n'+'|'*(idx+1))
        modstr[-1] = '\n'
        modstr.append("|"+"-"*(idx)+'> '+self.merger.__repr__())
        return ''.join(modstr)

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        try:
            _ = torch.einsum('bnchwk,bnchwk -> bnchw', torch.randn(1,3,5,7,9,11).cuda(), torch.randn(1,3,5,7,9,11).cuda())
        except:pass

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        # logging.info("x %s", x.shape)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # logging.info("padded_x %s", padded_x.shape)
        # logging.info("q_out %s", q_out.shape)
        padding_size = self.kernel_size//2

        k_out = F.pad(k_out, [padding_size,padding_size,padding_size,padding_size])
        v_out = F.pad(v_out, [padding_size,padding_size,padding_size,padding_size])


        # logging.info("k_out %s", k_out.shape)
        # logging.info("v_out %s", v_out.shape)

        # k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # logging.info("k_out type %s", type(k_out))
        # logging.info("k_out type %s", repr(k_out.unfold))

        k_out = k_out.unfold(2, self.kernel_size, step=self.stride).unfold(3, self.kernel_size, step=self.stride)
        v_out = v_out.unfold(2, self.kernel_size, step=self.stride).unfold(3, self.kernel_size, step=self.stride)

        # logging.info("k_out %s", k_out.shape)
        # logging.info("v_out %s", v_out.shape)

        v_out_h, v_out_w = v_out.split(self.out_channels // 2, dim=1)
        v_out = torch.cat((v_out_h + self.rel_h, v_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)

        # logging.info("k_out %s", k_out.shape)
        # logging.info("v_out %s", v_out.shape)
        # logging.info("q_out %s", q_out.shape)
        # logging.info("out %s", out.shape)

        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        logger.info("AttentionConv init called")
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)



####################
# Useful blocks
####################

class second_order_residual_block(nn.Module):
    def __init__(self, 
        in_nc,
        out_nc,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type='zero',
        norm_type=None,
        act_type='relu',
        mode='CNA',
    ):
        super(second_order_residual_block, self).__init__()
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None

        self.first_order_residual = conv_block(in_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        self.second_order_residual = AttentionConv(in_nc, out_nc, kernel_size, stride=1, padding=0, groups=1, bias=False)
        self.merger = nn.Conv2d(3*out_nc, out_nc, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        order_residuals = [x, self.first_order_residual(x), self.second_order_residual(x)]  
        t = torch.cat(order_residuals, dim=1)
        output = self.merger(t)
        return output

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


def conv1d_spectral_norm(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    # borrowed from fastai
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return SN.spectral_norm(conv)

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer"
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return nn.Sequential(conv)


class SelfAttention(nn.Module):
    # borrowed from fastai
    "Self attention layer for nd."
    def __init__(self, n_channels:int,normalise:bool=True):
        super().__init__()
        if normalise:
            _conv1d = conv1d_spectral_norm
        else:
            _conv1d = conv1d
        self.query = _conv1d(n_channels, n_channels//8)
        self.key   = _conv1d(n_channels, n_channels//8)
        self.value = _conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        try:
            # to remove torch.bmm erro happening in cublas !!!
            _ = torch.bmm(torch.randn(12,6,6,).cuda(),torch.randn(12,6,6).cuda())
        except:
          pass

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        f_T = f.permute(0,2,1).contiguous()
        f_t_g = torch.bmm(f_T, g)
        beta = nn.functional.softmax(f_t_g, dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class DenseBlock_5C(nn.Module):

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA',self_attention:bool=False, self_attention_normalise:bool=True):
        super(DenseBlock_5C, self).__init__()
        
        
        def core_conv_block():
             return conv_block(nc, nc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        
        conv1 = core_conv_block()
        conv2 = core_conv_block()
        conv3 = core_conv_block()
        conv4 = core_conv_block()
        
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
            
        #self.conv5 = conv_block(nc+4*nc, nc, 1, stride, bias=False, pad_type=pad_type, norm_type=norm_type, act_type=None, mode=mode)
        
        merge_layer = nn.Conv2d(nc+4*nc, nc, kernel_size=1, stride=1, bias=False)

        F = MergeBlock(
            stages=[conv1,conv2,conv3,conv4], 
            merger=merge_layer
        )

        if self_attention:
            logger.info(f"DenseBlock_5C SelfAttention True")
            self.f = nn.Sequential(F, SelfAttention(nc, self_attention_normalise))
        else: self.f = F
 
    def forward(self, x):
        return self.f(x)

    
    
class DDDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA', self_attention:bool=False, self_attention_normalise:bool=True):
        super(DDDB, self).__init__()
        
        DDB1 = DenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        DDB2 = DenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        DDB3 = DenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        
        merge_layer = nn.Conv2d(4*nc, nc, kernel_size=1, stride=1, bias=False)

        F = MergeBlock(
            stages=[DDB1,DDB2,DDB3], 
            merger=merge_layer
        )

        if self_attention:
            logger.info(f"DDDB SelfAttention True")
            self.f = nn.Sequential(F, SelfAttention(nc, self_attention_normalise))
        else: self.f = F
  
    def forward(self, x):
        return self.f(x)



####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)







# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True).to(device)
        else:
            model = torchvision.models.vgg19(pretrained=True).to(device)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output



## ################ ##
## NIMA loss blocks ##
## ################ ##

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
def mobilenet_conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def mobilenet_conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [mobilenet_conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(mobilenet_conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        
        self.normalise_image = MeanShift(
            rgb_range=1.0,
            rgb_mean=(0.485, 0.456, 0.406), 
            rgb_std=(0.229, 0.224, 0.225)
        )
        
        base_model = MobileNetV2()
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear') 
        x = self.normalise_image(x)
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
