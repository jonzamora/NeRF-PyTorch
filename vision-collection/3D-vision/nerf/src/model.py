'''
NeRF Model Definition and Helper Functions
'''
from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeRF(nn.Module):
    def __init__(self, num_layers=8, num_units=128, in_channels=3, in_views=3, out_channels=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.in_channels = in_channels
        self.in_views = in_views
        self.out_channels = out_channels
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.in_channels, self.num_units)] + [nn.Linear(self.num_units, self.num_units)
            if layer not in self.skips else nn.Linear(self.num_units + self.in_channels, self.num_units) for layer in range(self.num_layers-1)]
        )

        self.linear_views = nn.ModuleList([nn.Linear(self.in_channels + self.num_units, self.num_units // 2)])

        if self.use_viewdirs:
            self.linear_feature = nn.Linear(self.num_units, self.num_units)
            self.linear_alpha = nn.Linear(self.num_units, 1)
            self.linear_rgb = nn.Linear(self.num_units // 2, 3)
        else:
            self.linear_output = nn.Linear(self.num_units, self.out_channels)
    
    def forward(self, x):
        input_points, input_views = torch.split(x, [self.in_channels, self.in_views], dim=-1)
        h = input_points

        for idx, layer in enumerate(self.linear_layers):
            h = self.linear_layers[idx](h)
            h = F.relu(h)

            if idx in self.skips:
                h = torch.cat([input_points, h], -1)
        
        if self.use_viewdirs:
            alpha = self.linear_alpha(h)
            feature = self.linear_feature(h)
            h = torch.cat([feature, input_views], -1)

            for idx, layer in enumerate(self.linear_views):
                h = self.linear_views[idx](h)
                h = F.relu(h)

            rgb = self.linear_rgb(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.linear_output(h)
        
        return outputs


class PositionalEncoder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.encoder_function()
    
    def encoder_function(self):
        encoder_functions = []
        input_dims = self.kwargs["input_dims"]
        out_dims = 0

        if self.kwargs["include_input"]:
            encoder_functions.append(lambda p : p)
            out_dims += input_dims
        
        max_freq_log2 = self.kwargs["max_freq_log2"]
        num_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            frequency_bands = 2.0 ** torch.linspace(start=0.0, end=max_freq_log2, steps=num_freqs)
        else:
            frequency_bands = torch.linspace(start=1.0, end=2.0 ** max_freq_log2, steps=num_freqs)
        
        for frequency in frequency_bands:
            for fn in self.kwargs["periodic_functions"]:
                encoder_functions.append(lambda p, fn=fn, f=frequency : fn(p * f))
                out_dims += input_dims
        
        self.encoder_functions = encoder_functions
        self.out_dims = out_dims
    
    def encode(self, x):
        return torch.cat([fn(x) for fn in self.encoder_functions], -1)


def positional_encoding_embeddings(multires):
    
    encoder_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_functions": [torch.sin, torch.cos]
    }

    position_encoder = PositionalEncoder(**encoder_kwargs)
    encode = lambda input, encoder=position_encoder : encoder.encode(input)

    return encode, position_encoder.out_dims

def batchify(function, chunk):
    if chunk is None:
        return function

    def ret(inputs):
        return torch.cat([function(inputs[idx:idx+chunk]) for idx in range(0, inputs.shape[0], chunk)], 0)
    
    return ret
        

def run_nerf(inputs, view_directions, function, encode_function, encode_directions_function, chunk=1024*64):
    flat_inputs = torch.reshape(inputs, [-1, inputs.shape[-1]])
    encoded = encode_function(flat_inputs)

    if view_directions:
        input_directions = view_directions[:, None].expand(inputs.shape)
        flat_input_directions = torch.reshape(input_directions, [-1, input_directions.shape[-1]])
        encoded_directions = encode_directions_function(flat_input_directions)
        encoded = torch.cat([encoded, encoded_directions], -1)
    
    flat_outputs = batchify(function, chunk)(encoded)
    outputs = torch.reshape(flat_outputs, list(inputs.shape[:-1]) + [flat_outputs.shape[-1]])

    return outputs


def build_nerf(view_directions=True, fine_samples_per_ray=128):
    encoding_fn, in_channels = positional_encoding_embeddings(multires=10)
    in_channel_views = 0
    encoding_fn_view_directions = None

    if view_directions:
        encoding_fn_view_directions, in_channel_views = positional_encoding_embeddings(multires=4)
    
    out_channels = 5 if fine_samples_per_ray > 0 else 4

    coarse_nerf = NeRF(out_channels=out_channels, in_views=in_channel_views).to("cuda")
    grad_params = list(coarse_nerf.parameters())

    fine_nerf = None
    
    if fine_samples_per_ray > 0:
        fine_nerf = NeRF(out_channels=out_channels, in_views=in_channel_views).to("cuda")
        grad_params += list(fine_nerf.parameters())
    
    model_query_fn = lambda inputs, view_directions, model_fn : run_nerf(inputs, view_directions, model_fn, encoding_fn,
                                                                         encoding_fn_view_directions, chunk=1024*64)
    
    optimizer = torch.optim.Adam(params=grad_params, lr=5e-4, betas=(0.9, 0.999))

    start = 0

    render_kwargs_train = {
        "model_query_fn": model_query_fn,
        "perturb_jitter": 1.0,
        "fine_samples_per_ray": fine_samples_per_ray,
        "fine_nerf": fine_nerf,
        "coarse_samples_per_ray": 64,
        "coarse_nerf": coarse_nerf,
        "use_viewdirs": view_directions,
        "white_background": True,
        "raw_noise_std": 0.0
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb_jitter"] = 0.0
    
    return render_kwargs_train, render_kwargs_test, start, grad_params, optimizer