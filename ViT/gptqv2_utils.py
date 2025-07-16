import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import timm
import logging
import functools

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQv2:

    def __init__(self, layer, cls_token=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []
        self.cls_token = cls_token

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp[:, self.cls_token:, :]
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float().to(self.dev) * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())

        del self.fp_inp[0]

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False,ours = False, beta = 0.0003
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        fp_weight = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        if ours:
            fp_weight = W.clone()
            print(fp_weight.shape)
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            if ours:
                fp_weight = fp_weight.to(W.device)[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        alpha = 0.25
        P = alpha * ((self.dXXT @ Hinv.T).triu(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            if ours:
                fp_weight1 = fp_weight[:, i1:i2]
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                if ours:
                    W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) + w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0)) - (W1[:, i:]-fp_weight1[:, i:]) @ (Hinv1[i:,i:].t()@Hinv1[i:,i:]) * beta
                else:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            if ours:
                W[:, i2:] = W[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:]) + W1.matmul(P[i1:i2, i2:]) - (W[:, i2:] - fp_weight[:, i2:]) @ (Hinv[i2:, i2:].t()@Hinv[i2:, i2:]) * beta
            else:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])
            if ours:
                del fp_weight1
        if ours:
            del fp_weight
        torch.cuda.synchronize()
        

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


class FPInputsCache:
    """
    class for saving the full-precision output in each layer.
    """
    def __init__(self, sequential, cls_token=0):
        self.fp_cache = {}
        self.names = sequential[0]+sequential[1]+sequential[2]+sequential[3]
        print(self.names)
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []
        self.cls_token = cls_token

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach().cpu()
        if len(inp.shape) == 3:
            inp = inp[:, self.cls_token:, :]
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name] += [inp.t()]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


@torch.no_grad()
def gptqv2_fwrd(model, calib_data, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    print('-----GPTQv2 Quantization-----')

    layers = model.blocks

    model = model.cuda()
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.fwd_kwargs = {}
            self.inps = None

        def forward(self, inp, **kwargs):
            self.inps = inp.data.clone()
            self.fwd_kwargs = kwargs
            print('data collected')
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        model(calib_data.to(dev))
    except ValueError:
        pass

    inps = layers[0].inps
    fwd_kwargs = layers[0].fwd_kwargs

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model = model.cpu()

    del calib_data
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    quantizers = {}
    if isinstance(model, timm.models.Eva):
        sequential = [
            ['attn.q_proj.module', 'attn.k_proj.module', 'attn.v_proj.module'],
            ['attn.proj.module'],
            ['mlp.fc1_g.module', 'mlp.fc1_x.module'],
            ['mlp.fc2.module']
        ]
    elif isinstance(model, timm.models.VisionTransformer):
        sequential = [
            ['attn.qkv.module'],
            ['attn.proj.module'],
            ['mlp.fc1.module'],
            ['mlp.fc2.module']
        ]
    else:
        raise NotImplementedError

    fp_inputs_cache = FPInputsCache(sequential, cls_token=0)
    fp_inps = inps.clone()

    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)
        fp_inps = layer(fp_inps, **fwd_kwargs)
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'head' in name:
                    layer_weight_bits = 16
                    continue
                gptq[name] = GPTQv2(subset[name], cls_token=0)
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            outs = layer(inps, **fwd_kwargs)  # forward calibration data
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False,ours = args.ours, beta = args.beta
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        outs = layer(inps, **fwd_kwargs)  # forward calibration data

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    utils.cleanup_memory(verbos=True)
    print('-----GPTQv2 Quantization Done-----\n')
    return quantizers
