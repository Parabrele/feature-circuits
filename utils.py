from __future__ import annotations
from graphviz import Digraph
from collections import defaultdict
import os
import json
import random
import torch as t
import torch.nn.functional as F
from dataclasses import dataclass
from torchtyping import TensorType

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

##########
# SparseAct
##########

class SparseAct():
    """
    A SparseAct is a helper class which represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term.
    A SparseAct may have three fields:
    act : the feature activations in the sparse basis
    res : the SAE error term
    resc : a contracted SAE error term, useful for when we want one number per feature and error (instead of having d_model numbers per error)
    """

    def __init__(
            self, 
            act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None, 
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            resc: TensorType["batch_size", "n_ctx"] = None, # contracted residual
            ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'SparseAct':
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseAct(**kwargs)
        
    def __mul__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            # Handle SparseAct * SparseAct
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return SparseAct(**kwargs)

    def __rmul__(self, other) -> 'SparseAct':
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)
    
    def __matmul__(self, other: SparseAct) -> SparseAct:
        # dot product between two SparseActs, except only the residual is contracted
        return SparseAct(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))
    
    def __add__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) + getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return SparseAct(**kwargs)
    
    def __radd__(self, other: SparseAct) -> SparseAct:
        return self.__add__(other)
    
    def __sub__(self, other: SparseAct) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return SparseAct(**kwargs)
    
    def __truediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseAct(**kwargs)

    def __rtruediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseAct(**kwargs)

    def __neg__(self) -> SparseAct:
        sparse_result = -self.act
        res_result = -self.res
        return SparseAct(act=sparse_result, res=res_result)
    
    def __invert__(self) -> SparseAct:
            return self._map(lambda x, _: ~x)


    def __gt__(self, other) -> SparseAct:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) > other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")
    
    def __lt__(self, other) -> SparseAct:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) < other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")
    
    def __getitem__(self, index: int):
        return self.act[index]
    
    def __repr__(self):
        if self.res is None:
            return f"SparseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")
    
    def amax(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).amax(dim)
        return SparseAct(**kwargs)

    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseAct(**kwargs)
    
    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseAct(**kwargs)
    
    def nonzero(self):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return SparseAct(**kwargs)
    
    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return SparseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseAct(**kwargs)
    
    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseAct(**kwargs)
    
    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return SparseAct(**kwargs)

    def save(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self
    
    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseAct(act=self.act, res=self.res)

    @staticmethod
    def zeros_like(other):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(other, attr) is not None:
                kwargs[attr] = t.zeros_like(getattr(other, attr))
        return SparseAct(**kwargs)

    
    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            # act shape : (batch_size, n_ctx, d_dictionary)
            # resc shape : (batch_size, n_ctx)
            # cat needs the same number of dimensions, so use unsqueeze to make the resc shape (batch_size, n_ctx, 1)
            try:
                if self.resc.dim() == self.act.dim() - 1:
                    return t.cat([self.act, self.resc.unsqueeze(-1)], dim=-1)
            except:
                pass
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)
    
    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)
    
    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())
    
    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))
    
    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)
    
    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))
    
    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))
    
    def abs(self):
        return self._map(lambda x, _: x.abs())

##########
# Circuit plotting
##########

def get_name(component, layer, idx):
    match idx:
        case (seq, feat):
            if feat == 32768: feat = 'ε'
            if layer == -1: return f'{seq}, embed/{feat}'
            return f'{seq}, {component}_{layer}/{feat}'
        case (feat,):
            if feat == 32768: feat = 'ε'
            if layer == -1: return f'embed/{feat}'
            return f'{component}_{layer}/{feat}'
        case _: raise ValueError(f"Invalid idx: {idx}")

def plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, pen_thickness=1, annotations=None, save_dir='circuit'):
    """
    TODO :
    edges _ - n have weights w_i, color them by w_i / sum(w_i)

    """
    
    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))

    # for deciding shade of node
    def to_hex(number):
        number = number / scale
        
        # Define how the intensity changes based on the number
        # - Negative numbers increase red component to max
        # - Positive numbers increase blue component to max
        # - 0 results in white
        if number < 0:
            # Increase towards red, full intensity at -1.0
            red = 255
            green = blue = int((1 + number) * 255)  # Increase other components less as it gets more negative
        elif number > 0:
            # Increase towards blue, full intensity at 1.0
            blue = 255
            red = green = int((1 - number) * 255)  # Increase other components less as it gets more positive
        else:
            # Exact 0, resulting in white
            red = green = blue = 255 
        
        # decide whether text is black or white depending on darkness of color
        text_hex = "#000000" if (red*0.299 + green*0.587 + blue*0.114) > 170 else "#ffffff"

        # Convert to hex, ensuring each component is 2 digits
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        
        return hex_code, text_hex
    
    if annotations is None:
        def get_label(name):
            return name
    else:
        def get_label(name):
            match name.split(', '):
                case seq, feat:
                    if feat in annotations:
                        component = feat.split('/')[0]
                        component = feat.split('_')[0]
                        return f'{seq}, {annotations[feat]} ({component})'
                    return name
                case [feat]:
                    if feat in annotations:
                        component = feat.split('/')[0]
                        component = feat.split('_')[0]
                        return f'{annotations[feat]} ({component})'

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    # rename embed to resid_-1
    nodes_by_submod = {
        'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
    }
    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }
    edges['resid_-1'] = edges['embed']
    
    for layer in range(-1, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1 and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    fillhex, texthex = to_hex(effect)
                    if name[-1:].endswith('ε'):
                        subgraph.node(name, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
        
        # add edges to previous layer resid
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
                for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                    weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name('resid', layer-1, upstream_idx)
                        dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=True)

def plot_circuit_posaligned(nodes, edges, layers=6, length=6, example_text="The managers that the parent likes",
                            node_threshold=0.1, edge_threshold=0.01, pen_thickness=3, annotations=None, save_dir='circuit'):

    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))

    words = example_text.split()

    # for deciding shade of node
    def to_hex(number):
        number = number / scale
        
        # Define how the intensity changes based on the number
        # - Negative numbers increase red component to max
        # - Positive numbers increase blue component to max
        # - 0 results in white
        if number < 0:
            # Increase towards red, full intensity at -1.0
            red = 255
            green = blue = int((1 + number) * 255)  # Increase other components less as it gets more negative
        elif number > 0:
            # Increase towards blue, full intensity at 1.0
            blue = 255
            red = green = int((1 - number) * 255)  # Increase other components less as it gets more positive
        else:
            # Exact 0, resulting in white
            red = green = blue = 255 
        
        # decide whether text is black or white depending on darkness of color
        text_hex = "#000000" if (red*0.299 + green*0.587 + blue*0.114) > 170 else "#ffffff"

        # Convert to hex, ensuring each component is 2 digits
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        
        return hex_code, text_hex
    
    if annotations is None:
        def get_label(name):
            return name
    else:
        def get_label(name):
            seq, feat = name.split(", ")
            if feat in annotations:
                component = feat.split('/')[0]
                component = component.split('_')[0]
                return f'{seq}, {annotations[feat]} ({component})'
            return name

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    nodes_by_submod = {
        'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
    }
    nodes_by_seqpos = defaultdict(list)
    nodes_by_layer = defaultdict(list)
    edgeset = set()

    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }
    edges['resid_-1'] = edges['embed']

    # add words to bottom of graph
    with G.subgraph(name=f'words') as subgraph:
        subgraph.attr(rank='same')
        prev_word = None
        for idx in range(length):
            word = words[idx]
            subgraph.node(word, shape='none', group=str(idx), fillcolor='transparent',
                          fontsize="30pt")
            if prev_word is not None:
                subgraph.edge(prev_word, word, style='invis', minlen="2")
            prev_word = word

    for layer in range(-1, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1 and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    seq_pos, basename = name.split(", ")
                    fillhex, texthex = to_hex(effect)
                    if name[-1:] == 'ε':
                        subgraph.node(name, shape='triangle', group=seq_pos, width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), group=seq_pos, fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    
                    if len(nodes_by_seqpos[seq_pos]) == 0:
                        G.edge(words[int(seq_pos)], name, style='dotted', arrowhead='none', penwidth="1.5")
                        edgeset.add((words[int(seq_pos)], name))

                    nodes_by_seqpos[seq_pos].append(name)
                    nodes_by_layer[layer].append(name)

                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))
        
        # add edges to previous layer resid
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
                for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                    weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name('resid', layer-1, upstream_idx)
                        dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )
            edgeset.add((uname, dname))

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=True)

##########
# Wtf is this?
##########

@dataclass
class DictionaryCfg():
    def __init__(
        self,
        dictionary_dir,
        dictionary_size
        ) -> None:
        self.dir = dictionary_dir
        self.size = dictionary_size

def load_examples(dataset, num_examples, model, seed=12, pad_to_length=None, length=None):
    examples = []
    dataset_items = open(dataset).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(data["clean_prefix"], return_tensors="pt",
                                        padding=False).input_ids
        patch_prefix = model.tokenizer(data["patch_prefix"], return_tensors="pt",
                                        padding=False).input_ids
        clean_answer = model.tokenizer(data["clean_answer"], return_tensors="pt",
                                        padding=False).input_ids
        patch_answer = model.tokenizer(data["patch_answer"], return_tensors="pt",
                                        padding=False).input_ids
        # only keep examples where answers are single tokens
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue
        # only keep examples where clean and patch inputs are the same length
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            patch_prefix = t.flip(F.pad(t.flip(patch_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
        
        example_dict = {"clean_prefix": clean_prefix,
                        "patch_prefix": patch_prefix,
                        "clean_answer": clean_answer.item(),
                        "patch_answer": patch_answer.item(),
                        "annotations": get_annotation(dataset, model, data),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def load_examples_nopair(dataset, num_examples, model, length=None):
    examples = []
    if isinstance(dataset, str):        # is a path to a .json file
        dataset = json.load(open(dataset))
    elif isinstance(dataset, dict):     # is an already-loaded dictionary
        pass
    else:
        raise ValueError(f"`dataset` is unrecognized type: {type(dataset)}. Must be path (str) or dict")
    
    max_len = 0     # for padding
    for context_id in dataset:
        context = dataset[context_id]["context"]
        if length is not None and len(context) > length:
            context = context[-length:]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                        padding=False).input_ids
        max_len = max(max_len, clean_prefix.shape[-1])

    for context_id in dataset:
        answer = dataset[context_id]["answer"]
        context = dataset[context_id]["context"]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                                    padding=False).input_ids
        clean_answer = model.tokenizer(answer, return_tensors="pt",
                                    padding=False).input_ids
        if clean_answer.shape[1] != 1:
            continue
        prefix_length_wo_pad = clean_prefix.shape[1]
        pad_length = max_len - prefix_length_wo_pad
        # left padding: reverse, right-pad, reverse
        clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))

        example_dict = {"clean_prefix": clean_prefix,
                        "clean_answer": clean_answer.item(),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def get_annotation(dataset, model, data):
    # First, understand which dataset we're working with
    structure = None
    if "within_rc" in dataset:
        structure = "within_rc"
        template = "the_subj subj_main that the_dist subj_dist"
    elif "rc.json" in dataset or "rc_" in dataset:
        structure = "rc"
        template = "the_subj subj_main that the_dist subj_dist verb_dist"
    elif "simple.json" in dataset or "simple_" in dataset:
        structure = "simple"
        template = "the_subj subj_main"
    elif "nounpp.json" in dataset or "nounpp_" in dataset:
        structure = "nounpp"
        template = "the_subj subj_main prep the_dist subj_dist"

    if structure is None:
        return {}
    
    annotations = {}

    # Iterate through words in the template and input. Get token spans
    curr_token = 0
    for template_word, word in zip(template.split(), data["clean_prefix"].split()):
        if word != "The":
            word = " " + word
        word_tok = model.tokenizer(word, return_tensors="pt", padding=False).input_ids
        num_tokens = word_tok.shape[1]
        span = (curr_token, curr_token + num_tokens-1)
        curr_token += num_tokens
        annotations[template_word] = span
    
    return annotations

##########
# Circuit Utils
##########

def get_hidden_states(
    model,
    submods,
    dictionaries,
    is_tuple,
    input,
):
    hidden_states = {}
    with model.trace(input, **tracer_kwargs), t.no_grad():
        for submodule in submods:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            x_hat, upstream_act = dictionary(x, output_features=True)
            hidden_states[submodule] = SparseAct(act=upstream_act.save(), res=(x - x_hat).save())
    hidden_states = {k : v.value for k, v in hidden_states.items()}
    return hidden_states

def get_min_value(dtype):
    if dtype.is_floating_point:
        return t.finfo(dtype).min
    elif dtype.is_signed:
        return t.iinfo(dtype).min
    else:
        return 0

@t.jit.script
def compile_for_loop(inverse, new_values, x_values):
    for i, j in enumerate(inverse):
        new_values[j] = t.max(new_values[j], x_values[i])
        
def sparse_coo_max(x, dim):
    """
    x : a sparse COO tensor
    dim : a dimension to reduce
    return a sparse COO tensor with the maximum value along the specified dimension
    """
    # coalesce x
    x = x.coalesce()
    x_shape = x.shape
    new_shape = x_shape[:dim] + x_shape[dim+1:]

    idxs = x.indices() # [len(x_shape), n]
    new_idxs = t.cat((idxs[:dim], idxs[dim+1:]), dim=0) # [len(x_shape)-1, n]
    new_idxs, inverse = new_idxs.unique(dim=-1, return_inverse=True)
    new_values = t.full((new_idxs.shape[1],), get_min_value(x.dtype), dtype=x.dtype, device=x.device)

    # for each new index, get the max value along original indexes that were merged to it. Use return_inverse
    compile_for_loop(inverse, new_values, x.values())

    return t.sparse_coo_tensor(new_idxs, new_values, new_shape).coalesce()

def sparse_coo_amax(x, dim):
    """
    x : a sparse COO tensor
    dim : a dimension to reduce/a tuple of dimensions to reduce
    return a sparse COO tensor with the maximum value along the specified dimension(s)
    """
    if isinstance(dim, int):
        return sparse_coo_max(x, dim)
    else:
        dim = sorted(list(dim), reverse=True)
        for d in dim:
            x = sparse_coo_max(x, d)
        return x

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def rearrange_weights(nodes, edges):
    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix

def aggregate_weights(
    nodes, edges, aggregation='max'
):
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'max':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_coo_amax(weight_matrix, dim=1)
                else:
                    weight_matrix = sparse_coo_amax(weight_matrix, dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].amax(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_coo_amax(weight_matrix, dim=0)
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_coo_amax(weight_matrix, dim=(0,2))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].amax(dim=0)


    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

##########
# save and load circuit utils
##########

def save_circuit(save_dir, nodes, edges, dataset_name, model_name, node_threshold, edge_threshold, num_examples):
    save_dict = {
        "nodes" : dict(nodes),
        "edges" : dict(edges)
    }
    save_basename = f"{dataset_name}_{model_name}_node{node_threshold}_edge{edge_threshold}_n{num_examples}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

def load_circuit(save_dir, dataset_name, model_name, node_threshold, edge_threshold, num_examples):
    path = f'{save_dir}{dataset_name}_{model_name}_node{node_threshold}_edge{edge_threshold}_n{num_examples}.pt'
    return load_from(path)

def load_from(circuit_path):
    with open(circuit_path, 'rb') as infile:
        save_dict = t.load(infile)
    nodes = save_dict['nodes']
    edges = save_dict['edges']
    return nodes, edges