import os

from graphviz import Digraph
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_faithfulness(
        outs,
        metric_fn_dict,
        save_path,
    ):
    """
    plot faithfulness results

    Plot all w.r.t. thresholds, and plot line for complete and empty graphs

    TODO : Plot edges w.r.t. nodes, and other plots if needed
    """

    thresholds = []
    n_nodes = []
    n_edges = []
    avg_deg = []
    density = []
    # modularity = []
    # z_score = []
    faithfulness = [[] for _ in metric_fn_dict]
    complete = []
    empty = []

    for t in outs:
        if t == 'complete':
            for i, fn_name in enumerate(metric_fn_dict):
                complete.append(outs[t][fn_name])
            continue
        if t == 'empty':
            for i, fn_name in enumerate(metric_fn_dict):
                empty.append(outs[t][fn_name])
            continue
        thresholds.append(t)
        n_nodes.append(outs[t]['n_nodes'])
        n_edges.append(outs[t]['n_edges'])
        avg_deg.append(outs[t]['avg_deg'])
        density.append(outs[t]['density'])
        # modularity.append(outs[t]['modularity'])
        # z_score.append(outs[t]['z_score'])
        for i, fn_name in enumerate(metric_fn_dict):
            faithfulness[i].append(outs[t]['faithfulness'][fn_name])

    fig = make_subplots(
        rows=4 + len(list(metric_fn_dict.keys())),
        cols=1,
    )

    for i, fn_name in enumerate(metric_fn_dict):
        print(faithfulness[i])
        fig.add_trace(go.Scatter(
                x=thresholds,
                y=faithfulness[i],
                mode='lines+markers',
                #title_text=fn_name+" faithfulness vs threshold",
                name=fn_name,
            ),
            row=i+1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=n_nodes,
            mode='lines+markers',
            #title_text="n_nodes vs threshold",
            name='n_nodes',
        ),
        row=len(list(metric_fn_dict.keys()))+1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=n_edges,
            mode='lines+markers',
            #title_text="n_edges vs threshold",
            name='n_edges',
        ),
        row=len(list(metric_fn_dict.keys()))+2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=avg_deg,
            mode='lines+markers',
            #title_text="avg_deg vs threshold",
            name='avg_deg',
        ),
        row=len(list(metric_fn_dict.keys()))+3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=density,
            mode='lines+markers',
            #title_text="density vs threshold",
            name='density',
        ),
        row=len(list(metric_fn_dict.keys()))+4, col=1
    )

    # Update x-axes to log scale
    fig.update_xaxes(type="log")

    # default layout is : height=600, width=800. We want to make it a bit bigger so that each plot has the original size
    fig.update_layout(
        height=600 + 400 * (4 + len(list(metric_fn_dict.keys()))),
        width=800,
        title_text="Faithfulness and graph properties w.r.t. threshold",
        showlegend=True,
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + "faithfulness.html")

# Functions defined by Marks et al. Here only for compatibility.

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