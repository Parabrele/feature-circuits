# import MNIST data and torch
print("Importing libraries...")
import graph_tool.all as gt
print("Imported graph_tool")

import os
import pickle

import numpy as np

import torch
import torch.nn as nn
print("Imported torch")

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from welford_torch import OnlineCovariance
print("Done importing libraries")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time
import graph_tool.all as gt
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# Hyper-parameters
USE_ATTR = True

num_classes = 10

epochs = 10
lr = 1e-3
batch_size = 100

# MNIST dataset
print("Loading MNIST dataset...")

root = '/scratch/pyllm/dhimoila/MNIST/'
scratch = '/scratch/pyllm/dhimoila/MNIST_tests/'
if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(scratch):
    os.makedirs(scratch)

train_dataset = torchvision.datasets.MNIST(
    root=root,
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root=root,
    train=False,   
    transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)
print("Done.")

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), # 28x28x1 -> 24x24x32
            nn.ReLU(),
            nn.MaxPool2d(2) # 24x24x32 -> 12x12x32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), # 12x12x32 -> 10x10x64
            nn.ReLU(),
            nn.MaxPool2d(2) # 10x10x64 -> 5x5x64
        )
        self.classifier = nn.Linear(5*5*64, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class SAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        """One layer MLP Autoencoder trained with sparsity constraint"""
        super(SAE, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        self.encoder.bias = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(input_size))
        self.hidden_size = hidden_size
        self.input_size = input_size

    def encode(self, x):
        return torch.relu(self.encoder(x))
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x, output_features=False):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        if output_features:
            return encoded, decoded
        return decoded

class IdentityDict:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, x):
        return x
    
    def decode(self, x):
        return x
    
    def forward(self, x, output_features=False):
        if output_features:
            return x, x
        return x

def train_CNN(model, train_loader, test_loader, loss_fn, optimizer, device, epochs=10):
    lossess = []
    step = 0
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                lossess.append(loss.item())
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader) * epochs}], Loss: {loss.item()}")
            step += 1
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {correct / total}")
    return lossess

def train_SAE(model, sae, train_loader, test_loader, loss_fn, alpha, optimizer, device, epochs=10):
    model.eval()
    act_buffer = []
    hook = model.layer2.register_forward_hook(lambda module, input, output: act_buffer.append(output))
    lossess = []
    step = 0

    for epoch in range(epochs):
        sae.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            _ = model(images)
            act = act_buffer.pop() # batch_size x 64 x 5 x 5
            act = act.reshape(act.size(0), -1)
            f, reconstructed = sae(act, output_features=True)
            mse = loss_fn(reconstructed, act)
            a = 1.
            L1 = ((f.abs()**a).sum(dim=-1)**(1/a)).mean()
            loss = mse + alpha * L1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                lossess.append(loss.item())
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader) * epochs}], Loss: {loss.item()}\n\tMSE: {mse.item()}, Variance explained: {1 - mse.item() / act.var().item()}\n\tL1: {L1.item()}, L0: {(f > 1e-6).sum(dim=-1).float().mean().item()}")

            step += 1

    # remove hook
    hook.remove()

##########
# train CNN
##########

def get_cnn():
    if not os.path.exists(scratch + 'mnist_cnn.pth'):
        cnn_model = MNIST_CNN().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
        losses = train_CNN(cnn_model, train_loader, test_loader, loss_fn, optimizer, device, epochs=10)

        # save CNN model
        torch.save(cnn_model.state_dict(), scratch + 'mnist_cnn.pth')
    else:
        cnn_model = MNIST_CNN().to(device)
        cnn_model.load_state_dict(torch.load(scratch + 'mnist_cnn.pth'))
    
    return cnn_model

##########
# train SAE
##########

n_neurons = 1600
exp_factor = 10
"""
# if not os.path.exists(scratch + 'mnist_sae.pth'):
#     sae = SAE(n_neurons, int(exp_factor * n_neurons)).to(device)
#     alpha = 1e-2
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
#     losses = train_SAE(cnn_model, sae, train_loader, test_loader, loss_fn, alpha, optimizer, device, epochs=epochs)

#     # save SAE model
#     torch.save(sae.state_dict(), scratch + 'mnist_sae.pth')
# else:
#     sae = SAE(n_neurons, int(exp_factor * n_neurons)).to(device)
#     sae.load_state_dict(torch.load(scratch + 'mnist_sae.pth'))
"""

##########
# get the feature correlation matrix
##########

@torch.enable_grad()
def IG(cnn_model, f, labels, device, steps=10):
    ig = 0
    for i in range(steps):
        alpha = i / (steps - 1)
        x = (alpha * f).detach().requires_grad_()
        x.retain_grad()
        logits = cnn_model.classifier(x)
        target = logits[torch.arange(logits.size(0), device=device), labels]
        target.sum().backward()
        ig += x.grad
    ig *= x / steps
    return ig

def get_covariance(cnn_model, train_loader, device, n_neurons):
    cov = OnlineCovariance()
    cnn_model.eval()
    # sae.eval()

    grad_or_no_grad = torch.no_grad if not USE_ATTR else torch.enable_grad
    with grad_or_no_grad():
        act_buffer = []
        hook = cnn_model.layer2.register_forward_hook(lambda module, input, output: act_buffer.append(output))
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            _ = cnn_model(images)
            act = act_buffer.pop() # batch_size x 64 x 5 x 5
            act = act.reshape(act.size(0), n_neurons)
            f = act#sae.encode(act)
            if not USE_ATTR:
                # use activations
                cov.add_all(f)
            else:
                # use IG attribution between 0 and f, metric : target logit
                ig = IG(cnn_model, f, labels, device)
                cov.add_all(ig)

        hook.remove()
    cov.cov.detach_()
    return cov

##########
# visualize the feature correlation matrix
##########

def plot_cov(cov):
    corr = cov.cov.cpu()
    eps = 1e-5
    diag = corr.diag().clone()
    dead_idx = diag < eps
    corr = corr[~dead_idx][:, ~dead_idx]
    diag = diag[~dead_idx]
    corr = corr / torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    diag_idx = torch.arange(corr.size(0))
    corr[diag_idx, diag_idx] = 0
    weights = 2 * torch.arctanh(corr)

    import matplotlib.pyplot as plt

    plt.imshow(corr.cpu().numpy())
    cbar = plt.colorbar()
    cbar.set_label("Correlation")
    plt.savefig(scratch + 'corr.png')
    plt.close()

    plt.imshow(weights.cpu().numpy())
    cbar = plt.colorbar()
    cbar.set_label("Edge weight")
    plt.savefig(scratch + 'weights.png')
    plt.close()

    return weights, dead_idx

##########
# spectral clustering : reordering the weights matrix based on the eigenvectors of the weights matrix
##########

def blocks_spectral_clustering(cov, weights, dead_idx):
    U, S, V = torch.svd(weights)
    k = 10
    Uk = U[:, :k]
    X = Uk / torch.sqrt(S[:k])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X.cpu().numpy())
    labels = kmeans.labels_
    idx = np.argsort(labels)
    plt.imshow(weights[idx][:, idx].cpu().numpy())

    labels_ = labels[idx]
    diff = np.diff(labels_)
    idx = np.where(diff)[0]
    # add vertical and horizontal lines at the cluster boundaries
    for i in idx:
        plt.axvline(i, color='black')
        plt.axhline(i, color='black')
    cbar = plt.colorbar()
    cbar.set_label("Edge weight")
    plt.savefig(scratch + 'weights_clustered.png')
    plt.close()

    # weights distribution
    plt.hist(weights.cpu().numpy().flatten(), bins=100)
    plt.savefig(scratch + 'weights_hist.png')
    plt.close()

    # label the original features idx, add a label for dead features :
    feature_labels = torch.zeros(cov.cov.size(0)).long().to(device)
    feature_labels[dead_idx] = -1
    feature_labels[~dead_idx] = torch.tensor(labels).long().to(device)

    return feature_labels

##########
# Do for SBM blocks
##########

def blocks_sbm(cov, weights, dead_idx, device):
    if not os.path.exists(scratch + 'state.pkl'):
        edge_list = torch.ones_like(weights).nonzero().cpu().numpy()
        weight_assignment = weights.numpy()[edge_list[:, 0], edge_list[:, 1]]

        G_gt = gt.Graph(directed=False)
        G_gt.add_edge_list(edge_list)
        G_gt.ep['weight'] = G_gt.new_edge_property("float", vals=weight_assignment)

        state_args = {
            'recs': [G_gt.ep['weight']],
            'rec_types': ['real-normal'],
        }

        print("Minimizing nested blockmodel...")
        start = time.time()
        state = gt.minimize_nested_blockmodel_dl(G_gt, state_args=state_args)
        end = time.time()
        print(f"Done in {end - start} seconds.")
        print("Before refinement :")
        print(state.print_summary())
        start = time.time()
        for i in tqdm(range(100)):
            state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        end = time.time()
        print(f"Done in {end - start} seconds.")
        print("After refinement :")
        print(state.print_summary())

        # save the state (not draw, save the object)
        with open(scratch + 'state.pkl', 'wb') as f:
            pickle.dump(state, f)
    else:
        with open(scratch + 'state.pkl', 'rb') as f:
            state = pickle.load(f)

    blockstate = state.project_level(1)#state.get_levels()[1]
    b = gt.contiguous_map(blockstate.get_blocks())
    blockstate = blockstate.copy(b=b)

    propert_map = blockstate.get_blocks()
    propert_map = torch.tensor(propert_map) # 1384 entries with B distinct values

    perm = torch.argsort(propert_map)
    weights = weights[perm][:, perm]

    plt.imshow(weights.cpu().numpy())
    cbar = plt.colorbar()
    cbar.set_label("Edge weight")
    plt.savefig(scratch + 'weights_sbm.png')
    plt.close()

    e = blockstate.get_matrix()
    B = blockstate.get_nonempty_B()
    contracted_edges = e.toarray()[:B, :B]
    print(state)
    print(state.print_summary())
    print(blockstate)
    print(B)
    print(contracted_edges.shape)
    plt.imshow(contracted_edges)
    cbar = plt.colorbar()
    cbar.set_label("Contracted edge weight")
    plt.savefig(scratch + 'contracted_edges.png')
    plt.close()

    n_blocks = B + 1
    n_samples = train_dataset.data.size(0)
    all_labels = list(set(propert_map.numpy())) + [B]

    feature_labels = torch.zeros(cov.cov.size(0)).long().to(device)
    feature_labels[dead_idx] = B
    feature_labels[~dead_idx] = torch.tensor(propert_map).long().to(device)

    return feature_labels, n_blocks, all_labels

##########
# Visualize actuvation across block & input
##########

@torch.no_grad()
def get_block_vs_sample(cnn_model, data_loader, device, feature_labels, all_labels, n_blocks):
    dataset = data_loader
    inputs = dataset.data.unsqueeze(1).float() / 255
    labels = dataset.targets

    n_samples = inputs.size(0)

    print(f"Number of blocks: {n_blocks}, Number of samples: {n_samples}")

    act_over_sample = torch.zeros(n_blocks, n_samples).to(device)
    act_over_class = torch.zeros(n_blocks, num_classes).to(device)

    act_buffer = []
    hook = cnn_model.layer2.register_forward_hook(lambda module, input, output: act_buffer.append(output))

    class_perm = torch.argsort(labels)
    images = inputs.to(device)[class_perm]
    labels = labels[class_perm]

    _ = cnn_model(images)
    act = act_buffer.pop() # batch_size x 64 x 5 x 5
    act = act.reshape(act.size(0), n_neurons) # batch_size x 1600
    f = act#sae.encode(act) # batch_size x 1600
    if USE_ATTR:
        f = IG(cnn_model, f, labels, device)
    for i, class_idx in enumerate(range(num_classes)):
        for label in all_labels:
            class_idxes = (labels == class_idx)
            feature_idx = (feature_labels == label)
            act_over_sample[label, class_idxes] = f[class_idxes][:, feature_idx].mean(dim=(1))
            act_over_class[label, class_idx] = f[class_idxes][:, feature_idx].mean()
        
    hook.remove()

    return act_over_sample, act_over_class

def run_block_vs_sample(cnn_model, data_loader, device, feature_labels, all_labels, n_blocks, suffix=''):
    act_over_sample, act_over_class = get_block_vs_sample(cnn_model, data_loader, device, feature_labels, all_labels, n_blocks)
    
    # rearrange matrices such that blocks are ordered by most active class :
    # act_over_class : n_blocks x num_classes
    most_active_class = act_over_class.argmax(dim=1)

    perm = torch.argsort(most_active_class)
    act_over_sample_ = act_over_sample[perm]
    act_over_class_ = act_over_class[perm]

    # plot the feature activation over samples
    fig, ax = plt.subplots()
    im = ax.imshow(act_over_sample_.cpu().numpy(), aspect='auto')
    ax.set_xlabel("Samples")
    ax.set_ylabel("Blocks")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean activation" if not USE_ATTR else "Mean attribution")
    plt.savefig(scratch + 'block_vs_sample' + suffix + '.png')
    plt.close()

    # plot the feature activation over classes
    fig, ax = plt.subplots()
    im = ax.imshow(act_over_class_.cpu().numpy(), aspect='auto')
    ax.set_xlabel("Classes")
    ax.set_ylabel("Blocks")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean activation" if not USE_ATTR else "Mean attribution")
    plt.savefig(scratch + 'block_vs_class' + suffix + '.png')
    plt.close()

    return act_over_sample, act_over_class

##########
# Intervention
##########

@torch.no_grad()
def run_checking_ablation(cnn_model, data_loader, device, feature_labels, all_labels, n_blocks, suffix='', n_thresholds = 20):
    """
    For all input, ablate all blocks that are most active on that input. Record accuracy as % of ablation.
    Report result as a curve of accuracy vs % of ablation.
    """
    ABLATE_ALL = True # whether to ablate all blocks assigned to a class, or only those that activate *only* on that class
    
    inputs = data_loader.data.unsqueeze(1).float() / 255
    labels = data_loader.targets

    n_samples = inputs.size(0)

    thresholds = torch.linspace(0, 1, n_thresholds).to(device)
    accuracy = torch.zeros(n_thresholds).to(device)
    completeness = torch.zeros(n_thresholds).to(device)

    act_buffer = []
    hook = cnn_model.layer2.register_forward_hook(lambda module, input, output: act_buffer.append(output))
    for t, threshold in enumerate(thresholds):
        print(f"Threshold {threshold.item()}")
        for i in tqdm(range(n_samples // 100)):
            # sort blocks by activation on this sample
            _ = cnn_model(inputs[i*100:(i+1)*100].to(device))
            sample_act = act_buffer.pop()
            sample_act = sample_act.reshape(sample_act.size(0), n_neurons)
            if USE_ATTR:
                sample_act = IG(cnn_model, sample_act, labels[i*100:(i+1)*100], device)
            # mean per block
            sample_act_mean = torch.zeros(n_blocks, 100).to(device)
            for label in all_labels:
                feature_idx = (feature_labels == label)
                sample_act_mean[label, :] = sample_act[:, feature_idx].mean(dim=(1))

            perm = torch.argsort(sample_act_mean, descending=True, dim=0)
            blocks_to_ablate = perm[:int(threshold.item() * n_blocks), :]

            ablate_idx = torch.zeros(100, feature_labels.size(0)).to(device).bool()
            for k in range(100):
                for block in blocks_to_ablate[:, k]:
                    ablate_idx[k] |= (feature_labels == block)
            
            cnn_model.eval()
            class_inputs = inputs[i*100:(i+1)*100].to(device)
            class_labels = labels[i*100:(i+1)*100].to(device)

            _ = cnn_model(class_inputs)
            act = act_buffer.pop()
            act = act.reshape(act.size(0), n_neurons)

            # ablate the target class :
            f = act.clone() # shape : n_samples x n_neurons
            f[ablate_idx] = 0
            logits = cnn_model.classifier(f) # shape : n_samples x num_classes
            pred = logits.argmax(dim=1) # shape : n_samples
            acc = (pred == class_labels).float().mean()
            accuracy[t] += acc / (n_samples // 100)

            # ablate all but the target class :
            f = act.clone()
            f[~ablate_idx] = 0
            logits = cnn_model.classifier(f)
            pred = logits.argmax(dim=1)
            comp = (pred == class_labels).float().mean()
            completeness[t] += comp / (n_samples // 100)

        print(f"Threshold {threshold.item()}, Accuracy: {accuracy[t].item()}")
    hook.remove()

    thresholds = thresholds.cpu()

    # Plot Baselines :
    rand_accuracy = 1 / num_classes
    plt.plot(thresholds, torch.ones_like(thresholds) * rand_accuracy, label="Random baseline", linestyle='--', color='black')

    original_accuracy = torch.zeros_like(thresholds) + accuracy[0].cpu() # accuracy with no ablation
    plt.plot(thresholds, original_accuracy, label="Unmodified prediction baseline", color='black', linestyle=':')

    # Plot our curve :
    plt.plot(thresholds, accuracy.cpu().numpy(), label="Our method - SBM")

    # Plot "checking paper" curve :
    checking_thresholds = torch.tensor([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.41, 0.45, 0.5, 0.54, 0.59, 1.0])
    checking_accuracy = torch.tensor([0.99, 0.97, 0.98, 0.95, 0.91, 0.85, 0.89, 0.9, 0.92, 0.88, 0.82, 0.85, 0.81, 0.805, 0.7, 0.81, 0.08])
    plt.plot(checking_thresholds, checking_accuracy, label="Lu et al. - Biclusters")

    plt.xlabel("ablated percentage")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(scratch + 'ablation_per_sample' + suffix + '.png')
    plt.close()

    plt.plot(thresholds, completeness.cpu().numpy())
    plt.xlabel("untouched percentage")
    plt.ylabel("Accuracy")
    plt.savefig(scratch + 'completeness_per_sample' + suffix + '.png')
    plt.close()

@torch.no_grad()
def run_ablation(cnn_model, data_loader, device, feature_labels, all_labels, act_over_class, suffix=''):
    """
    For all classes, ablate all blocks that are most active on that class. Record accuracy class-wise.
    Report result in a matrix class x class. (ablated x measured)
    """
    ABLATE_ALL = True # whether to ablate all blocks assigned to a class, or only those that activate *only* on that class
    
    inputs = data_loader.data.unsqueeze(1).float() / 255
    labels = data_loader.targets

    acc_matrix = torch.zeros(num_classes, num_classes).to(device)
    completeness_matrix = torch.zeros(num_classes, num_classes).to(device)

    feature_block_labels = feature_labels.clone()
    feature_class_labels = torch.zeros_like(feature_labels) # feature_class_labels[i] = j means that block b_i is most active on class j

    for block_idx in all_labels:
        class_idx = act_over_class[block_idx].argmax()
        feature_class_labels[feature_block_labels == block_idx] = class_idx

    act_buffer = []
    hook = cnn_model.layer2.register_forward_hook(lambda module, input, output: act_buffer.append(output))
    for i, class_ablate in enumerate(range(num_classes)):
        for class_measure in range(num_classes):
            ablate_idx = (feature_class_labels == class_ablate) # TODO : if not ABLATE_ALL, change this
            input_idx = (labels == class_measure)
            
            cnn_model.eval()
            class_inputs = inputs[input_idx].to(device)
            class_labels = labels[input_idx].to(device)

            _ = cnn_model(class_inputs)
            act = act_buffer.pop()
            act = act.reshape(act.size(0), n_neurons)

            # ablate the target class :
            f = act.clone() # shape : n_samples x n_neurons
            f[:, ablate_idx] = 0
            logits = cnn_model.classifier(f) # shape : n_samples x num_classes
            pred = logits.argmax(dim=1) # shape : n_samples
            acc = (pred == class_labels).float().mean()
            acc_matrix[class_ablate, class_measure] = acc

            # ablate all but the target class :
            f = act.clone()
            f[:, ~ablate_idx] = 0
            logits = cnn_model.classifier(f)
            pred = logits.argmax(dim=1)
            comp = (pred == class_labels).float().mean()
            completeness_matrix[class_ablate, class_measure] = comp
    hook.remove()

    plt.imshow(acc_matrix.cpu().numpy())
    cbar = plt.colorbar()
    cbar.set_label("Accuracy")
    plt.xlabel("Measured class")
    plt.ylabel("Ablated class")
    plt.savefig(scratch + 'ablation' + suffix + '.png')
    plt.close()

    plt.imshow(completeness_matrix.cpu().numpy())
    cbar = plt.colorbar()
    cbar.set_label("Accuracy")
    plt.xlabel("Measured class")
    plt.ylabel("Kept class")
    plt.savefig(scratch + 'completeness' + suffix + '.png')
    plt.close()

def get_most_activating_input(act_over_sample, act_over_class, train_dataset):
    most_act_sample = []

    inputs = train_dataset.data.unsqueeze(1).float() / 255
    labels = train_dataset.targets

    for block in range(act_over_sample.size(0)):
        class_of_interest = act_over_class[block].argmax()
        # keep only samples from class_of_interest
        class_idx = (labels == class_of_interest)
        samples = act_over_sample[block, class_idx]
        most_act_sample.append(inputs[samples.argmax()])

    return torch.tensor(most_act_sample)

def run_steer(cnn_model, train_dataset, device, feature_labels, all_labels, n_blocks, n_samples, act_over_class, suffix=''):
    pass

##########
# Run everything :
##########

if __name__ == '__main__':
    cnn_model = get_cnn()
    #sae = get_sae()

    cnn_model.eval()
    #sae.eval()

    cov = get_covariance(cnn_model, train_loader, device, n_neurons)
    weights, dead_idx = plot_cov(cov)

    feature_labels = blocks_spectral_clustering(cov, weights, dead_idx)
    all_labels = list(set(feature_labels.cpu().numpy()))
    n_blocks = len(all_labels)
    n_samples = train_dataset.data.size(0)

    act_over_sample, act_over_class = run_block_vs_sample(cnn_model, test_dataset, device, feature_labels, all_labels, n_blocks, suffix='_spectral')
    run_checking_ablation(cnn_model, test_dataset, device, feature_labels, all_labels, n_blocks, suffix='_spectral', n_thresholds=10)
    run_ablation(cnn_model, test_dataset, device, feature_labels, all_labels, act_over_class, suffix='_spectral')

    feature_labels, n_blocks, all_labels = blocks_sbm(cov, weights, dead_idx, device)
    act_over_sample, act_over_class = run_block_vs_sample(cnn_model, test_dataset, device, feature_labels, all_labels, n_blocks, suffix='_sbm')
    run_checking_ablation(cnn_model, test_dataset, device, feature_labels, all_labels, n_blocks, suffix='_sbm', n_thresholds=20)
    run_ablation(cnn_model, test_dataset, device, feature_labels, all_labels, act_over_class, suffix='_sbm')