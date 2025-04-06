# BNN_InfoSharing

This Repo will have the details to implement a Bayesian Neural Network, in which the weight distributions are determined by a common layer by layer prior through Bayesian Information Sharing.

TODO:

- Set up a demo file that fits and Normal BNN and a learnable priors BNN, and shows all the diagnostics of how this impacts the BNN
- Set up a demo of normal BNN and learnable, but for an actual dataset like MNIST



I also want to create a tool that can focus on suspected exploded and vanish gradients, and see how the learnable parameters affected those distributions. Maybe like we look at the top exploding distributions and compare them across models.

I also want to look at how changing the width of these networks could change the learnable priors.

Maybe look into how learnable parameters fit in with the other optimization techniques

Maybe would be a good idea to measure the KL divergence between every posterior distribution (per layer?) and then sum? What would that show?



**Final Clean Up:**

- Look into how I can set up all of my functions into one file that can be pip installed
- Fix up Colab files so that the represent examples of how to use the code
- Fix up code and functions into python files
- Clean up functions and rework them. Maybe that graphing stuff



This is what the parameters log looks like:

distribution_log = {
    epoch: {
        'layer1': {
            'posterior': {
                'weight': {'mu': np.array(...), 'sigma': np.array(...)},
                'bias':   {'mu': np.array(...), 'sigma': np.array(...)}
            },
            'prior': {
                'weight': {'mu': np.array(...), 'sigma': np.array(...)},
                'bias':   {'mu': np.array(...), 'sigma': np.array(...)}
            }
        },
        'layer2': { ... },
        # etc.
    },
    # next epoch...
}




# Bayesian Neural Networks (BNN) Logging & Analysis

This repository provides a set of tools for logging, analyzing, and visualizing the learned distribution parameters in Bayesian Neural Networks (BNNs). The code tracks both posterior (learned from data) and prior (learned or fixed) parameters for every node in a BNN. These tools make it easier to inspect how these distributions evolve during training and how similar they are within each layer.

---

## Features

- **Logging Distribution Parameters**  
  Log the current distribution parameters (means and standard deviations) of weights and biases for each layer during training.

- **Visualization**  
  Plot the evolution of individual parameter distributions over epochs or overlay all distributions in a given epoch.

- **Statistical Analysis**  
  Calculate overall and per-layer summary statistics (mean, standard deviation) for the learned weight distributions.

- **KL Divergence Analysis**  
  Compute and display the pairwise symmetric KL divergences among weight and bias distributions in each layer, including both the sum and average values.

- **Custom BNN with Shared Layer Priors**  
  An example of a custom Bayesian linear layer with shared (hierarchical) priors is provided for inspiration.

---

## Distribution Logging Data Structure

The logging functions record the parameters in a nested dictionary structure called `distribution_log`. The structure is as follows:

```python
distribution_log = { 
    epoch: { 
        'layer_name': { 
            'posterior': { 
                'weight': {'mu': np.array(...), 'sigma': np.array(...)}, 
                'bias': {'mu': np.array(...), 'sigma': np.array(...)} 
            }, 
            'prior': { 
                'weight': {'mu': np.array(...), 'sigma': np.array(...)}, 
                'bias': {'mu': np.array(...), 'sigma': np.array(...)} 
            } 
        }, 
        ... # Other layers 
    }, 
    ... # Other epochs 
}
```

### Traversing the Data Structure

- **By Epoch:**  
  Each top-level key is an epoch (an integer).  
  ```python
  for epoch, epoch_data in distribution_log.items():
      # Process data for each epoch
  ```
- **By Layer:**  
  Within each epoch, keys represent layer names.
  ```python
  for layer_name, layer_data in epoch_data.items():
      # layer_data contains 'posterior' and 'prior'
  ```
- **By Distribution Type:**  
  For each layer, `layer_data` contains two keys: 'posterior' and 'prior'.
  ```python
  posterior_data = layer_data['posterior']
  prior_data = layer_data['prior']
  ```
- **By Parameter Type:**  
  Within each distribution type, parameters are stored under 'weight' and (optionally) 'bias', each containing a dictionary with keys 'mu' and 'sigma'.
  ```python
  weight_mu = layer_data['posterior']['weight']['mu']
  weight_sigma = layer_data['posterior']['weight']['sigma']
  ```

## Provided Functions

1. **get_all_distribution_params(model, epoch, step_size)**
   - **Purpose:**  
     Extracts current distribution parameters from the BNN.
   - **Output:**  
     Returns a nested dictionary (see above) for the specified epoch (only logs every `step_size` epochs).
   - **Usage Example:**
     ```python
     log_data = get_all_distribution_params(model, epoch, step_size=5)
     if log_data:
         distribution_log[epoch] = log_data
     ```

2. **plot_params_log_distributions(distribution_log, num_points=200, dist_type='posterior')**
   - **Purpose:**  
     Plots the evolution of each parameter's Gaussian distribution (using μ and σ) over epochs.
   - **Output:**  
     For each parameter element, creates a figure with curves colored along a gradient from dark red (earlier epochs) to neon green (later epochs).
   - **Usage Example:**
     ```python
     plot_params_log_distributions(distribution_log, num_points=200, dist_type='posterior')
     ```

3. **plot_epoch_distributions(distribution_log, num_points=200, alpha=0.6, dist_type='posterior')**
   - **Purpose:**  
     Overlays all the distributions (for either weights or biases) from a specific epoch.
   - **Output:**  
     A single plot with each distribution in a unique color from the 'viridis' colormap.
   - **Usage Example:**
     ```python
     plot_epoch_distributions(distribution_log, epoch=95, num_points=200, alpha=0.6, dist_type='posterior')
     ```

4. **get_overall_weight_stats(distribution_log, epoch=None, dist_type='posterior')**
   - **Purpose:**  
     Computes aggregated statistics for weight distributions across all layers at a given epoch.
   - **Output:**  
     Returns a table (Pandas DataFrame) with:
       - Overall statistics (first row): mean and standard deviation of μ's and σ's across all layers.
       - Per-layer statistics: same metrics computed for each layer individually.
   - **Usage Example:**
     ```python
     stats_df = get_overall_weight_stats(distribution_log, epoch=50, dist_type='posterior')
     print(stats_df)
     ```

5. **compute_layer_kl_divergences(distribution_log, epoch=None, dist_type='posterior')**
   - **Purpose:**  
     Computes pairwise symmetric KL divergences for the weight (and bias) distributions within each layer.
   - **Output:**  
     Returns a table (Pandas DataFrame) with, for each layer:
       - Weight KL Sum: Sum of symmetric KL divergences over all weight pairs.
       - Weight KL Average: Average symmetric KL divergence over all weight pairs.
       - Bias KL Sum and Bias KL Average: Similar metrics for biases.
   - **Usage Example:**
     ```python
     kl_table = compute_layer_kl_divergences(distribution_log, epoch=95, dist_type='posterior')
     print(kl_table)
     ```

## Defining Your Own Bayesian Neural Network

Below is a simplified example of how to define a custom Bayesian linear layer with shared layer priors (a hierarchical Bayesian linear layer).

```python
import torch
import torch.nn as nn
import math
from torch.nn import Parameter

class HierarchicalBayesLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(HierarchicalBayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Posterior parameters for weights (learned to fit the data)
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        
        # Shared prior parameters for weights
        self.prior_mu = Parameter(torch.zeros(1, 1))
        self.prior_log_sigma = Parameter(torch.full((1, 1), -3.0))

        if bias:
            # Posterior parameters for bias
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            # Shared prior for bias
            self.prior_bias_mu = Parameter(torch.zeros(1))
            self.prior_bias_log_sigma = Parameter(torch.full((1,), -3.0))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_parameter('prior_bias_mu', None)
            self.register_parameter('prior_bias_log_sigma', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize posterior parameters for weights
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -3.0)
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, input):
        # For simplicity, only returning the mean prediction
        return nn.functional.linear(input, self.weight_mu, self.bias_mu)
```

### Using Your Custom Layer

You can incorporate `HierarchicalBayesLinear` into your network like this:

```python
import torch.nn.functional as F

class SimpleBNN(nn.Module):
    def __init__(self):
        super(SimpleBNN, self).__init__()
        self.fc1 = HierarchicalBayesLinear(1, 10)
        self.fc2 = HierarchicalBayesLinear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Example model instantiation:
bnn_model = SimpleBNN()
```

## How to Run

### Install Dependencies:
Make sure you have PyTorch, NumPy, Pandas, and Matplotlib installed.

```bash
pip install torch numpy pandas matplotlib
```

### Log and Analyze Distributions:
Use the provided functions to log distribution parameters during training and later analyze them.

Example training loop snippet:

```python
distribution_log = {}
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    mse = mse_loss(output, y)
    kl = kl_loss(model)
    loss = mse + kl_weight * kl
    loss.backward()
    optimizer.step()
    
    log_data = get_all_distribution_params(model, epoch, step_size=5)
    if log_data:
        distribution_log[epoch] = log_data
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: MSE = {mse.item():.4f}, KL = {kl.item():.4f}")
```

### Visualize and Compute Statistics:
Use the plotting and statistical analysis functions as described in the sections above.

## Summary

This repository provides a modular approach to:

- Log the evolution of distribution parameters in a BNN.
- Visualize and analyze these parameters through various plots and summary tables.
- Customize Bayesian layers with shared priors to experiment with hierarchical models.

Feel free to modify and extend these tools to fit your specific research or application needs!

This README is complete and can be easily copied and pasted into your GitHub repository.

