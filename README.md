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

