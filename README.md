# BNN_InfoSharing

This Repo will have the details to implement a Bayesian Neural Network, in which the weight distributions are determined by a common layer by layer prior through Bayesian Information Sharing.

TODO:

I need a way of showing that the posterior distributions treated with learnable priors are are significantly "different" from the posterior weight distributions of the vanilla bayes model. So I need some kind of metric that measures the differences between ensambles of distributions.

At the end of the day alot of this project will be about creating diagnostics that I can use to measure these distributions.

I also want to create a tool that can focus on suspected exploded and vanish gradients, and see how the learnable parameters affected those distributions. Maybe like we look at the top exploding distributions and compare them across models.

I also want to look at how changing the width of these networks could change the learnable priors.

Maybe look into how learnable parameters fit in with the other optimization techniques
