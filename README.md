# Problem Statement 

Build a generative model, e.g. a VAE, that can encode an image (x) to a learned latent representation, z or q(z).

We want this latent representation to be invariant to small transformations (small rotations (+/-10 degrees), scales (+/- 20%) and translations (+/- 4 pixels) i.e. q(z|x) = q(z|T(x)).

The accompanying latent space should be appropriate for classification using a simple linear classifier, and the classification accuracy should be evaluated.

# Feedback 

Very nicely presented, clean template. Good use of appropriate equations and figures, nice to see diagrammatic representations of the model and examples from the data. Clean references.
 
Note extrapolation artefacts in the transformed images - could use different types of padding.
 
How was Figure 4 made?
 
The figure with confusion matrix is not labelled?
 
Good comparison between baselines models and different variants.
 
One potential issue in the loss function is that you're using the MSE as the likelihood function, which is like a Gaussian negative log likelihood with a variance of 1! This means the latent space may be over-regularised as you've normalised your data to have a variance of 1.
 
Overall, seems like a nice investigation - a problem with plugging the STNs in is the decoder needs to know what transformation was applied to the feature representation - plus your STN training pains.
 
A good effort!