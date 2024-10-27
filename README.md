Next Steps:
- Implement patch embedding for diffusion transformer

Models:
- GANs (Try different types of GANs - DCGAN, CGAN, WGAN)
- VAEs (Sampling from N in the latent space)
- Diffusion Models (Denosing, Latent, Non-Latent)
- Diffusion Transformer (Latent, Non-Latent)

Techniques to try:
- Basic: Train each model on the given datasets, conditional generation, and then generate each until there are ~8000 of each, except NonDemented
    - Pros: Easy to train, not much compute
    - Cons: Same issue of lack of data for certain classes
- Weighted training: Train each model as basic, but use a weighted loss to give higher priority to the smaller classes
    - Pros: Solve the issue of lack of data
    - Cons: Might cause the opposite issue, will have to try many weight functions
- Dataset Repetition: Repeat the smaller datasets a few times so the model sees all equally
    - Pros: Solve the issue of lack of data
    - Cons: Might cause the model to generate the same type of data for smaller classes
- Dataset Augmentation: Augment the smaller datasets a few times so the model sees all equally
    - Pros: Solve the issue of lack of data
    - Cons: Might cause unnatural generation of data, depends heavily on augmentation
- Unconditional Training: Train a unique model for each class, and then augment using each model
    - Pros: Solves all issues above, as classes are treated as independent
    - Cons: Lots of compute needed to train 4 seperate Models
- Unconditional + Average of Models: Train the 4 models, and then take their weighted average and finetune the new model
    - Pros: While augmentating, saves compute and storage
    - Cons: Still compute needed to train 4 different models