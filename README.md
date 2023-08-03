# Online Agentic Learning

- Using Context Free grammars for deterministic model parsing.

- Speculative sampling for faster generations of tokens. Using rejection sampling based on CFG state.
    - Using smaller model to generate tokens for the grammar structure
    - Using larger LLM for decisions 
- Fine Tuning with forward passes to learn use case with forward passes only at inference time


The main idea is for use cases where you need an LLM agent so you need to 
- constrain the generation to get useful outputs
- be able to learn from previous actions, there are multiple ways to achieve this
    - fine tuning with forward passes (using random perturbations)
    - normal fine tuning
    - rlhf
Fine tuning with forward passes has the benefit that the training can be done at inference time, hence you have an online improvement scheme. 

Bringing these ideas together you get a framework where
- you always get useful outputs and mitigate the case where LLM chaining causes
a compounding of error 
- You improve your agent at inference time by confirming actions. Allows for low overhead improvement of the model 


TODO:

    Constrained sampling
        - simple jsonformer type implementation (DONE)
        - Speculative sampling w/ rejection samplign
        - SMC steering w/ speculative samping
        - Mask w/FSM with pushdown automato
    Learning 
        - Implement simple fine tuning with forward passes
    
    UX
        - increase ergonomics for ease of use



# Building on the shoulders of giants :)

Inspirations:

- [Paper 1](https://arxiv.org/pdf/2302.01318.pdf)
- [Paper 2](https://arxiv.org/pdf/2305.17333.pdf)

