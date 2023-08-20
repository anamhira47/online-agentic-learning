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



## Setting Up the Environment

Before you can run the code, you'll need to set up a Python environment. We recommend using [Conda](https://docs.conda.io/en/latest/), which is a package manager that works well for managing Python environments and packages.

Here are the steps to create a new Conda environment and install PyTorch:

1. **Install Conda**

   If you haven't installed Conda yet, you can download it from the [official website](https://docs.conda.io/en/latest/miniconda.html). Choose the Python 3 version.

2. **Create a new Conda environment**

   Open a terminal and run the following command to create a new Conda environment named `myenv`:


   ```bash
   conda create --name myenv
   ```

   You can replace `myenv` with any name you like.

3. **Activate the Conda environment**

   Use the following command to activate the environment:

   ```
   conda activate myenv
   ```




4. **Install PyTorch**

   You can install PyTorch in your Conda environment with the following command:

```
conda install pytorch torchvision torchaudio -c pytorch
```


Current plan for ergonomics.

Looking to implement something useable by istantiating an OnlineLearning object
think something like 
```pip install onlinelearning
```
JSON_SCHEMA=json_schema
LLM=<model_name>
agent = Online(JSON_SCHEMA, LLM)

# generate method to generate based on json schema and given prompt
agent.generate(prompt)

# feedback method to take feedback from human or other programmatic part of the environment to use for online fine tuning

agent.feedback(feedback)

```


TODO AUG 16

- Evaluate MIND2Web w/ seed data
- Evaluate positive reinforcement style training
- Evaluate generic multi choice dataset w/ online after building seed corpus
AUG 17 onwards (other tasks todo)
 - Setup observabilitiy
 - Setup Live example of web agent w/candidate generation + inference
 - Setup multi agent w/critic to decentralize learning
 


# Building on the shoulders of giants :)

Inspirations:

- [Paper 1](https://arxiv.org/pdf/2302.01318.pdf)
- [Paper 2](https://arxiv.org/pdf/2305.17333.pdf)

