# Class for increased ergonomics when using the Online Agent



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


Current Optimization debt

- Better method for constrained generation using SMC sampling and pushdown automata
- Speculative sampling for quicker inference generation
- RoPe scaling
- Overall better inference setup practices (huggingface side)

Some Ideas to explore

- Distributed learning paradigm
    - m_1,...m_n small nodes with larger LLM for fwd pass learning (memory time tradeoff)

    

