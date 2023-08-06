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


