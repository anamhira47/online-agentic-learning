# Sample data
from feedback import Framework, OurArguments
from tasks import get_task

samples = [
    {
        "data": {
            "sentence1": "The chicken is going to cross the road.",
            "sentence2": "I can't wait to cross the finish line.",
            "word": "cross"
        }
    },
    {
        "data": {
            "sentence1": "He broke the window with a rock.",
            "sentence2": "He's trying to break the record.",
            "word": "break"
        }
    },
    {
        "data": {
            "sentence1": "She will perform a song at the concert.",
            "sentence2": "The doctor will perform the surgery.",
            "word": "perform"
        }
    },
    {
        "data": {
            "sentence1": "He will present his project to the class.",
            "sentence2": "This is a present for you.",
            "word": "present"
        }
    },
    {
        "data": {
            "sentence1": "I left my keys on the table.",
            "sentence2": "He left the party early.",
            "word": "left"
        }
    },
    {
        "data": {
            "sentence1": "She likes to play the piano.",
            "sentence2": "The children are playing in the park.",
            "word": "play"
        }
    },
    {
        "data": {
            "sentence1": "He's going to check the mail.",
            "sentence2": "Can you check the answer for me?",
            "word": "check"
        }
    },
    {
        "data": {
            "sentence1": "I need to charge my phone.",
            "sentence2": "The bull charged at the matador.",
            "word": "charge"
        }
    },
    {
        "data": {
            "sentence1": "He's going to pass the ball.",
            "sentence2": "I hope I pass the test.",
            "word": "pass"
        }
    },
    {
        "data": {
            "sentence1": "She's going to light the candle.",
            "sentence2": "It's light outside.",
            "word": "light"
        }
    }
]

# Instantiate the template
#template = WICTemplate()

# Use the template to encode the sample
args = OurArguments()
print(args.task_name)

framework = Framework(args, get_task(args.task_name))
