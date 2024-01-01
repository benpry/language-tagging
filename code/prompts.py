from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

example_formatter = """sentence: {sentence}
Tag: {tag}"""

example_prompt = PromptTemplate(
    input_variables=["sentence", "tag"], template=example_formatter
)

suffix = "sentence: {sentence}\nTag:"

dynamics_prefix = """Classify the following as one of:

1. dynamics, or how the world works including explanations or affordances
2. not dynamics, or how the world works including explanations or affordances"""

policy_prefix = """Classify the following as one of:

1. policy, or what actions to take including strategies or instructions
2. not policy, or what actions to take including strategies or instructions"""

abstract_prefix = """Classify the following as one of:

1. abstract, complex, high-level information
2. concrete, simple, low-level information
3. ignorance statements or specific experiences"""

valence_prefix = """Classify the following as one of:

1. winning, including mentions of scoring points, victory, success, goals, solutions, best strategies
2. neutral information
3. losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped"""

abstract_examples = [
    {
        "sentence": "Press the spacebar to fire bullets.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Letting the yellow cube change all the green to blue results in a game over.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Be careful not to box yourself in.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "Use Space bar to shoot the falling orange blocks.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "You can safely walk over the red squares.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Beware of the random moving squares.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "The light blue will kill you.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "The best approach is to carve the path through the grey blocks while avoiding light blue and blue blocks altogether.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "I don't understand how to win",
        "tag": "ignorance statements or specific experiences",
    },
    {
        "sentence": "avoid green squares unless you really have to - they will cost you a life!",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "Get the red and yellow blocks.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Using the spacebar key will destroy a square in front of you at the cost of 1 level point (you will regain the points once the red squares make it to the green one).",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "This can be dangerous if overused -- you need some walls to guide the red squares.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "Touching a red square will turn it yellow.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Go for white blocks first and then move.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Avoid pink & orange blocks.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "If all the cubes have been hit, you lose a life.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Later levels it is easy to get stuck!",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "You can use the green blocks to protect yourself by pushing them.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "Push the BLUE box into GREEN boxes to move them, and push all GREEN boxes to a RED box to beat the level.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Move with ARROW KEYS.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Red, Orange, pink, green.",
        "tag": "ignorance statements or specific experiences",
    },
    {
        "sentence": "Do NOT let the yellow touch you either.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Try to get the yellow block.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Do not touch light blue blocks, or you will die.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Grab the dark orange square and bring it to the green square.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Every movement counts.",
        "tag": "abstract, complex, high-level information",
    },
    {
        "sentence": "You play as the dark blue tile.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Once the purple square hits the light blue square, they vanish.",
        "tag": "concrete, simple, low-level information",
    },
    {
        "sentence": "Remove all blue squares to win.",
        "tag": "concrete, simple, low-level information",
    },
]

dynamics_examples = [
    {
        "sentence": "Carry the red block to the green goal.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You control the darkest blue square.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Push the BLUE box into GREEN boxes to move them, and push all GREEN boxes to a RED box to beat the level.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "All GREEN boxes turning PURPLE will also cost a life.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Be very quick in avoiding the moving boxes, and choose the right moments to move.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "concentrate on the bottom yellow blocks, save them somehow.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Green boxes cannot move through other green boxes and you cannot push more than one box at a time.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Your goal is to get to the YELLOW box.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You will complete the level by getting rid of all the blue blocks.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You play as the blue cube.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Go for white blocks first and then move.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "It is not worthwhile trying to protect all the squares, as it does not seem to yield more points.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You play as the dark blue tile.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "focus on targets.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You are not able to push multiple blocks together.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Red blocks will spawn from the purple block and will try to reach the green block.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "The best strategy for this game is to rapidly use the arrow keys on your keyboard to the green squares whilst avoiding the red ones.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Here is a hierarchy of BOLDEST to least bold colors, so gathering the color furthest left on this list will end the level.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "If there are no red blocks, get all the blocks there are.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": 'Use the Space Bar to "shoot" a BROWN box from the BLUE box, in the last direction the BLUE box moved.',
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Not sure, was unsuccessful.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "The green square is a trigger.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Not sure what green block is there for.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Patience is key for this game.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You die if you touch one of the moving squares.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Defending your position from a far side of the map and shooting directly across will prevent enemy squares from reaching their destination.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "I didn't touch a red box while it was moving.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Light blue tiles can harm you.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "Move with ARROW KEYS.",
        "tag": "not dynamics, or how the world works including explanations or affordances",
    },
    {
        "sentence": "You can safely walk over the red squares.",
        "tag": "dynamics, or how the world works including explanations or affordances",
    },
]

policy_examples = [
    {
        "sentence": "Push the green square on top of the red square to get rid of them.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "When they reach the bottom yellow squares, they turn green.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Avoid yellow block.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "You are the blue block.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "The best approach is to carve the path through the grey blocks while avoiding light blue and blue blocks altogether.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Just plan ahead before you click start and it's extremely easy.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "See move ahead of push to not block yourself in.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "You play as the dark blue tile.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Collect all of the LEFT-MOST COLOR.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Not sure, was unsuccessful.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Be careful to stay away from the green squares unless you really have to - they will cost you a life!",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Push Green blocks into the Orange ones to get rid of them if they’re in your way.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "You are the white block this time.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "This game seems to be missing the purple squares.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "make sure to figure out your route before you start.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "I don't understand how to win",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "It appears that the goal is to push all the green squares into any of the red squares.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Avoid touching the yellow square.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "imagine the purple square on the position you want to push it to first.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Solve the puzzle by first gathering enough white boxes to pass through the green boxes and get to the yellow box.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Green boxes cannot move through other green boxes and you cannot push more than one box at a time.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "That's all I know.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "avoid pink & Brown blocks.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "The other cubes will be moving quickly and randomly and will harm you.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "The goal is to get to the yellow cube.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "imagine the purple square on the position you want to push it to first.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "You are not able to push multiple blocks together.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "make sure to figure out your route before you start.",
        "tag": "policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "The moving squares are very fast and do not seem to move in a logical manner.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
    {
        "sentence": "Green is bad.",
        "tag": "not policy, or what actions to take including strategies or instructions",
    },
]

valence_examples = [
    {
        "sentence": "The yellow cube changes all the green cubes to purple.",
        "tag": "neutral information",
    },
    {
        "sentence": "I died on Level 1, but I learned some things.",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {"sentence": "You are the white block this time.", "tag": "neutral information"},
    {
        "sentence": "If you eat all the proper squares you win.",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {
        "sentence": "Just plan ahead before you click start and it's extremely easy.",
        "tag": "neutral information",
    },
    {
        "sentence": "If a red tile hits a light blue tile, your score will go down by 2 points.",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {"sentence": "avoid pink & Brown blocks.", "tag": "neutral information"},
    {
        "sentence": "Your objective of the game is to try to touch the brown square.",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {
        "sentence": 'Use the Space Bar to "shoot" a BROWN box from the BLUE box, in the last direction the BLUE box moved.',
        "tag": "neutral information",
    },
    {
        "sentence": "You generally want to avoid this as many red squares colliding into light blue squares will lower your score.",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {
        "sentence": "You can only push the purple squares one at a time, not when they are stacked against each other.",
        "tag": "neutral information",
    },
    {
        "sentence": "We don't know what the red square does, so be careful.",
        "tag": "neutral information",
    },
    {
        "sentence": "All GREEN boxes turning PURPLE will also cost a life.",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {
        "sentence": "Push the green square on top of the red square to get rid of them.",
        "tag": "neutral information",
    },
    {"sentence": "Watch out for the red blocks.", "tag": "neutral information"},
    {
        "sentence": "The best approach is to carve the path through the grey blocks while avoiding light blue and blue blocks altogether.",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {"sentence": "You control the darkest blue square.", "tag": "neutral information"},
    {
        "sentence": "Do not touch light blue blocks, or you will die.",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {"sentence": "Patience is key for this game.", "tag": "neutral information"},
    {
        "sentence": "A good strategy is going to the side and spamming the spacebar",
        "tag": "neutral information",
    },
    {
        "sentence": "Using the spacebar key will destroy a square in front of you at the cost of 1 level point (you will regain the points once the red squares make it to the green one).",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {
        "sentence": "Red or yellow squares that have changed colours can be reverted by touching them.",
        "tag": "neutral information",
    },
    {"sentence": "You cannot move purple blocks.", "tag": "neutral information"},
    {
        "sentence": "White is your shield and yellow is the goal.",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {
        "sentence": "Don't go too fast when close to the moving squares, they're very erratic.",
        "tag": "neutral information",
    },
    {
        "sentence": "The dark blue will kill you",
        "tag": "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped",
    },
    {
        "sentence": "Combining light red + dark red and pushing the dark red into the light red also results in a yellow.",
        "tag": "neutral information",
    },
    {
        "sentence": "Eventually once everything is destroyed the level should be completed!",
        "tag": "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies",
    },
    {"sentence": "The green square is a trigger.", "tag": "neutral information"},
    {"sentence": "Try to get the yellow block.", "tag": "neutral information"},
]

examples = {
    "abstract": abstract_examples,
    "dynamics": dynamics_examples,
    "valence": valence_examples,
    "policy": policy_examples,
}


def make_chat_template(instruction, examples):
    system_message = SystemMessage(content=instruction)
    messages = []
    for example in examples:
        messages.append(HumanMessage(content=example["sentence"]))
        messages.append(AIMessage(content=example["tag"]))
    messages.append(HumanMessagePromptTemplate.from_template("{test_sentence}"))

    return ChatPromptTemplate.from_messages([system_message, *messages])


dynamics_chat = make_chat_template(dynamics_prefix, dynamics_examples)
abstract_chat = make_chat_template(abstract_prefix, abstract_examples)
policy_chat = make_chat_template(policy_prefix, policy_examples)
valence_chat = make_chat_template(valence_prefix, valence_examples)

dynamics = FewShotPromptTemplate(
    examples=dynamics_examples,
    example_prompt=example_prompt,
    prefix=dynamics_prefix,
    suffix=suffix,
    input_variables=["sentence"],
)

abstract = FewShotPromptTemplate(
    examples=abstract_examples,
    example_prompt=example_prompt,
    prefix=abstract_prefix,
    suffix=suffix,
    input_variables=["sentence"],
)

policy = FewShotPromptTemplate(
    examples=policy_examples,
    example_prompt=example_prompt,
    prefix=policy_prefix,
    suffix=suffix,
    input_variables=["sentence"],
)

valence = FewShotPromptTemplate(
    examples=valence_examples,
    example_prompt=example_prompt,
    prefix=valence_prefix,
    suffix=suffix,
    input_variables=["sentence"],
)

prompt_templates = {
    "dynamics": dynamics,
    "policy": policy,
    "abstract": abstract,
    "valence": valence,
}

chat_prompt_templates = {
    "dynamics": dynamics_chat,
    "policy": policy_chat,
    "abstract": abstract_chat,
    "valence": valence_chat,
}

if __name__ == "__main__":
    print(dynamics_chat.format_prompt(test_sentence="TESTxyz").to_messages())
    chatgpt = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = LLMChain(llm=chatgpt, prompt=dynamics_chat)
    resp = chain.run("TEST SENTENCE")
