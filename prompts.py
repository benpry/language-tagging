"""
define prompts for tagging.
"""

initial_context = """
You will see sentences written by participants who played a game and wrote messages to each other on how to play the game. The game consisted of a grid of squares that interact with each other. Some example interactions include squares changing the color of other blocks on contact, moving squares to different spots on the board, and destroying squares. In each of these games, the player played as one of the squares, moving it around with the arrow keys. The player might have also been able to fire a projectile by pressing the space bar. Players did not know the rules of the game before they started playing and only had a limited number of lives, so they needed to pass messages to subsequent players to help them play the game.
"""

analogy_prompt = """
Classify the following as one of:

1. analogy, drawing parallels between the game environment and the outside world.
2. not analogy, exclusively describing the game world in terms of squares on a board.

Sentences that describe the game world using synonyms of words inside the game environment, like "blocks" or "boxes", should be classified as not analogy.
"""
