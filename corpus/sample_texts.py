"""
Small built-in sample texts for unit tests and dry runs.
Each entry mirrors the TextEntry schema used by CorpusLoader.
"""

from __future__ import annotations
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Format:
#   {
#     "id": str,
#     "domain": str,
#     "text": str,
#     "questions": [{"question": str, "answer": str}, ...],
#     "reference_summary": str,
#   }
# ---------------------------------------------------------------------------

SAMPLE_TEXTS: List[Dict[str, Any]] = [
    {
        "id": "enc_001",
        "domain": "encyclopedic",
        "text": (
            "The French Revolution was a period of radical political and societal change in France "
            "that began with the Estates General of 1789 and ended with the formation of the French "
            "Consulate in November 1799. Many of its ideas are considered fundamental principles of "
            "liberal democracy, while the values and institutions it created remain central to modern "
            "French political discourse. The Revolution resulted in the abolition of feudalism and "
            "the old regime, the proclamation of the Declaration of the Rights of Man and of the "
            "Citizen, and the execution of King Louis XVI. The causes of the Revolution included "
            "financial difficulties, social inequality, and the influence of Enlightenment ideas. "
            "The period was marked by significant violence, including the Reign of Terror, during "
            "which thousands of people were executed by guillotine. Napoleon Bonaparte rose to "
            "prominence during this period and eventually took control of the French government."
        ),
        "questions": [
            {"question": "When did the French Revolution begin?", "answer": "1789"},
            {"question": "When did the French Revolution end?", "answer": "November 1799"},
            {"question": "Who was executed during the French Revolution?", "answer": "King Louis XVI"},
            {"question": "What was the name of the violent period during the Revolution?", "answer": "Reign of Terror"},
            {"question": "Who rose to prominence and took control of the French government?", "answer": "Napoleon Bonaparte"},
        ],
        "reference_summary": (
            "The French Revolution (1789–1799) was a period of radical political change in France "
            "that abolished feudalism, proclaimed the Rights of Man, and executed King Louis XVI. "
            "It ended with Napoleon Bonaparte taking control of the government."
        ),
    },
    {
        "id": "nar_001",
        "domain": "narrative",
        "text": (
            "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, "
            "his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly "
            "through the glass doors of Victory Mansions, though not quickly enough to prevent a "
            "swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage "
            "and old rag mats. At one end of it a coloured poster, too large for the hallway, had "
            "been tacked to the wall. It depicted simply an enormous face, more than a metre wide: "
            "the face of a man of about forty-five, with a heavy black moustache and ruggedly "
            "handsome features. Winston made for the stairs. It was no use trying the lift. Even at "
            "the best of times it was seldom working, and at present the electric current was cut off "
            "during daylight hours. It was part of the economy drive in preparation for Hate Week. "
            "The flat was seven flights up, and Winston, who was thirty-nine and had a varicose ulcer "
            "above his right ankle, went slowly, resting several times on the way."
        ),
        "questions": [
            {"question": "What was the weather like at the start of the passage?", "answer": "bright cold"},
            {"question": "What building did Winston Smith enter?", "answer": "Victory Mansions"},
            {"question": "What did the hallway smell of?", "answer": "boiled cabbage and old rag mats"},
            {"question": "How old was Winston Smith?", "answer": "thirty-nine"},
            {"question": "What event was the economy drive in preparation for?", "answer": "Hate Week"},
        ],
        "reference_summary": (
            "Winston Smith enters Victory Mansions on a cold April day, passing a large poster of a "
            "man's face. He climbs seven flights of stairs because the lift is broken, struggling "
            "with a varicose ulcer on his ankle."
        ),
    },
    {
        "id": "sci_001",
        "domain": "scientific",
        "text": (
            "Large language models (LLMs) have demonstrated remarkable capabilities across a wide "
            "range of natural language processing tasks. These models, trained on vast corpora of "
            "text data, exhibit emergent abilities that were not explicitly programmed. Recent "
            "research has focused on understanding the scaling laws that govern LLM performance, "
            "showing that model capabilities tend to improve predictably with increases in model "
            "size, training data, and compute. However, certain capabilities appear to emerge "
            "abruptly at specific scale thresholds, a phenomenon termed emergent abilities. "
            "The mechanisms underlying these emergent abilities remain poorly understood. "
            "Interpretability research aims to shed light on the internal representations and "
            "computations of LLMs, with the goal of making these models more transparent and "
            "trustworthy. Key challenges include the high dimensionality of model activations, "
            "the distributed nature of knowledge representation, and the difficulty of attributing "
            "model outputs to specific input features."
        ),
        "questions": [
            {"question": "What phenomenon describes capabilities that appear abruptly at scale thresholds?", "answer": "emergent abilities"},
            {"question": "What does interpretability research aim to do?", "answer": "make models more transparent and trustworthy"},
            {"question": "What three factors govern LLM performance scaling?", "answer": "model size, training data, and compute"},
            {"question": "What is a key challenge in interpretability?", "answer": "high dimensionality of model activations"},
            {"question": "What are LLMs trained on?", "answer": "vast corpora of text data"},
        ],
        "reference_summary": (
            "Large language models show emergent abilities at scale thresholds that are not fully "
            "understood. Interpretability research seeks to make LLMs more transparent by studying "
            "their internal representations, facing challenges like high dimensionality and "
            "distributed knowledge."
        ),
    },
    {
        "id": "conv_001",
        "domain": "conversational",
        "text": (
            "Alice: Have you seen the new coffee shop that opened on Main Street? "
            "Bob: No, I haven't. Is it any good? "
            "Alice: It's amazing! They have this lavender latte that I can't stop thinking about. "
            "Bob: Lavender in coffee? That sounds unusual. "
            "Alice: I know, I was skeptical too, but it works really well. The floral notes "
            "complement the espresso perfectly. "
            "Bob: Maybe I'll give it a try. What's the place called? "
            "Alice: It's called The Morning Bloom. They also do really good avocado toast. "
            "Bob: Classic. Every new cafe has avocado toast these days. "
            "Alice: True, but theirs has a poached egg and chili flakes on top, which makes it "
            "actually worth ordering. "
            "Bob: Alright, you've convinced me. Want to go this weekend? "
            "Alice: Absolutely. Saturday morning works for me."
        ),
        "questions": [
            {"question": "What is the name of the new coffee shop?", "answer": "The Morning Bloom"},
            {"question": "What unusual drink does Alice recommend?", "answer": "lavender latte"},
            {"question": "What food item does the cafe also serve?", "answer": "avocado toast"},
            {"question": "What toppings are on the avocado toast?", "answer": "poached egg and chili flakes"},
            {"question": "When do Alice and Bob plan to visit?", "answer": "Saturday morning"},
        ],
        "reference_summary": (
            "Alice tells Bob about a new coffee shop called The Morning Bloom, recommending their "
            "lavender latte and avocado toast with poached egg and chili flakes. They agree to "
            "visit together on Saturday morning."
        ),
    },
    {
        "id": "proc_001",
        "domain": "procedural",
        "text": (
            "To make a classic French omelette, start by cracking three large eggs into a bowl. "
            "Add a pinch of salt and whisk vigorously until the yolks and whites are fully "
            "combined and the mixture is slightly frothy. Heat a small non-stick pan over "
            "medium-high heat and add one tablespoon of unsalted butter. When the butter has "
            "melted and begins to foam, pour in the egg mixture. Using a rubber spatula, "
            "immediately begin stirring the eggs in small circular motions while simultaneously "
            "shaking the pan. Continue this motion for about 30 seconds until the eggs are "
            "mostly set but still slightly wet on top. Remove the pan from heat. Tilt the pan "
            "at a 45-degree angle and use the spatula to fold one third of the omelette over "
            "the center. Then roll the omelette onto a plate so it forms a neat cylinder. "
            "The finished omelette should be pale yellow with no browning."
        ),
        "questions": [
            {"question": "How many eggs are needed?", "answer": "three"},
            {"question": "What type of butter should be used?", "answer": "unsalted butter"},
            {"question": "How long should you stir the eggs?", "answer": "about 30 seconds"},
            {"question": "What angle should you tilt the pan?", "answer": "45-degree angle"},
            {"question": "What color should the finished omelette be?", "answer": "pale yellow"},
        ],
        "reference_summary": (
            "A classic French omelette is made by whisking three eggs, cooking them in butter "
            "while stirring constantly, then folding and rolling the omelette onto a plate. "
            "The result should be pale yellow with no browning."
        ),
    },
]

# Made with Bob
