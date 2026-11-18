"""Second-round MedMCQA debate prompt."""

DEBATE_UPDATE_SYSTEM = ""

DEBATE_UPDATE_USER = """These are the solutions to the problem from other agents: {peer_solutions}

Using the reasoning from other agents as additional advice, give an updated answer. Examine their reasoning and your own step by step. Put your answer in the form (X) at the end of your response."""

DEBATE_UPDATE_PEER_SOLUTION = """

One agent solution: ```{peer_response}```"""
