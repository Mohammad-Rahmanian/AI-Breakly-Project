# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from sys import maxsize


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        preFood = currentGameState.getFood()
        total_score, closest_food_distance, closest_ghost_distance, closest_ghost_index = 0, 0, 0, 0
        food_difference = len(preFood.asList()) - len(newFood.asList())
        for index, food in enumerate(newFood.asList()):
            if manhattanDistance(food, newPos) < closest_food_distance or index == 0:
                closest_food_distance = manhattanDistance(food, newPos)
        for index, ghost in enumerate(newGhostStates):
            if manhattanDistance(ghost.getPosition(), newPos) < closest_ghost_distance or index == 0:
                closest_ghost_distance = manhattanDistance(ghost.getPosition(), newPos)
                closest_ghost_index = index
        closest_ghost_scared = False
        for index, remained_time in enumerate(newScaredTimes):
            if remained_time > 0 and index == closest_ghost_index:
                closest_ghost_scared = True
                break
        if closest_ghost_scared:
            total_score += (1000 * food_difference) + (- closest_food_distance)
        else:
            total_score += (100 * food_difference) + (- closest_food_distance) + (-closest_ghost_distance)
            if closest_ghost_distance < 2:
                total_score -= (1000 * (closest_ghost_distance + 1))
        return total_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        return self.max_value(gameState, 0, 0)[0]

    def min_value(self, gameState, depth, agentIndex):
        current_action = None
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return current_action, self.evaluationFunction(gameState)
        min_val = maxsize
        for action in gameState.getLegalActions(agentIndex):
            if (agentIndex + 1) % gameState.getNumAgents() == 0:
                _, value = self.max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
                if value < min_val:
                    min_val = value
                    current_action = action
            else:
                _, value = self.min_value(gameState.generateSuccessor(agentIndex, action), depth,
                                          (agentIndex + 1) % gameState.getNumAgents())
                if value < min_val:
                    min_val = value
                    current_action = action

        return current_action, min_val

    def max_value(self, gameState, depth, agentIndex):
        current_action = None
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return current_action, self.evaluationFunction(gameState)
        max_val = -maxsize
        for action in gameState.getLegalActions(agentIndex):
            _, value = self.min_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value > max_val:
                max_val = value
                current_action = action
        return current_action, max_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0, -maxsize, maxsize)[0]

    def max_value(self, gameState, depth, agent_index, alpha, beta):
        current_action = None
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return current_action, self.evaluationFunction(gameState)

        max_val = -maxsize
        for action in gameState.getLegalActions(agent_index):
            _, value = self.min_value(gameState.generateSuccessor(agent_index, action), depth + 1, agent_index + 1,
                                      alpha, beta)

            if value > max_val:
                max_val = value
                current_action = action

            if max_val > beta:
                return current_action, max_val

            if max_val > alpha:
                alpha = max_val

        return current_action, max_val

    def min_value(self, gameState, depth, agent_index, alpha, beta):
        current_action = None
        if gameState.isWin() or gameState.isLose():
            return current_action, self.evaluationFunction(gameState)

        min_val = maxsize
        for action in gameState.getLegalActions(agent_index):
            if (agent_index + 1) % gameState.getNumAgents() == 0:
                _, value = self.max_value(gameState.generateSuccessor(agent_index, action), depth, 0, alpha, beta)
            else:
                _, value = self.min_value(gameState.generateSuccessor(agent_index, action), depth, agent_index + 1,
                                          alpha, beta)

            if value < min_val:
                min_val = value
                current_action = min_val

            if min_val < alpha:
                return current_action, min_val

            if min_val < beta:
                beta = min_val

        return current_action, min_val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0)[0]
        # util.raiseNotDefined()

    def average_value(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return self.evaluationFunction(gameState)
        sum_value = 0
        for action in gameState.getLegalActions(agentIndex):
            if (agentIndex + 1) % gameState.getNumAgents() == 0:
                _, value = self.max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                value = self.average_value(gameState.generateSuccessor(agentIndex, action), depth,
                                           (agentIndex + 1) % gameState.getNumAgents())

            sum_value += value
        expected_value = sum_value / len(gameState.getLegalActions(agentIndex))
        return expected_value

    def max_value(self, gameState, depth, agentIndex):
        currentAction = None
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return currentAction, self.evaluationFunction(gameState)
        max_val = -maxsize
        for action in gameState.getLegalActions(agentIndex):
            value = self.average_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value > max_val:
                max_val = value
                currentAction = action
        return currentAction, max_val


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    total_score, closest_food_distance, closest_ghost_distance = 0, 0, 0
    for index, food in enumerate(foods.asList()):
        if manhattanDistance(food, pacmanPosition) < closest_food_distance or index == 0:
            closest_food_distance = manhattanDistance(food, pacmanPosition)
    for index, ghost_pos in enumerate(ghostPositions):
        if manhattanDistance(ghost_pos, pacmanPosition) < closest_ghost_distance or index == 0:
            closest_ghost_distance = manhattanDistance(ghost_pos, pacmanPosition)

    scared_ghost_number = 0
    for remained_time in scaredTimers:
        if remained_time > 0:
            scared_ghost_number += 1

    total_score -= (1000 * len(currentGameState.getCapsules()))
    total_score -= (100 * len(foods.asList()))
    total_score -= (5 *closest_food_distance)
    total_score -= scared_ghost_number
    if closest_ghost_distance < 2:
        if closest_ghost_distance == 0:
            total_score -= (10000 * (closest_ghost_distance + 1))
        else:
            total_score -= (10000 * closest_ghost_distance)

    return total_score


# Abbreviation
better = betterEvaluationFunction
