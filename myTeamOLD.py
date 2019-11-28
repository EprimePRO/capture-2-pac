# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveReflexAgent', second='AggressiveOffenseAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        (v, action) = self.maxValue(gameState, -2147483648, 2147483648, 0, 'Stop')
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        return action

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        features['successorScore'] = self.getScore(gameState)
        return features

    def getWeights(self, gameState):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}

    def maxValue(self, gameState, al, be, depth, action):
        v = -2147483648  # set v to neg inf
        if gameState.isOver():
            return 60000, action
        if depth > 2:
            return self.evaluate(gameState), action
        actions = gameState.getLegalActions(self.index)
        for act in actions:
            successor = self.getSuccessor(gameState, act)
            m, a = self.minValue(successor, al, be, depth + 1, act)
            v = max(v, m)
            if v == m:
                action = act
            if v >= be:
                return v, act
            al = max(al, v)
        return v, action

    def minValue(self, gameState, al, be, depth, action):
        v = 2147483648  # set v to inf
        if gameState.isOver():
            return 60000, action
        if depth > 2:
            return self.evaluate(gameState), action
        opponents = self.getOpponents(gameState)
        # change this later (might not work may have to do both enemy moves at the same time...)
        actions = []
        successors = []
        for opp in opponents:
            actions.append(gameState.getLegalActions(opp))
        for act in actions[0]:
            succ = gameState.generateSuccessor(opponents[0], act)
            for act1 in actions[1]:
                successors.append(succ.generateSuccessor(opponents[1], act1))

        for succ in successors:
            m, a = self.maxValue(succ, al, be, depth + 1, action)
            v = min(v, m)
            if v <= al:
                return v, action
            be = min(be, v)
        return v, action


class AggressiveOffenseAgent(ReflexCaptureAgent):

    def getFeatures(self, gameState):
        features = util.Counter()
        foodList = self.getFood(gameState).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food
        myState = gameState.getAgentState(self.index)
        action = myState.configuration.direction

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myState.getPosition(), food) for food in foodList])
            features['distanceToFood'] = minDistance

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # get closest enemy distance
        opponents = self.getOpponents(gameState)
        oppPos = [gameState.getAgentState(x).getPosition() for x in opponents]
        oppDist = [self.getMazeDistance(myState.getPosition(), x) for x in oppPos]
        features['enemyDistance'] = 0
        if action != Directions.STOP and action != rev:
            if myState.isPacman and min(oppDist) < 6:
                features['enemyDistance'] = min(oppDist) - 6

        # if there is an invader
        oppPacPos = [gameState.getAgentState(x).getPosition() for x in opponents if gameState.getAgentState(x).isPacman]
        oppPacDist = [self.getMazeDistance(myState.getPosition(), x) for x in oppPacPos]
        if len(oppPacDist) > 0:
            features['invaderDistance'] = min(oppPacDist)

        # once we have food
        features['distanceToHome'] = 0
        if gameState.getAgentState(self.index).numCarrying >= 1:
            features['distanceToFood'] = 0
            features['distanceToHome'] = self.getMazeDistance(gameState.getAgentPosition(self.index), self.start)
        return features

    def getWeights(self, gameState):
        return {'successorScore': 100, 'distanceToFood': -1, 'enemyDistance': 5, 'distanceToHome': -1,
                'stop': -100, 'reverse': -50, 'invaderDistance': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def getFeatures(self, gameState):
        features = util.Counter()

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        action = myState.configuration.direction

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            if gameState.getAgentState(self.index).scaredTimer == 0:
                features['invaderDistance'] = min(dists)
            else:
                features['invaderDistance'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
