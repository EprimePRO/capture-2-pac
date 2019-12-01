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
import math

import itertools

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
        self.depth = 6
        self.start = gameState.getAgentPosition(self.index)
        mazeWalls = gameState.getWalls()
        self.deadEnds = gameState.getWalls().deepCopy()
        self.floodFillDeadEnds(self.deadEnds)
        self.deadEndsList = self.deadEnds.asList()
        x = int(mazeWalls.width / 2)
        y = int(mazeWalls.height / 2)
        y = self.chooseYwithNoWall(mazeWalls, x, y)
        self.midMazePos = (x, y)
        CaptureAgent.registerInitialState(self, gameState)
        #self.debugDraw(self.deadEnds.asList(), [0,0,1], False)

    def floodFillDeadEnds(self, deadEnds):
        deadEndStack = util.Stack()
        for x in range(deadEnds.width):
            for y in range(deadEnds.height):
                if not deadEnds[x][y]:
                    numFreeNeighbors, neighbor = self.findFreeNeighbors(deadEnds, x, y)
                    if numFreeNeighbors == 1:
                        deadEndStack.push(((x, y), neighbor))

        while not deadEndStack.isEmpty():
            top = deadEndStack.pop()
            deadEnd = top[0]
            neighbor = top[1]
            deadEnds[deadEnd[0]][deadEnd[1]] = True
            numFreeNeighbors, newNeighbor = self.findFreeNeighbors(deadEnds, neighbor[0], neighbor[1])
            if numFreeNeighbors == 1:
                deadEndStack.push((neighbor, newNeighbor))

    def findFreeNeighbors(self, deadEnds, x, y):
        numFreeNeighbors = 0
        newX = newY = 0
        if y + 1 <= deadEnds.height - 1:
            if not deadEnds[x][y + 1]:
                numFreeNeighbors += 1
                newX = x
                newY = y + 1
        if y - 1 >= 0:
            if not deadEnds[x][y - 1]:
                numFreeNeighbors += 1
                newX = x
                newY = y - 1
        if x + 1 <= deadEnds.width - 1:
            if not deadEnds[x + 1][y]:
                numFreeNeighbors += 1
                newX = x +1
                newY = y
        if x - 1 >= 0:
            if not deadEnds[x - 1][y]:
                numFreeNeighbors += 1
                newX = x - 1
                newY = y
        return numFreeNeighbors, (newX, newY)

    def chooseYwithNoWall(self, mazeWalls, x, y):
        while mazeWalls[x][y]:
            y += 1
            if y >= mazeWalls.height - 1:
                y = 0
        return y

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
        if depth > self.depth:
            return self.evaluate(gameState), action
        actions = gameState.getLegalActions(self.index)
        actions = [x for x in actions if x != 'Stop']
        random.shuffle(actions)
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
        if depth > self.depth:
            return self.evaluate(gameState), action
        opponents = self.getOpponents(gameState)


        closestEnemy = None
        minDist = 60000
        for opp in opponents:
            tmpDist = self.getMazeDistance(gameState.getAgentPosition(opp), gameState.getAgentPosition(self.index))
            if tmpDist < minDist:
                minDist = tmpDist
                closestEnemy = opp
        actions = gameState.getLegalActions(closestEnemy)
        actions = [x for x in actions if x != 'Stop']


        '''opp0Actions = gameState.getLegalActions(opponents[0])
        opp0Actions = [x for x in opp0Actions if x != 'Stop']
        opp1Actions = gameState.getLegalActions(opponents[1])
        opp1Actions = [x for x in opp1Actions if x != 'Stop']
        actions = [(a,b) for a in opp0Actions for b in opp1Actions]'''


        for act in actions:
            # succtmp = gameState.generateSuccessor(opponents[0], act[0])
            #succ = succtmp.generateSuccessor(opponents[1], act[1])
            succ = gameState.generateSuccessor(closestEnemy, act)
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
        features['successorScore'] = self.getScore(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # get closest enemy distance
        opponents = self.getOpponents(gameState)
        oppPos = [gameState.getAgentState(x).getPosition() for x in opponents]
        oppDist = [self.getMazeDistance(myPos, x) for x in oppPos]
        # features['enemyDistance'] = min(oppDist)

        # Compute distance to the nearest food
        features['distanceToFood'] = 0
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            safeFood = [self.getMazeDistance(myPos, food) for food in foodList if food not in self.deadEndsList]
            if len(safeFood)>0:
                minDistance = min(safeFood)
                features['distanceToFood'] = minDistance
            else:
                features['distanceToFood'] = self.getMazeDistance(foodList[0], myPos)


        '''
        # if there is an invader
        oppPacPos = [gameState.getAgentState(x).getPosition() for x in opponents if gameState.getAgentState(x).isPacman]
        oppPacDist = [self.getMazeDistance(myState.getPosition(), x) for x in oppPacPos]
        if len(oppPacDist) > 0:
            features['distanceToFood'] = features['distanceToFood']/4
            features['invaderDistance'] = min(oppPacDist)
        '''
        # once we have food
        features['distanceToHome'] = 0
        features['foodCarrying'] = myState.numCarrying
        if myState.numCarrying >= 1:
            features['distanceToHome'] = self.getMazeDistance(myPos, self.midMazePos)

        # try not to go into dead ends
        if self.deadEnds[int(myPos[0])][int(myPos[1])] and min(oppDist)<16:
            features['deadEnd'] = 1
        return features

    def getWeights(self, gameState):
        return {'successorScore': 1000, 'distanceToFood': -4, 'distanceToHome': -20,
                'foodCarrying': 500, 'deadEnd': -550}


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

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        foodLeft = self.getFoodYouAreDefending(gameState).asList()
        features['foodLeft'] = len(foodLeft)

        foodList = self.getFoodYouAreDefending(gameState).asList()
        features['distanceToFood'] = 0
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        enemies = [gameState.getAgentState(i) for i in
                   self.getOpponents(gameState)]  # Computes distance to invaders we can see
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        '''
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            if gameState.getAgentState(self.index).scaredTimer == 0:
                features['invaderDistance'] = min(dists)
            else:
                features['invaderDistance'] = 0
        '''
        if len(invaders) == 0:
            features['defenseDist'] = self.getMazeDistance(self.midMazePos, myPos)
        else:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        return features

    def getWeights(self, gameState):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -5, 'stop': -100, 'reverse': -2,
                'defenseDist': -10, 'distanceToFood': -1, 'foodLeft': 1}
