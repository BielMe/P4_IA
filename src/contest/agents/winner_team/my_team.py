# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point




#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}



class OffensiveReflexAgent(CaptureAgent):
    """
    Un agente reflexivo que busca comida, evita defensores y prioriza regresar la comida a su base.
    """

    def __init__(self, index):
        super().__init__(index)
        self.start = None
        self.last_positions = []  # Para rastrear movimientos repetitivos

    def register_initial_state(self, game_state):
        """
        Se llama al inicio del juego para inicializar el estado del agente.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    def powerCheck(self, gameState):
        myState = gameState.get_agent_astate(self.index)
        enemies = [gameState.getAgentState(i) for i in self.get_opponents(gameState)]

        minGhostScare = min(enemy.scaredTimer for enemy in enemies)
        if minGhostScare < 15:
            self.powerMode = False
        else:
            self.powerMode = True
            self.survivalMode = False
            self.survivalPoint = self.start
    def gotocap(self,gameState):

        actions = gameState.getLegalActions(self.index)
        bestDist = 9999
        bestAction = None
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostPos = [a.getPosition() for a in enemies if not a.isPacman and a.get_position() != None and a.scaredTimer == 0]
        for action in actions[:]:
            if self.checkEmptyPath(gameState,action,20):
                actions.remove(action)

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            # if capsules dissappear, we got it!!
            if self.getCapsules(successor) != self.getCapsules(gameState):
                return action
            pos2 = successor.getAgentPosition(self.index)
            capsules = self.getCapsules(successor)
            minDis = 9999
            minCap = (1, 1)
            for cap in capsules:
                dist = self.getMazeDistance(pos2, cap)
                if dist < minDis:
                    minDis = dist
                    minCap = cap 

            dist = self.getMazeDistance(pos2, minCap)
            if len(ghostPos) > 0:
                if pos2 in ghostPos or pos2 == self.start:
                    dist = dist + 99999999
                elif min([self.getMazeDistance(pos2,gp) for gp in ghostPos]) <2 :
                    dist = dist + 99999999
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction
       
        
    def adaptiveAction(self, gameState):
        
        # Evaluar la comida más cercana y su densidad
        food_density = self.getFoodDensity(gameState)
        best_food = min(food_density, key=food_density.get)  # Comida más densa
        
        # Evaluar la proximidad de los fantasmas
        ghost_distances = self.getGhostDistances(gameState)
        close_ghosts = [ghost for ghost in ghost_distances if ghost[1] < 5]
        
        if close_ghosts:
            # Si hay fantasmas cercanos, priorizar la evasión
            action = self.avoidGhosts(gameState, best_food)
        else:
            # Si no hay fantasmas, ir por la comida
            action = self.moveToFood(gameState, best_food)
        
        return action


    def choose_action(self, game_state):
        """
        Elige una acción basada en la mayor puntuación de evaluación.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, action) for action in actions]  # Llamamos a evaluate
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        chosen_action = random.choice(best_actions)

        # Rastrear posiciones para evitar oscilaciones
        self.track_position(game_state, chosen_action)

        return chosen_action

    def evaluate(self, game_state, action):
        """
        Calcula una combinación lineal de características y pesos.
        Esta función debe ser definida en el agente.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        
        # Multiplicamos las características por los pesos
        evaluation = sum(features[key] * weights.get(key, 0) for key in features)
        return evaluation
    def checkDeadEnd(self, gameState, action, depth):
        if depth == 0:
            return False
        successor = gameState.generate_successor(self.index, action)
        actions = successor.get_legal_actions(self.index)
        actions.remove(Directions.STOP)
        curDirct = successor.getAgentState(self.index).configuration.direction
        revDirct = Directions.REVERSE[curDirct]
        if revDirct in actions:
            actions.remove(revDirct)

        if len(actions) == 0:
            return True
        for action in actions:
            if not self.checkDeadEnd(successor, action, depth-1):
                return False
        return True
        
    

    def get_features(self, game_state, action):
        """
        Devuelve un diccionario de características para el par estado-acción.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # Posición actual
        my_pos = successor.get_agent_state(self.index).get_position()

        # Características relacionadas con la comida
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # Puntuación negativa para la comida restante

        if len(food_list) > 0:  # Distancia a la comida más cercana
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Características relacionadas con los defensores (fantasmas)
        defenders = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in defenders if not a.is_pacman and a.get_position() is not None]

        if len(ghosts) > 0:  # Penaliza estar cerca de defensores
            dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_defender_distance = min(dists)
            features['distance_to_defender'] = min_defender_distance

            # Penaliza si estamos demasiado cerca de los fantasmas
            if min_defender_distance < 2:
                features['too_close_to_ghost'] = 1  # Muy cerca de un fantasma
            elif min_defender_distance < 4:
                features['too_close_to_ghost'] = 0.5  # Distancia moderada, pero aún peligroso
        else:
            features['distance_to_defender'] = 10  # Sin defensores visibles, es seguro

        # Comprobar si el agente lleva comida y si está cerca de casa
        carrying_food = successor.get_agent_state(self.index).num_carrying
        features['carrying_food'] = carrying_food

        if carrying_food > 0:  # Fomentar regresar a casa con la comida
            home_distance = self.get_distance_to_home(successor, my_pos)
            features['distance_to_home'] = home_distance

            # Penaliza fuertemente si no nos movemos hacia casa
            if home_distance > 0:
                features['not_moving_home'] = 1

        # Penaliza la oscilación si se está visitando la misma posición
        if my_pos in self.last_positions:
            features['revisit_penalty'] = 1

        return features
    
    def get_weights(self, game_state, action):
        """
        Asigna pesos a las características de acuerdo al estado actual del agente:
        - Sin comida: buscar comida (evitar fantasmas).
        - Con comida: regresar a casa, pero recoger comida muy cercana en el camino.
        """
        carrying_food = game_state.get_agent_state(self.index).num_carrying
        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_state(self.index).get_position()
        print(carrying_food)

        # 1. Sin comida (buscar comida)
        if carrying_food == 0:
            # Prioridades: Buscar comida sin acercarse mucho a los fantasmas
            return {
                'successor_score': 100,          # Alta prioridad por conseguir comida
                'distance_to_food': -10,         # Fuerte incentivo para acercarse a la comida
                'distance_to_defender': 10,      # Evitar defensores
                'too_close_to_ghost': -1000,     # Alta penalización por acercarse a los fantasmas
                'carrying_food': 0,              # No hay comida que cargar
                'distance_to_home': 0,           # Ignorar casa cuando no se lleva comida
                'not_moving_home': 0,            # Ignorar este caso cuando no llevamos comida
                'revisit_penalty': -200,         # Penaliza por revisitar posiciones anteriores
            }

        # 2. Si estamos cargando comida
        else:
            # Si hay comida MUY cerca (priorizar recogerla si está a un paso)
            very_close_food = [food for food in food_list if self.get_maze_distance(my_pos, food) <= 2]

            if very_close_food:
                return {
                    'successor_score': -100,          # Alta prioridad por comida cercana
                    'distance_to_food': -15,         # Incentivar acercarse a comida cercana
                    'distance_to_defender': 15,      # Evitar defensores
                    'too_close_to_ghost': -1000,     # Huir de fantasmas
                    'carrying_food': 200,            # Incentivar cargar comida
                    'distance_to_home': -100,        # Aún podemos recoger comida, pero seguimos priorizando ir a casa
                    'not_moving_home': -500,         # Penaliza si no vamos hacia casa
                    'revisit_penalty': -200,         # Penaliza revisitar posiciones
                }
            else:
                return {
                    'successor_score': -100,         # Prioridad baja para la comida (ya llevamos)
                    'distance_to_food': -100,        # Ignorar comida cuando ya la llevamos
                    'distance_to_defender': 10,      # Evitar defensores
                    'too_close_to_ghost': -1000,     # Huir de los fantasmas
                    'carrying_food': 200,            # Incentivar llevar comida
                    'distance_to_home': -300,        # Priorizar regresar a casa
                    'not_moving_home': -1000,        # Fuerte penalización si no vamos a casa
                    'revisit_penalty': -200,         # Penaliza revisitar posiciones
                }

    def get_successor(self, game_state, action):
        """
        Encuentra el siguiente sucesor (estado resultante de la acción).
        """
        successor = game_state.generate_successor(self.index, action)
        return successor

    def get_distance_to_home(self, game_state, position):
        """
        Calcula la distancia desde la posición del agente hasta el límite de la base.
        """
        boundaries = self.get_home_boundaries(game_state)
        return min([self.get_maze_distance(position, boundary) for boundary in boundaries])

    def get_home_boundaries(self, game_state):
        """
        Devuelve la lista de posiciones que representan el límite de la base.
        """
        layout_width = game_state.data.layout.width
        home_x = layout_width // 2 - 1 if self.red else layout_width // 2
        height = game_state.data.layout.height

        return [(home_x, y) for y in range(height) if not game_state.has_wall(home_x, y)]

    def track_position(self, game_state, action):
        """
        Rastrea las últimas posiciones para evitar oscilaciones.
        """
        pos = game_state.generate_successor(self.index, action).get_agent_state(self.index).get_position()
        self.last_positions.append(pos)

        # Mantiene solo las últimas 5 posiciones
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Un agente defensor diseñado para eliminar pacman enemigos.
    Su único objetivo es detectar y perseguir a los invasores hasta erradicarlos.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.initFood = self.getFoodYouAreDefending(gameState).asList()
        self.chaseDest = []
        height = (gameState.data.layout.height - 2) / 2
        i = 0

        # Calculate the central point of the map, and set it as the defender initial position
        if self.red:
            for central in range((gameState.data.layout.width - 2) / 2,0,-1):
                if not gameState.hasWall(central, height):
                    self.detectDest = [(central,height)]
                    i = i+1
                    if i == 2:
                        break

        else:
            for central in range((gameState.data.layout.width - 2) / 2 + 1 ,gameState.data.layout.width,1):
                if not gameState.hasWall(central, height):
                    self.detectDest = [(central,height)]
                    i = i+1
                    if i == 2:
                        break


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).

        This function considers the following cases: 
            1.  No invader, no lost food (initial stage).
                Go to the initial defender place.
            2.  Detect there is an invader.
                Go to the invader place to eat it.
            3.  No invader detected, detect lost food. 
                Go to the lost food position(updated regularly).
            4.  Find the invader on the way to the lost food position.
                Give the priority to chase invader.
            5.  Go to enemy territory to eat food during first 30 steps 
            of "scared" situation, otherwise stay in the initial defender 
            position to defend invaders. 

        """

        actions = gameState.get_legalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        currEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currInvader = [enemy for enemy in currEnemies if enemy.isPacman and enemy.getPosition() != None]
        foodList = self.get_food_you_are_defending(gameState).as_list()
        loc = gameState.get_agent_state(self.index).getPosition()


        if len(self.initFood) - len(foodList) > 0:
            eatenFood = list(set(self.initFood).difference(set(foodList)))
            self.initFood = foodList
            self.chaseDest = eatenFood
            agentState = gameState.get_agent(self.index)

        ''' Attack !!! '''
        if agentState.scaredTimer > 10: 
            attackValues = [self.evaluateAttack(gameState, a) for a in actions]
            maxAttackValue = max(attackValues)
            bestAttackActions = [a for a, v in zip(actions, attackValues) if v == maxAttackValue]
            return random.choice(bestAttackActions)

        elif agentState.scaredTimer <= 10:
            if len(currInvader) > 0:
                foodLeft = len(self.getFood(gameState).asList())
                self.chaseDest = []
        elif len(self.chaseDest)>0:
            self.chaseFood = self.AStar(gameState,self.chaseDest[0]) 
            if len(self.chaseFood)>0:        
                return self.chaseFood[0]
            if loc == self.chaseDest[0]:      
                self.chaseDest = []
    
        elif len(self.detectDest)>0:
            self.searchActions = self.AStar(gameState, self.detectDest[0])
            if len(self.searchActions)>0:       
                return self.searchActions[0]
        return random.choice(bestActions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Si estamos en defensa (y no somos pacman)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Encuentra los invasores (pacman enemigos)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Calcula la distancia al invasor más cercano
            invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_invader_distance = min(invader_distances)
            features['invader_distance'] = closest_invader_distance

            # Si estamos muy cerca del invasor (distancia 1), es hora de capturarlo
            if closest_invader_distance == 1:
                features['capture_invader'] = 100  # Alta recompensa por capturar un invader
            else:
                features['capture_invader'] = 0  # Si no está cerca, no hay recompensa

            # La prioridad es perseguir al invasor
            features['pursuit'] = -closest_invader_distance  # Penaliza más lejos del invasor

        else:
            # Si no hay invasores, la "distancia al invasor" es arbitraria
            features['invader_distance'] = 10
            features['pursuit'] = 0

        # Penaliza detenerse (debemos estar en movimiento)
        if action == Directions.STOP:
            features['stop'] = 1

        # Penaliza si nos damos la vuelta (reversa), no tiene sentido en este contexto
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Los pesos de las características. Se enfoca en la persecución de invaders.
        """
        return {
            'num_invaders': -1000,            # Penaliza si hay invasores no detectados
            'on_defense': 500,                # Alta prioridad en defender
            'invader_distance': -100,          # Mientras más cerca de un invasor, mejor
            'pursuit': -5,                    # Penaliza cuando estamos más lejos de un invasor
            'capture_invader': 1000,          # Recompensa por capturar un invader
            'stop': -100000,                     # Penaliza detenerse (no podemos quedarnos quietos)
            'reverse': -50                    # Penaliza dar marcha atrás
        }

    def get_successor(self, game_state, action):
        """
        Encuentra el sucesor (el estado resultante de la acción).
        """
        successor = game_state.generate_successor(self.index, action)
        return successor