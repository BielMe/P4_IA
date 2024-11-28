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
    A reflex agent that seeks food, avoids defenders, and returns food to its side.
    """

    def __init__(self, index):
        super().__init__(index)
        self.start = None
        self.last_positions = []  # To track repetitive movements

    def register_initial_state(self, game_state):
        """
        Called once at the beginning of the game to initialize.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Chooses an action based on the highest evaluation score.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        chosen_action = random.choice(best_actions)

        # Track positions to avoid oscillations
        self.track_position(game_state, chosen_action)

        return chosen_action

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a dictionary of features for the state-action pair.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # Current position
        my_pos = successor.get_agent_state(self.index).get_position()

        # Debug: Draw the agent's planned path
        self.debug_draw([my_pos], [0, 0, 1], clear=False)

        # Compute food-related features
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # Negative score for remaining food

        if len(food_list) > 0:  # Distance to nearest food
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Compute defender-related features
        defenders = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in defenders if not a.is_pacman and a.get_position() is not None]

        if len(ghosts) > 0:  # Penalize being close to defenders
            dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_defender_distance = min(dists)
            features['distance_to_defender'] = min_defender_distance

            # Penalize if too close to defenders
            if min_defender_distance < 3:
                features['too_close_to_ghost'] = 1
        else:
            features['distance_to_defender'] = 10  # No visible defenders = safe

        # Compute if agent is carrying food and close to home
        carrying_food = successor.get_agent_state(self.index).num_carrying
        features['carrying_food'] = carrying_food

        if carrying_food > 0:  # Encourage returning home with food
            home_distance = self.get_distance_to_home(successor, my_pos)
            features['distance_to_home'] = home_distance

            # Strong penalty if stuck near home and not crossing
            if home_distance < 3 and features['distance_to_defender'] < 3:
                features['stuck_near_home'] = 1

        # Penalize oscillation if position is being revisited
        if my_pos in self.last_positions:
            features['revisit_penalty'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Assigns weights to features.
        """
        return {
            'successor_score': 100,          # High priority for collecting food
            'distance_to_food': -1,         # Move closer to food
            'distance_to_defender': 10,     # Avoid defenders
            'too_close_to_ghost': -1000,    # Strongly avoid defenders close by
            'carrying_food': 200,           # High incentive for carrying food
            'distance_to_home': -50,        # Strongly encourage returning home with food
            'stuck_near_home': -1000,       # Avoid being stuck near home
            'revisit_penalty': -100         # Avoid revisiting positions
        }

    def get_successor(self, game_state, action):
        """
        Finds the next successor (resulting state from action).
        """
        successor = game_state.generate_successor(self.index, action)
        return successor

    def get_distance_to_home(self, game_state, position):
        """
        Computes the distance from the agent's position to the home boundary.
        """
        boundaries = self.get_home_boundaries(game_state)
        return min([self.get_maze_distance(position, boundary) for boundary in boundaries])

    def get_home_boundaries(self, game_state):
        """
        Returns the list of positions that represent the home boundary.
        """
        layout_width = game_state.data.layout.width
        home_x = layout_width // 2 - 1 if self.red else layout_width // 2
        height = game_state.data.layout.height

        return [(home_x, y) for y in range(height) if not game_state.has_wall(home_x, y)]

    def track_position(self, game_state, action):
        """
        Tracks the last few positions to avoid oscillations.
        """
        pos = game_state.generate_successor(self.index, action).get_agent_state(self.index).get_position()
        self.last_positions.append(pos)

        # Keep track of the last 5 positions onlyy
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
