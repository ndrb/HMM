# -*- coding: utf-8 -*-

#####
# Nader Baydoun (20156885)
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np

#################################
# Solution serpents et échelles #
#################################

# Serpants and ladder. At any state you have 3 possible actions. There are 20 states and a system of special states
# to represent chutes and ladders

# Class MDP est deja fourni et represente le MDP du jeu:
# etat: states S (int 0-19)
# actions: D,DD,1 same three actions for every state
# modele de transition: a dict with, the values are a tuple (s,a). The keys are a list of tuples that contain the next state and its probability
# So for modele transition[(6,'D')], the keys are [(3, 0.166), (8, 0.166), (9, 0.166), (10, 0.166), (12, 0.166), (18, 0.166)]
# Recompenses: a 1D table that returns the reward for every state

#####
# calcul_valeur: Fonction qui retourne le tableau de valeurs d'un plan (politique).
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# plan: Un plan donnant l'action associée à chaque état possible (dictionnaire).
#
# retour: Un tableau Numpy 1D de float donnant la valeur de chaque état du mdp, selon leur ordre dans mdp.etats.
### 
def calcul_valeur(mdp, plan):
    values = [1] * len(mdp.etats)
    real_actions = mdp.actions[0]

    i = 0
    while i < len(mdp.etats):
        cumulative_sum = 0

        # So for modele transition[(6,'D')], the keys are [(3, 0.166), (8, 0.166), (9, 0.166), (10, 0.166), (12, 0.166), (18, 0.166)]
        for move in real_actions:
            for element in mdp.modele_transition[(i,move)]:
                cumulative_sum += element[1]*values[i]

        values[i] = mdp.recompenses[i] + mdp.escompte * cumulative_sum
        i += 1

    return values


#####
# calcul_plan: Fonction qui retourne un plan à partir d'un tableau de valeurs.
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# valeur: Un tableau de valeurs pour chaque état (tableau Numpy 1D de float).
#
# retour: Un plan (dictionnaire) qui maximise la valeur future espérée, en fonction du tableau "valeur".
### 
def calcul_plan(mdp, valeur):
    is_value_changed = True
    iterations = 0
    real_actions = mdp.actions[0]

    plan = dict([(s, ' ') for s in mdp.etats])
    while is_value_changed:
        is_value_changed = False
        iterations += 1

        z = 0
        while z < len(mdp.etats):
            best = valeur[z]
            y = 0
            while y < len(real_actions):
                cumulative_sum = 0
                for element in mdp.modele_transition[(z, real_actions[y])]:
                    cumulative_sum += element[1]
                answer_calc = mdp.recompenses[z] + mdp.escompte * cumulative_sum
                if answer_calc > best:
                    plan[z] = y
                    best = answer_calc
                    is_value_changed = True
                y += 1
            z += 1
    return plan

#####
# iteration_politiques: Algorithme d'itération par politiques, qui retourne le plan optimal et sa valeur.
#
# plan_initial: Le plan à utiliser pour initialiser l'algorithme d'itération par politiques.
#
# retour: Un tuple contenant le plan optimal et son tableau de valeurs.
### 
def iteration_politiques(mdp,plan_initial):

    # plan = plan_initial/arbitraire
    # Step 1:
    plan = plan_initial

    # Etape pour convergence
    # Step 2: Calcule Valeur
    # Pour tous etat, on calcule la valeur du plan sur cette etat en resolvant le system de |S| equation et |S| inconnu
    # aka: V(plan, etat[i]) = R[s] + y * sum(P(s_prime | s, plan(s))) * V(plan,s_prime)
    values = calcul_valeur(mdp, plan)


    # Step 3: Calcule Plan
    # Pour tous etat si il existe une action a tell que:
    # (R(s) + y* sum( P(s_prime | s,a) V(plan,s_prime))) > V(plan,s) then loop!
    plan = calcul_plan(mdp, values)

    # BS START
    states = mdp.etats
    actions = [0,1,2]
    N_STATES = len(states)
    N_ACTIONS = len(actions)
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability

    for etat in states:
        for act in actions:
            holder_value = 0
            if act == 0:
                holder_value = '1'
            if act == 1:
                holder_value = 'D'
            if act == 2:
                holder_value = 'DD'
            lol = mdp.modele_transition[(etat,holder_value)]
            for x in lol:
                P[etat, act, x[0]] = x[1]

    #print(P)

    gamma = mdp.escompte

    # initialize policy and value arbitrarily
    policy = [0 for s in range(N_STATES)]
    V = np.zeros(N_STATES)

    is_value_changed = True
    iterations = 0
    while is_value_changed:
        is_value_changed = False
        iterations += 1
        # run value iteration for each state
        for s in range(N_STATES):
            V[s] = sum([P[s, policy[s], s1] * (mdp.recompenses[s1] + gamma * V[s1]) for s1 in range(N_STATES)])
            # print "Run for state", s

        for s in range(N_STATES):
            q_best = V[s]
            # print "State", s, "q_best", q_best
            for a in range(N_ACTIONS):
                q_sa = sum([P[s, a, s1] * (mdp.recompenses[s1] + gamma * V[s1]) for s1 in range(N_STATES)])
                if q_sa > q_best:
                    policy[s] = a
                    q_best = q_sa
                    is_value_changed = True
        # print "Policy now", policy
    # BS FINISH

    #return plan
    plan = dict([(s, ' ') for s in mdp.etats])
    i = 0
    while i < len(policy):
        plan[i] = policy[i]
        i += 1
    j = 0
    while j < len(plan):
        if plan[j] == 0:
            plan[j] = '1'
        if plan[j] == 1:
            plan[j] = 'D'
        if plan[j] == 2:
            plan[j] = 'DD'
        j += 1
    return plan, V
