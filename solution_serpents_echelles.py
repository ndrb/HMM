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
    values = [0] * len(mdp.etats)
    i = 0
    while i < len(mdp.etats):
        cumulative_sum = 0
        j = 0
        while j < len(mdp.etats):
            cumulative_sum += mdp.modele_transition[i,plan[j]][0][1] * (mdp.recompenses[j] + mdp.escompte * values[j])
            j += 1
        values[i] = cumulative_sum
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
    plan_prime = [1] * len(mdp.etats)

    actions = [0,1,2]
    P = np.zeros( (len(mdp.etats), len(actions), len(mdp.etats)) )  # transition probability
    for etat in mdp.etats:
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

    values = [0] * len(mdp.etats)

    is_value_changed = True
    while is_value_changed:
        is_value_changed = False

        for s in range(len(mdp.etats)):
            cumulative_sum_test = 0
            for s1 in range(len(mdp.etats)):
                cumulative_sum_test += P[s, plan_prime[s], s1]  * (mdp.recompenses[s1] + mdp.escompte * values[s1])
            values[s] = cumulative_sum_test


        for s in range(len(mdp.etats)):
            q_best = values[s]
            for a in range(len(actions)):
                q_sa = sum([P[s, a, s1] * (mdp.recompenses[s1] + mdp.escompte * values[s1]) for s1 in range(len(mdp.etats))])
                if q_sa > q_best:
                    plan_prime[s] = a
                    q_best = q_sa
                    is_value_changed = True

    #return plan
    plan = dict([(s, ' ') for s in mdp.etats])
    i = 0
    while i < len(plan_prime):
        plan[i] = plan_prime[i]
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
    return plan

#####
# iteration_politiques: Algorithme d'itération par politiques, qui retourne le plan optimal et sa valeur.
#
# plan_initial: Le plan à utiliser pour initialiser l'algorithme d'itération par politiques.
#
# retour: Un tuple contenant le plan optimal et son tableau de valeurs.
### 
def iteration_politiques(mdp, plan_initial):
    values = calcul_valeur(mdp,plan_initial)
    plan = calcul_plan(mdp, values)
    return plan, values
