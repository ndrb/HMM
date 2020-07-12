# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np

from importlib.machinery import SourceFileLoader


class MDP:

    def __init__(self, sol, etats, actions, modele_transition, recompenses, escompte):
        self.sol = sol
        self.etats = etats                              # liste des états
        # dict etat => tuple d'actions (str)
        self.actions = actions
        # dict (etat,action) => liste de paires (etat suivant, probabilité)
        self.modele_transition = modele_transition
        self.recompenses = recompenses                  # ndarray des récompenses
        # facteur d'escompte (float)
        self.escompte = escompte
        self.plan = dict([(s, ' ') for s in etats])

    def calcul_valeur(self, plan):
        return self.sol.calcul_valeur(self, plan)

    def calcul_plan(self, valeur):
        return self.sol.calcul_plan(self, valeur)

    def assigner_plan(self, plan):
        self.plan = plan

    def iteration_politiques(self, plan_initial):
        return self.sol.iteration_politiques(self, plan_initial)


class SerpentsEchelles(MDP):

    def __init__(self, sol, passages_serpents_echelles, recompenses, escompte):

        # Construire MDP à partir des serpents et echelles
        self.passages_serpents_echelles = passages_serpents_echelles

        # Contient 20 cases
        etats = list(range(20))

        # 3 actions: '1', 'D', 'DD'
        actions = dict([(s, ('1', 'D', 'DD')) for s in etats])

        # Calcul du modèle de transition
        modele_transition = {}
        for s in etats:
            for a in actions[s]:
                # Définir la liste (redondante) des états suivants
                # sans tenir compte des règles
                if s == 19:
                    etats_suivants = [s]
                else:
                    if a == '1':
                        etats_suivants = [s + 1]
                    elif a == 'D':
                        etats_suivants = [s + i for i in range(1, 7)]
                    elif a == 'DD':
                        D_etats_suivants = [s + i for i in range(1, 7)]
                        etats_suivants = []

                        for s_new in D_etats_suivants:
                            etats_suivants += [s_new + i for i in range(1, 7)]

                # Tenir compte des règles et accumuler
                # les probabilités
                transition_pour_s = {}
                n_etats_suivants = len(etats_suivants)
                for s_new in etats_suivants:
                    # Appliquer les regles
                    if s_new >= 20:
                        s_new = s    # ne bouge pas si dépasse la case 19
                    else:
                        # tenir compte des serpents et des echelles
                        s_new = self.appliquer_serpents_echelles(s_new)

                    # Cumuler les probabilités
                    if s_new in transition_pour_s:
                        p = transition_pour_s[s_new]
                        transition_pour_s[s_new] = p + 1. / n_etats_suivants
                    else:
                        transition_pour_s[s_new] = 1. / n_etats_suivants

                modele_transition[(s, a)] = list(transition_pour_s.items())

        MDP.__init__(self, sol, etats, actions,
                     modele_transition, recompenses, escompte)

    def appliquer_serpents_echelles(self, s):
        while s in self.passages_serpents_echelles:
            s = self.passages_serpents_echelles[s]
        return s

    def __str__(self):
        line = '-------' * 5 + '-''\n'
        for s in self.etats:
            line += '|'
            token = str(s)

            if s in self.passages_serpents_echelles:
                token += '=>' + str(self.passages_serpents_echelles[s])

            token += ' ' * (6 - len(token))
            line += token

            if (s + 1) % 5 == 0:
                line += '|\n'

                for s in range(s - 4, s + 1):
                    line += '|  '
                    token = self.plan[s]
                    line += token + ' ' * (4 - len(token))

                line += '|\n'
                line += '|______' * 5 + '|\n'

        return line


# Pour comparer avec la solution attendue
def compare_plans(plan_1, plan_2, cases_sans_action):
    for s1, a1 in list(plan_1.items()):
        if s1 not in cases_sans_action:
            if s1 not in plan_2 or plan_2[s1] != a1:
                return False

    return True


def test_mdp(solution, validation_file, escompte=0.9):

    passages_serpents_echelles = {1: 4, 7: 3, 11: 13, 13: 18, 14: 11}
    cases_sans_action = list(passages_serpents_echelles.keys())
    recompenses = np.zeros((20,))
    recompenses[-1] = 1

    solution = os.path.abspath(solution)
    name = solution.replace('/', '.').replace('.', '_')
    sol = SourceFileLoader(name, solution).load_module(name)

    mdp = SerpentsEchelles(
        sol, passages_serpents_echelles, recompenses, escompte)

    # Charger solution attendue
    with open(validation_file, 'rb') as f:
        plan_test_cmp, valeur_test_cmp, plan_cmp, valeur_cmp = pickle.load(
            f, encoding='latin1')

    # Test calcul_valeur
    print('Test de la fonction calcul_valeur')
    plan = dict([(s, mdp.actions[s][s % 3]) for s in mdp.etats])
    valeur_test = mdp.calcul_valeur(plan)

    if np.sum(np.abs(valeur_test - valeur_test_cmp)) > 1e-8:
        print('* Erreur *')
        print('La fonction a retourné:')
        print(valeur_test)
        print('La solution attendue est:')
        print(valeur_test_cmp)
    else:
        print('* Succès ! *')
        print(valeur_test_cmp)

    print('')

    # Test calcul_plan
    print('Test de la fonction calcul_plan')
    plan_test = mdp.calcul_plan(valeur_test_cmp)

    if not compare_plans(plan_test_cmp, plan_test, cases_sans_action):
        print('* Erreur *')
        print('La fonction a retourné:')
        print(plan_test)
        print('La solution attendue est:')
        print(plan_test_cmp)
    else:
        print('* Succès ! *')

    print('')

    # Test iteration_politiques
    print('Test de la fonction iteration_politiques')
    plan = dict([(s, '1') for s in range(20)])
    plan, valeur = mdp.iteration_politiques(plan)

    print('L\'algorithme retourne le plan (politique) suivant')
    mdp.assigner_plan(plan)
    print(mdp)

    if not compare_plans(plan, plan_cmp, cases_sans_action):
        print('* Erreur *')
        print("Ceci n'est pas le bon plan (politique)")
    else:
        print('* Succès ! *')
        print('Ceci est le bon plan (politique)')
        print('')
        print('L\'algorithme calcule la valeur suivante à ce plan')
        print(valeur)

        if np.sum(np.abs(valeur - valeur_cmp)) > 1e-8:
            print('* Erreur *')
            print("Ceci n'est pas la bonne valeur pour le plan optimal")
        else:
            print('* Succès ! *')
            print('Ceci est la bonne valeur pour le plan optimal')

    print('')


#####
# Execution en tant que script
###
DESCRIPTION = "Lancer le jeu de serpents-échelles."


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paramètres globaux
    p.add_argument('-solution', dest="solution", action='store', type=str, required=False, metavar="FICHIER",
                   default="solution_serpents_echelles.py",
                   help="solution à évaluer.")

    p.add_argument('-valider', dest="validation_file", action='store', type=str, required=False, metavar="FICHIER",
                   default="serpents_echelles_validation.pkl",
                   help="fichier pickle contenant le résultat attendu.")

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    solution = args.solution
    validation_file = args.validation_file

    if not os.path.isfile(solution):
        parser.error(
            "-solution '{0}' doit être un fichier!".format(os.path.abspath(solution)))

    if not os.path.isfile(validation_file):
        parser.error(
            "-valider '{0}'  doit être un fichier!".format(os.path.abspath(validation_file)))

    test_mdp(solution, validation_file)


if __name__ == "__main__":
    main()
