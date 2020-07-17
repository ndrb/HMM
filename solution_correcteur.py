# -*- coding: utf-8 -*-

#####
# Nader Baydoun (20156885)
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


class Correcteur:
    def __init__(self, p_init, p_transition, p_observation, int2letters, letters2int):
        '''Correcteur de frappes dans un mot.

        Modèle de Markov caché (HMM) permettant de corriger les erreurs de frappes
        dans un mot. La correction est dictée par l'inférence de l'explication
        la plus pausible du modèle.

        Parameters
        ------------
        p_init : array-like shape (N,)
                 Probabilités initiales de l'état caché à la première position.

        p_transition : array-like shape (X,Y)
                       Modèle de transition.

        p_observation : array-like shape (X,Y)
                        Modèle d'observation.

        int2letters : list
                      Associe un entier (l'indice) à une lettre.

        letters2int : dict
                      Associe une lettre (clé) à un entier (valeur).
        '''
        self.p_init = p_init
        self.p_transition = p_transition
        self.p_observation = p_observation
        self.int2letters = int2letters
        self.letters2int = letters2int

    def corrige(self, mot):
        '''Corrige les frappes dans un mot.

        Retourne la correction du mot donné et la probabilité p(mot, mot corrigé).

        Parameters
        ------------
        mot : string
              Mot à corriger.

        Returns
        -----------
        mot_corrige : string
                      Le mot corrigé.

        prob : float
               Probabilité dans le HMM du mot observé et du mot corrigé.
               C'est-à-dire 'p(mot, mot_corrige)'.
        '''

        # Implémenter un correcteur de frappes dans un mot basé sur un HMM.
        #       Vous aurez besoin des variable suivantes :
        #       self.p_init
        #       self.p_transition
        #       self.p_observation
        #       self.int2letters
        #       self.letters2int

        if mot.isdigit():
            return self.corrige_binaire(mot)

        return self.correcter(mot)





    def correcter(self, mot):
        dim = np.zeros((26, len(mot)))

        # Init: alpha(i,1) = P(S1 = S1 | H1 = i) P(H1=i)
        i = 0
        while i < 26:
            dim[i][0] = self.p_observation[self.letters2int[mot[0]], i] * self.p_init[i]
            i += 1

        righteous = []

        k = 1
        while k < len(mot):
            i = 0
            while i < 26:
                # alpha(i,t+1) = P(S(t+1) | H(t+1)=i) max{P(H(t+1)|H(t)=j) alpha(j, t)}
                # calculate 26 values then choose max
                calcul = []
                t = 0
                while t < 26:
                    lol = k - 1
                    calcul.append(self.p_transition[i, t] * dim[t][lol])
                    t += 1
                dim[i][k] = self.p_observation[self.letters2int[mot[k]], i] * max(calcul)
                #For every cell in the column, we have 26 elements in calcul, we only want the position of max value
                righteous.append(self.get_position_of_max(calcul))
                #righteous is gonna have the position of the maximum value for every cell
                i += 1
            k += 1

        i = 0
        pos = -1
        maxi = -1
        while i < 26:
            if (dim[i][-1]) > maxi:
                maxi = (dim[i][-1])
                pos = i
            i += 1


        #print(righteous)

        righteous_two = np.array(righteous)
        righteous_two = righteous_two.reshape(len(mot)-1,26)

        #print(righteous_two)

        #using the 'pos' I need to backtrack
        final_positions = []
        final_positions.append(pos)
        index = len(mot)-2
        dude_plz_work = pos
        while index >= 0:
            final_positions.append(righteous_two[index][dude_plz_work])
            dude_plz_work = righteous_two[index][dude_plz_work]
            index -=1


        dudeee = list(reversed(final_positions))
        real_final = []
        for x in dudeee:
            real_final.append(self.int2letters[x])

        mot_co = ""
        for x in real_final:
            mot_co += x

        # Retourne le mot sans correction avec une probabilité de 0.0 (.~= À MODIFIER =~.)
        return mot_co, maxi





    def corrige_binaire(self, mot):
        observation = [[0.9, 0.2],
                       [0.1, 0.8]]
        transition = [[0.3, 0.6],
                       [0.7, 0.4]]
        initial = [0.5, 0.5]

        dim = np.zeros((2, len(mot)))
        i = 0
        while i < 2:
            hldr = int(mot[0])
            dim[i][0] = observation[hldr][i] * initial[i]
            i += 1

        righteous = []

        k = 1
        while k < len(mot):
            i = 0
            while i < 2:
                calcul = []
                t = 0
                while t < 2:
                    lol = k - 1
                    calcul.append(transition[i][t] * dim[t][lol])
                    t += 1
                dim[i][k] = observation[int(mot[k])][i] * max(calcul)
                righteous.append(self.get_position_of_max(calcul))
                i += 1
            k += 1



        i = 0
        pos = -1
        maxi = -1
        while i < 2:
            if (dim[i][-1]) > maxi:
                maxi = (dim[i][-1])
                pos = i
            i += 1


        righteous_two = np.array(righteous)
        #rightous two has a huge array in it, and it contains the position of the max value for every cell
        righteous_two = righteous_two.reshape(len(mot)-1,2)

        #print(righteous_two)

        #using the 'pos' I need to backtrack
        final_positions = []
        final_positions.append(pos)
        index = len(mot)-2
        dude_plz_work = pos
        while index >= 0:
            final_positions.append(righteous_two[index][dude_plz_work])
            dude_plz_work = righteous_two[index][dude_plz_work]
            index -=1


        dudeee = list(reversed(final_positions))
        real_final = []
        for x in dudeee:
            real_final.append(self.int2letters[x])

        mot_co = ""
        for x in real_final:
            mot_co += x

        return mot_co, maxi

    def get_position_of_max(self, calcul):
        i = 0
        pos = -1
        maxi = -1
        while i < len(calcul):
            if calcul[i] > maxi:
                pos = i
                maxi = calcul[i]
            i += 1
        return pos
