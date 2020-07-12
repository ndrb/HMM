# -*- coding: utf-8 -*-

#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
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

        #print(self.p_init)
        #print(self.p_observation)
        #print("KMSKMDKNFNSDJIFBJIDSBFHDS \n \n \n \n \n")
        #print(self.p_transition)
        #print("KMSKMDKNFNSDJIFBJIDSBFHDS \n \n \n \n \n")

        if mot.isdigit():
            return self.corrige_binaire(mot)

        dim = np.zeros((26, len(mot)))

        # Init: alpha(i,1) = P(S1 = S1 | H1 = i) P(H1=i)
        i = 0
        while i < 26:
            dim[i][0] = self.p_observation[self.letters2int[mot[0]], i] * self.p_init[i]
            #print(self.p_observation[self.letters2int[mot[0]], i] * self.p_init[i])
            i += 1


        k = 1
        while k < len(mot):
            i = 0
            while i < 26:
                # alpha(i,t+1) = P(S(t+1) | H(t+1)=i) max{P(H(t+1)|H(t)=j) alpha(j, t),,,,,}
                # alpha(0,2) = P(S2 = 0 | H2 = 0) max{P(H2=0|H1=0) alpha(0,1), P(H2=0|h1=1 alpha(1,1))}
                # calculate 26 values then choose max
                calcul = []
                t = 0
                while t < 26:
                    lol = k-1
                    calcul.append( self.p_transition[ i, t] * dim[t][lol] )
                    #print(self.p_transition[ self.letters2int[mot[k]] , self.letters2int[mot[lol]] ])
                    #print(dim[t][lol])
                    t += 1
                #print(calcul)
                dim[i][k] = self.p_observation[self.letters2int[mot[k]], i] * max(calcul)
                i += 1
            k += 1


        #print(dim)
        maxi = 0
        i = 0
        while i < 26:
            if (dim[i][-1]) > maxi:
                maxi = (dim[i][-1])
            i += 1

        # Retourne le mot sans correction avec une probabilité de 0.0 (.~= À MODIFIER =~.)
        return mot, maxi

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
                    #print(transition[i][t])
                    #print(dim[t][lol])
                    calcul.append(transition[i][t] * dim[t][lol])
                    t += 1
                dim[i][k] = observation[int(mot[k])][i] * max(calcul)
                righteous.append(self.get_position_of_max(calcul))
                i += 1
            k += 1

        #print(dim)
        #print(righteous)
        righteous_two = np.array(righteous)
        righteous_two = righteous_two.reshape(len(mot)-1,2)
        #print(righteous_two)
        righteous_final = []
        for x in righteous_two:
            righteous_final.append(max(x))


        i = 0
        pos = -1
        maxi = -1
        while i < 2:
            if (dim[i][-1]) > maxi:
                maxi = (dim[i][-1])
                pos = i
            i += 1
        righteous_final.append(pos)
        #print(righteous_final)
        mot_co = ""
        for x in righteous_final:
            mot_co += str(x)
        return mot_co, maxi

    def get_position_of_max(selfself, calcul):
        i = 0
        pos = -1
        maxi = -1
        while i < len(calcul):
            if calcul[i] > maxi:
                pos = i
                maxi = calcul[i]
            i += 1
        return pos
