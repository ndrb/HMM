# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np

from importlib.machinery import SourceFileLoader


# Associe un entier à chaque lettre
INT2LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

# Associe une lettre à chaque entier
LETTERS2INT = {}
for i, l in enumerate(INT2LETTERS):
    LETTERS2INT[l] = i

NB_LETTERS = len(LETTERS2INT)

# Identifie les voisins de chaque lettre sur mon clavier mac --Hugo
KEYBOARD_NEIGHBORS = {'a': ['q', 'w', 's', 'z'],
                      'b': ['v', 'g', 'h', 'n'],
                      'c': ['x', 'd', 'f', 'v'],
                      'd': ['s', 'e', 'r', 'f', 'c', 'x'],
                      'e': ['w', 's', 'd', 'r'],
                      'f': ['d', 'r', 't', 'g', 'v', 'c'],
                      'g': ['f', 't', 'y', 'h', 'b', 'v'],
                      'h': ['g', 'y', 'u', 'j', 'n', 'b'],
                      'i': ['u', 'j', 'k', 'o'],
                      'j': ['h', 'u', 'i', 'k', 'm', 'n'],
                      'k': ['j', 'i', 'o', 'l', 'm'],
                      'l': ['k', 'o', 'p'],
                      'm': ['n', 'j', 'k'],
                      'n': ['b', 'h', 'j', 'm'],
                      'o': ['i', 'k', 'l', 'p'],
                      'p': ['o', 'l'],
                      'q': ['a', 'w'],
                      'r': ['e', 'd', 'f', 't'],
                      's': ['a', 'w', 'e', 'd', 'x', 'z'],
                      't': ['r', 'f', 'g', 'y'],
                      'u': ['y', 'h', 'j', 'i'],
                      'v': ['c', 'f', 'g', 'b'],
                      'w': ['q', 'a', 's', 'e'],
                      'x': ['z', 's', 'd', 'c'],
                      'y': ['t', 'g', 'h', 'u'],
                      'z': ['a', 's', 'x']}


def generate_model_observation(prob_correct=0.9):
    p_observation = np.zeros((NB_LETTERS, NB_LETTERS))
    for i in range(NB_LETTERS):
        l = INT2LETTERS[i]
        p_observation[i, i] = prob_correct
        neighbors = KEYBOARD_NEIGHBORS[l]
        for neighbor in neighbors:
            p_observation[LETTERS2INT[neighbor], i] = (
                1. - prob_correct) / len(neighbors)

    p_observation /= p_observation.sum(axis=0).reshape((1, -1))

    return p_observation


def Corrector_factory(corrector):
    if corrector.endswith('.py'):
        corrector = os.path.abspath(corrector)
        name = corrector.replace('/', '.').replace('.', '_')

        solution = SourceFileLoader(name, corrector).load_module(name)

        return solution.Correcteur

    return None


def correct_words(corrector, words):
    max_letters = max(list(map(len, words)))
    for word in words:
        corrected_word, prob = corrector.corrige(word)
        print((str.ljust(word, max_letters) + ' -> ' + str.ljust(corrected_word,
                                                                 max_letters) + ' p(mot_corrige,mot)={0}'.format(prob)))


def evaluation(letters_corrector, Corrector, validation_file):
    # Charger les mots, les mots corrigés et les probabilités de l'évaluation.
    with open(validation_file, 'rb') as f:
        words, corrected_words, probs = pickle.load(f)

    #mots = ['bonjojr', 'imtelligwmce', 'infprmqtiwue', 'poiwwon', 'trzvajl', 'pfpgrqmme']
    # reponses_attendues = [('bonjour', 1.20818392722e-09),
    #                      ('intelligence', 4.67443191136e-18),
    #                      ('informatique', 3.25462496877e-19),
    #                      ('poisson', 8.1786639759e-11),
    #                      ('travail', 1.72119029099e-11),
    #                      ('programme', 2.28282587957e-15)]

    reussi = True
    for word, true_corrected_word, true_prob in zip(words, corrected_words, probs):
        corrected_word, prob = letters_corrector.corrige(word)
        print(('Mot:', word))
        print(('Correction:', corrected_word + ', p(mot_corrige,mot)=' + str(prob)))
        print(('Solution attendue:', true_corrected_word +
               ', p(mot_corrige,mot)=' + str(true_prob)))

        if corrected_word != true_corrected_word or np.abs(prob - true_prob) / true_prob > 1e-6:
            print('MAUVAISE RÉPONSE')
            reussi = False

        print('')

    # Test sur exemple du cours
    p_observation_bits = np.array([[0.9, 0.2], [0.1, 0.8]])
    p_transition_bits = np.array([[0.3, 0.6], [0.7, 0.4]])
    p_init_bits = np.array([0.5, 0.5])
    bits_sequence = '0001'
    bits2int = {'0': 0, '1': 1}
    int2bits = ['0', '1']

    bits_corrector = Corrector(
        p_init_bits, p_transition_bits, p_observation_bits, int2bits, bits2int)
    corrected_word, prob = bits_corrector.corrige(bits_sequence)
    true_corrected_word = '0101'
    true_prob = 0.0190512

    print(('Sequence bits:', bits_sequence))
    print(('Correction:', corrected_word + ', p(mot_corrige,mot)=' + str(prob)))
    print(('Solution attendue:', true_corrected_word +
           ', p(mot_corrige,mot)=' + str(true_prob)))

    if corrected_word != true_corrected_word or np.abs(prob - true_prob) / true_prob > 1e-6:
        print('MAUVAISE RÉPONSE')
        reussi = False

    print('')

    if reussi:
        print('Bravo, toutes les solutions trouvées sont bonnes!')
    else:
        print('Les solutions trouvées ne sont pas toutes bonnes')


#####
# Execution en tant que script
###
DESCRIPTION = "Lancer l'auto-correcteur."


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('mots', action='store', type=str, nargs="*",
                   help="mots à corriger. " +
                        "Défaut: une liste prédéfinie de mots et sera comparée avec la réponse attendue. " +
                        "Une comparaison sera également faite avec la solution de l'exemple sur la " +
                        "séquence de bits '0001' vu dans le cours.")

    # Paramètres globaux
    p.add_argument('-correcteur', dest="corrector_file", action='store', type=str, required=False, metavar="FICHIER",
                   default="solution_correcteur.py",
                   help="solution à évaluer.")

    p.add_argument('-params', dest="params_file", action='store', type=str, required=False, default="params.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les probabilités initiales de l'état caché " +
                        "et le modèle de transitions.")

    p.add_argument('-valider', dest="validation_file", metavar="FICHIER", action='store', type=str, required=False,
                   default='correcteur_validation.pkl',
                   help="fichier permettant de valider votre correcteur.")

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    corrector_file = args.corrector_file
    words = args.mots
    params_file = args.params_file
    validation_file = args.validation_file

    if not os.path.isfile(corrector_file):
        parser.error(
            "-correcteur '{0}' doit être un fichier!".format(os.path.abspath(corrector_file)))

    if not os.path.isfile(params_file):
        parser.error(
            "-params '{0}' doit être un fichier!".format(os.path.abspath(params_file)))

    # Charger les probabilités initiales et le modèle de transition
    with open(params_file, 'rb') as f:
        p_init, p_transition = pickle.load(f, encoding='latin1')

    p_observation = generate_model_observation()
    Corrector = Corrector_factory(corrector_file)
    letters_corrector = Corrector(
        p_init, p_transition, p_observation, INT2LETTERS, LETTERS2INT)

    # Si la liste de mots à corriger est vide, on lance l'évaluation.
    if len(words) == 0:
        if not os.path.isfile(validation_file):
            parser.error(
                "-valider '{0}' doit être un fichier!".format(os.path.abspath(validation_file)))

        evaluation(letters_corrector, Corrector, validation_file)
        return

    # Convertit les majuscules en minuscules, s'il y a lieu.
    words = list(map(str.lower, words))
    correct_words(letters_corrector, words)


if __name__ == "__main__":
    main()
