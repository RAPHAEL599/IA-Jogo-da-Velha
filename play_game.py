import numpy as np
import tensorflow as tf
from keras import layers, models
import keras

# Função para inicializar o tabuleiro
def initialize_board():
    return np.zeros((3, 3))

# Função para verificar se um jogador ganhou
def check_winner(board, player):
    # Verifica linhas e colunas
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    # Verifica diagonais
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# Função para imprimir o tabuleiro
def print_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    for row in board:
        print(" | ".join(symbols[val] for val in row))
        print("---------")

# Função para jogar o jogo
def play_game(model):
    board = initialize_board()
    player = 1  # Jogador humano começa
    game_over = False
    
    while not game_over:
        if player == 1:
            print("Jogador humano (X), é sua vez.")
            print_board(board)
            
            # Captura da jogada do jogador humano
            while True:
                try:
                    move = int(input("Digite sua jogada (0-8): "))
                    if move < 0 or move > 8 or board[move // 3, move % 3] != 0:
                        print("Jogada inválida. Tente novamente.")
                    else:
                        break
                except ValueError:
                    print("Entrada inválida. Digite um número de 0 a 8.")

            board[move // 3, move % 3] = 1  # Marca a jogada do jogador humano (X)
        else:
            print("IA (O), é sua vez.")
            
            # Obter a jogada da IA
            move = get_next_move(model, board, player=-1)
            board[move // 3, move % 3] = -1  # Marca a jogada da IA (O)

        # Verifica se alguém ganhou
        if check_winner(board, player):
            print_board(board)
            if player == 1:
                print("Parabéns! Você ganhou!")
            else:
                print("A IA ganhou. Tente novamente!")
            game_over = True
        elif np.all(board != 0):  # Verifica se o tabuleiro está cheio (empate)
            print_board(board)
            print("Empate!")
            game_over = True

        # Alternar jogador
        player = -player

# Função para obter a próxima jogada da IA
def get_next_move(model, board, player):
    # Verificar se há uma jogada do jogador a ser bloqueada
    opponent = -player
    for i in range(3):
        # Verificar linhas
        if np.sum(board[i, :] == opponent) == 2 and np.sum(board[i, :] == 0) == 1:
            return np.argmax(board[i, :] == 0) + 3*i
        # Verificar colunas
        if np.sum(board[:, i] == opponent) == 2 and np.sum(board[:, i] == 0) == 1:
            return np.argmax(board[:, i] == 0)*3 + i
    # Verificar diagonal principal
    if np.sum(np.diag(board) == opponent) == 2 and np.sum(np.diag(board) == 0) == 1:
        return np.argmax(np.diag(board) == 0)*4
    # Verificar diagonal secundária
    if np.sum(np.diag(np.fliplr(board)) == opponent) == 2 and np.sum(np.diag(np.fliplr(board)) == 0) == 1:
        return np.argmax(np.diag(np.fliplr(board)) == 0)*2+2
    
    # Caso não haja jogada do jogador a ser bloqueada, faça uma jogada aleatória
    available_moves = np.where(board.flatten() == 0)[0]
    return np.random.choice(available_moves)

# Carregar o modelo treinado (substitua com o caminho correto do seu modelo)
try:
    model = keras.models.load_model('tic_tac_toe_model.h5')
    print("Modelo carregado com sucesso.")
except:
    print("Erro ao carregar o modelo. Certifique-se de que o modelo foi treinado.")

# Jogar o jogo com a IA
play_game(model)
