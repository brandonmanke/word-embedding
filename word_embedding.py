'''
Simple word embedding using Skip-Gram model
@author: Brandon Manke
'''
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    with open(path) as f:
        content = f.read().splitlines()
        return content

def unique_words(training_data):
    s = set()
    for sentence in training_data:
        for word in sentence.split():
            s.add(word)
    return list(s)

# normalizes K distinct values to which sum to 1
def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

# cost function, we want to minimize
def cost(activation_func):
        return -1 * np.sum(np.log(activation_func))

def train(training_data, win_size, e_dimension, alpha, epoch):
        vocab = unique_words(training_data)
        costs = []
        #epoch = 50
        U = np.random.randn(len(vocab), e_dimension) # U context words - NxD
        V = np.random.randn(len(vocab), e_dimension) # V center words - NxD
        while epoch > 0:
                U_grad = np.zeros([len(vocab), e_dimension]) # NxD
                V_grad = np.zeros([len(vocab), e_dimension]) # NxD
                J = 0
                for sentence in training_data:
                        sent_arr = sentence.split()
                        for i in range(len(sent_arr)):
                                word = sent_arr[i]
                                #Vc = V[i]

                                # get context words
                                lower_bound = max(i - win_size, 0)
                                upper_bound = min(i + win_size + 1, len(sent_arr))
                                ctx_words = sent_arr[lower_bound:upper_bound]
                                ctx_words.remove(word)

                                P = np.dot(U, V[i])
                                A = softmax(P)

                                s = 0 # sum
                                for j in range(len(vocab)):
                                        s = s + (A[j] * U[j])

                                for w in ctx_words:
                                        ctx_index = vocab.index(w)
                                        partial_derivative = V[i] * A[ctx_index] - V[i]
                                        U_grad[ctx_index] = U_grad[ctx_index] + partial_derivative

                                        #(s - U[ctx_index])
                                        V_grad[i] = V_grad[i] + ((-1 * U[ctx_index]) + s)

                                J = J + cost(A)
                                J /= len(vocab)

                U = U - (alpha * U_grad)
                V = V - (alpha * V_grad)
                epoch = epoch - 1
                print('epoch:', epoch, 'cost:', J)
                costs.append(J)

                if epoch is 120:
                        alpha /= 10

        plt.plot(costs)
        plt.show()
        return None

# 277 unique words in training.txt

# Given a center word we must predict the context words based off training data
if __name__ == '__main__':
    training_data = read_file('./training.txt')
    train(training_data, 2, 128, 0.001, 50)
