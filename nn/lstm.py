import numpy as np

class LSTMLayer:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # weights for input modulation gate
        self.W_i = np.random.randn(input_dim, hidden_dim)
        self.U_i = np.random.randn(hidden_dim, hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        
        # weights for forget gate
        self.W_f = np.random.randn(input_dim, hidden_dim)
        self.U_f = np.random.randn(hidden_dim, hidden_dim)
        self.b_f = np.zeros(hidden_dim)
        
        # weights for output modulation gate
        self.W_o = np.random.randn(input_dim, hidden_dim)
        self.U_o = np.random.randn(hidden_dim, hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        
        # weights for cell state modulation
        self.W_c = np.random.randn(input_dim, hidden_dim)
        self.U_c = np.random.randn(hidden_dim, hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        
        # initialize cell state and hidden state
        self.c = np.zeros(hidden_dim)
        self.h = np.zeros(hidden_dim)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x):
        # input modulation gate
        i = self.sigmoid(x.dot(self.W_i) + self.h.dot(self.U_i) + self.b_i)
        
        # forget gate
        f = self.sigmoid(x.dot(self.W_f) + self.h.dot(self.U_f) + self.b_f)
        
        # output modulation gate
        o = self.sigmoid(x.dot(self.W_o) + self.h.dot(self.U_o) + self.b_o)
        
        # cell state modulation
        c_tilda = self.tanh(x.dot(self.W_c) + self.h.dot(self.U_c) + self.b_c)
        
        # update cell state
        self.c = f * self.c + i * c_tilda
        
        # update hidden state
        self.h = o * self.tanh(self.c)
        
        return self.h, self.c
