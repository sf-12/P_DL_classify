import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss
from common.gradient import numerical_gradient
from common.optimizer import RMSProp

def he(n1):
    """
    Heの初期値を利用するための関数
    返り値は、見かけの標準偏差
    """    
    return np.sqrt(2/n1)


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01,weight_decay_lambda=0.01):
        """
        input_size : 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : 畳み込みの条件, dict形式　　例、{'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
        hidden_size : 隠れ層のノード数
        output_size : 出力層のノード数
        weight_init_std ：　重みWを初期化する際に用いる標準偏差
        """
        self.hidden_layer_num = 3
        self.weight_decay_lambda = weight_decay_lambda
        #filter_num = conv_param['filter_num']
        #filter_size = conv_param['filter_size']
        #filter_pad = conv_param['pad']
        #filter_stride = conv_param['stride']
        filter_num = 30
        filter_size = 5
        filter_pad = 0
        filter_stride = 1
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 重みの初期化
        self.params = {}
        std = weight_init_std
        self.params['W1'] = std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) # W1は畳み込みフィルターの重みになる
        self.params['b1'] = np.zeros(filter_num) #b1は畳み込みフィルターのバイアスになる
        #self.params['W2'] = std *  np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        #self.params['W3'] = std *  np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        #Heの初期値を使用
        self.params['W2'] = np.random.randn(pool_output_size, hidden_size) * he(pool_output_size)
        self.params['W3'] = np.random.randn(hidden_size, output_size) * he(hidden_size)
        
        

        # レイヤの生成
        self.layers = OrderedDict()
        #self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
        #                                   conv_param['stride'], conv_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           1, 0) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)

        # 荷重減衰を考慮した損失を求める
        lmd = self.weight_decay_lambda        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 1):
            W = self.params['W' + str(idx)]
            
            # 全ての行列Wについて、1/2* lambda * Σwij^2を求め、積算していく
            weight_decay += 0.5 * lmd * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay
        

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ

    減衰を考慮した損失を求める
        lmd = self.weight_decay_lambda        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            
            # 全ての行列Wについて、1/2* lambda * Σwij^2を求め、積算していく
            weight_decay += 0.5 * lmd * np.sum(W**2)

        return self.lastLayer.forward(y, t) + weight_decay

        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        # 荷重減衰を考慮しながら、dW, dbをgradsにまとめる
        lmd = self.weight_decay_lambda
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW + lmd * self.layers['Affine1'].W, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW + lmd * self.layers['Affine2'].W, self.layers['Affine2'].db

        return grads
    
    def save_params(self, file_name="CNNparams.pkl"):
        
        params = {}
        #for key, val in self.params.items():
        #    params[key] = val

        print("W1Start")
        params['W1'] = self.params['W1']
        print("b1Start")
        params['b1'] = self.params['b1']
        print("W2Start")
        params['W2'] = self.params['W2'] 
        print("b2Start")
        params['b2'] = self.params['b2']
        print("W3Start")
        params['W3'] = self.params['W3'] 
        print("b3Start")
        params['b3'] = self.params['b3']
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        
    def load_params(self, file_name="CNNparams.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        #for key, val in params.items():
        #    self.params[key] = val

        #for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
        #    self.layers[layer_idx].W = self.params['W' + str(i+1)]
        #    self.layers[layer_idx].b = self.params['b' + str(i+1)]
        self.params['W1'] = params['W1']
        self.params['b1'] = params['b1']
        self.params['W2'] = params['W2']
        self.params['b2'] = params['b2']
        self.params['W3'] = params['W3']
        self.params['b3'] = params['b3']

    def make_layers(self):
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           1, 0) # W1が畳み込みフィルタの重み, b1が畳み込みフィルタのバイアスになる
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
    