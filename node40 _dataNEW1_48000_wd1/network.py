import numpy as np
from common.activations import softmax, sigmoid
from common.loss import cross_entropy_error
from collections import OrderedDict
from common.layers import SoftmaxWithLoss, Affine, ReLU
import pickle

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        #self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        #self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        #Heの初期値を使用
        self.params['W1'] = np.random.randn(input_size, hidden_size) * he(input_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * he(hidden_size)

        # レイヤの生成
        self.layers = OrderedDict() # 順番付きdict形式.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss() # 出力層
        
    def predict(self, x):
        """
        推論関数
        x : 入力
        """
        for layer in self.layers.values():
            # 入力されたxを更新していく = 順伝播計算
            x = layer.forward(x)
        
        return x
        
    def loss(self, x, t):
        """
        損失関数
        x:入力データ, t:教師データ
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        """
        識別精度
        """
        # 推論. 返り値は正規化されていない実数
        y = self.predict(x)
        #正規化されていない実数をもとに、最大値になるindexに変換する
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : 
            """
            one-hotベクトルの場合、教師データをindexに変換する
            """
            t = np.argmax(t, axis=1)
        
        # 精度
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        """
        全パラメータの勾配を計算
        """
        
        # 順伝播
        self.loss(x, t)

        # 逆伝播
        dout = 1 # クロスエントロピー誤差を用いる場合は使用されない
        dout = self.lastLayer.backward(dout=1) # 出力層
        
        ## doutを逆向きに伝える 
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # dW, dbをgradsにまとめる
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    
    def save_params(self, file_name="params.pkl"):
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
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        
    def load_params(self, file_name="params.pkl"):
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
    
    
    
    def he(n1):
    """
    Heの初期値を利用するための関数
    返り値は、見かけの標準偏差
    """    
    return np.sqrt(2/n1)
    
    