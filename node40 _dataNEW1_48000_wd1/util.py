# 動作確認の目的でこのファイルを編集してもかまいません。
# ただし、識別精度算出ウェブサイトで計算する際は、
# このファイルが自動的に上書きされますので、このファイルの変更内容は反映されません。 

import numpy as np

def loaddata():
    """
    データを読む関数
    """
    test_data = np.load("../1_data/test_data.npy")
    return test_data

def accuracy(func_predict, test_data):
    """
    精度を計算する関数
    label_pred : numpy 1D array 
    """
    
    # データ数のチェック
    data_size = len(test_data)
    #print("len(test_data)=",len(test_data))
#    if data_size<14500:
#        error = "label_predのサイズが足りていません"
#        print(data_size, error)
#        print('Test loss:', error)
#        print('Test accuracy:', error)
#        return
#    elif data_size>14500:
#        error = "label_predのサイズが多すぎます"
#        print(data_size, error)
#        print('Test loss:', error)
#        print('Test accuracy:', error)        
#        return

    test_label = np.load("../1_data/test_label.npy")
    
    
    # 予測
    # サーバーへの負荷を軽減するため、ミニバッチ処理で行う
    
    batch_size = 3000
    
    minibatch_num = np.ceil( data_size / batch_size).astype(int) # ミニバッチの個数
    li_loss = []
    li_accuracy = []
    index = np.arange(data_size)
    
    for mn in range(minibatch_num):
        print("mn = ",mn)
        mask = index[batch_size*mn:batch_size*(mn+1)]        
        data = test_data[mask]
        label = test_label[mask]
        loss, accuracy  = func_predict(data, label)
        print(loss, accuracy)
        
        li_loss.append(loss)
        li_accuracy.append(accuracy)

    test_loss = np.mean(li_loss)
    test_accuracy = np.mean(li_accuracy)
    
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    return