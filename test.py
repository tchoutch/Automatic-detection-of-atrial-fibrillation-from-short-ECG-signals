from predict import predict_labels
from wettbewerb import load_references
import numpy as np
from datetime import datetime
import os
import sys

class Test():
    def __init__(self) -> None:
        pass

    def predToFile(self,file,prediction):
        for t in prediction:
            line = ' '.join(str(x) for x in t)
            file.write(line + ' ')
        file.write('\n')

    def make_test_dir(self,name):
        parent_dir = "Tests/" 
        path = os.path.join(parent_dir, name) 
        os.mkdir(path)
        return path

    
if __name__ == '__main__':
    t = Test()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M")
    path = t.make_test_dir(dt_string)
    f1 = open(os.path.join(path, "test.out"), 'w')
    sys.stdout = f1
    f2 = open(os.path.join(path, "errors.out"), 'w')
    sys.stderr = f2

    with open(os.path.join(path, 'result.txt'), 'w') as f:
        ecg_leads,ecg_labels,fs,ecg_names = load_references("test_examples/test_examples")
        f.write('Test 1: Test data is test examples - LSTM model - Multicalss\n')
        test1 = predict_labels(ecg_leads,300,ecg_names,'LSTM_v1',False)
        t.predToFile(f,test1)
        f.write('Test 1 finished\n')
        f.write('Test 2: Test data is test examples - LSTM model - Binary\n')
        test2 = predict_labels(ecg_leads,300,ecg_names,'LSTM_v1',True)
        t.predToFile(f,test2)
        f.write('Test 2 finished\n')
        f.write('Test 3: Test data is test examples - CNN model - Multiclass\n')
        test3 = predict_labels(ecg_leads,300,ecg_names,'CNN_v1',False)
        t.predToFile(f,test3)
        f.write('Test 3 finished\n')
        f.write('Test 4: Test data is test examples - CNN model - Binary\n')
        test4 = predict_labels(ecg_leads,300,ecg_names,'CNN_v1',True)
        t.predToFile(f,test4)
        f.write('Test 4 finished\n')

        ecg_leads = [np.zeros((200000,)),np.zeros((100,)),np.ones((18000,))]
        ecg_names = ['A','B','C']
        f.write('Test 5: Test data is random with different shapes - LSTM model - Multicalss\n')
        test5 = predict_labels(ecg_leads,300,ecg_names,'LSTM_v1',False)
        t.predToFile(f,test5)
        f.write('Test 5 finished\n')
        f.write('Test 6: Test data is random with different shapes - LSTM model - Binary\n')
        test6 = predict_labels(ecg_leads,300,ecg_names,'LSTM_v1',True)
        t.predToFile(f,test6)
        f.write('Test 6 finished\n')
        f.write('Test 7: Test data is random with different shapes - CNN model - Multiclass\n')
        test7 = predict_labels(ecg_leads,300,ecg_names,'CNN_v1',False)
        t.predToFile(f,test7)
        f.write('Test 7 finished\n')
        f.write('Test 8: Test data is random with different shapes - CNN model - Binary\n')
        test8 = predict_labels(ecg_leads,300,ecg_names,'CNN_v1',True)
        t.predToFile(f,test8)
        f.write('Test 8 finished\n')
        f.close()
    f1.close()
    f2.close()