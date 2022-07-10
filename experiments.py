from train import Model,Evaluation, Strategy,ModelType,Classification,LSTMModel,CNNModel,preprocessData
from datetime import datetime
import os
import sys

class Experiment():
  def __init__(self):
    pass

  def experiment(self):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M")
    if not os.path.exists("Experiments"):
        os.mkdir("Experiments")
    path = os.path.join("Experiments/", dt_string)
    os.mkdir(path)
    f = open(os.path.join(path, "experiment.out"), 'w')
    sys.stdout = f
    f2 = open(os.path.join(path, "errors.out"), 'w')
    sys.stderr = f2

    epochs = 75
    batch_size = 256
    strategy = Strategy.OVERSAMPLE
    over_sampling_factor = 0.2
    under_sampling_factor = 1
    print ("1. Experiment: LSTM - MULTICLASS - OVERSAMPLING")
    model= ModelType.LSTM
    mode= Classification.MULTICLASS
    print ("PLAY WITH EXTENSION" )
    for dim in [6000,9000,12000,15000,18000]:
      pd = preprocessData(mode)
      filtered_data = pd.resizeData(length = dim)
      balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
      X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
      model = LSTMModel(dimension,mode)
      history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size,verbose=0)
      y_pred = model.predict(X_test, batch_size)
      class_pred = model.predictitionToClass(y_pred) 
      ev = Evaluation(mode)
      print(f"Dimension = {dim}")
      print(ev.computeF1(class_pred,y_test))
      print(ev.computeF1(class_pred,y_test,"macro"))
      print(ev.computeF1(class_pred,y_test,"micro"))
    print("1. Experiment: FINISHED")
    print ("2. Experiment: LSTM - MULTICLASS- OVERSAMPLING")
    model= ModelType.LSTM
    mode= Classification.MULTICLASS
    print ("WITHOUT EXTENSION" )
    dim = 9000
    pd = preprocessData(mode)
    filtered_data = pd.filterData(dim)
    balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
    X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
    model = LSTMModel(dimension,mode)
    history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size)
    y_pred = model.predict(X_test, batch_size)
    class_pred = model.predictitionToClass(y_pred) 
    ev = Evaluation(mode)
    print(f"Dimension = {dim}")
    print(ev.computeF1(class_pred,y_test))
    print(ev.computeF1(class_pred,y_test,"macro"))
    print(ev.computeF1(class_pred,y_test,"micro"))
    print("2. Experiment: FINISHED")
    print ("3. Experiment: LSTM - BINARY - OVERSAMPLING")
    model= ModelType.LSTM
    mode= Classification.BINARY
    print ("PLAY WITH EXTENSION" )
    for dim in [6000,9000,12000,15000,18000]:
      pd = preprocessData(mode)
      filtered_data = pd.resizeData(length = dim)
      balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
      X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
      model = LSTMModel(dimension,mode)
      history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size,verbose=0)
      y_pred = model.predict(X_test, batch_size)
      class_pred = model.predictitionToClass(y_pred) 
      ev = Evaluation(mode)
      print(f"Dimension = {dim}")
      print(ev.computeF1(class_pred,y_test))
      print(ev.computeF1(class_pred,y_test,"macro"))
      print(ev.computeF1(class_pred,y_test,"micro"))
    print("3. Experiment: FINISHED")
    print ("4. Experiment: LSTM - BINARY- OVERSAMPLING")
    model= ModelType.LSTM
    mode= Classification.BINARY
    print ("WITHOUT EXTENSION" )
    dim = 9000
    pd = preprocessData(mode)
    filtered_data = pd.filterData(dim)
    balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
    X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
    model = LSTMModel(dimension,mode)
    history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size)
    y_pred = model.predict(X_test, batch_size)
    class_pred = model.predictitionToClass(y_pred) 
    ev = Evaluation(mode)
    print(f"Dimension = {dim}")
    print(ev.computeF1(class_pred,y_test))
    print(ev.computeF1(class_pred,y_test,"macro"))
    print(ev.computeF1(class_pred,y_test,"micro"))
    print("4. Experiment: FINISHED")

    print ("5. Experiment: CNN - MULTICLASS - OVERSAMPLING")
    model= ModelType.CNN
    mode= Classification.MULTICLASS
    print ("PLAY WITH EXTENSION" )
    for dim in [6000,9000,12000,15000,18000]:
      pd_CNN = preprocessData(mode)
      extended_data_CNN = pd_CNN.extendData(length = dim)
      balanced_data_CNN = pd_CNN.balanceData(extended_data_CNN,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor) #Strategy.OVERUNDERSAMPLE
      spectograms,dimension = pd_CNN.generateSpectograms(balanced_data_CNN[0])
      X_train, X_test, y_train, y_test = pd_CNN.splitData((spectograms,balanced_data_CNN[1]))
      model = CNNModel(dimension,mode)
      history = model.train(X_train.reshape((-1, *dimension)), X_test.reshape((len(y_test),*dimension)), y_train, y_test,epochs,batch_size)
      y_pred = model.predict(X_test.reshape((-1, *dimension)), batch_size)
      class_pred = model.predictitionToClass(y_pred) 
      ev = Evaluation(mode)
      print(f"Dimension = {dim}")
      print(ev.computeF1(class_pred,y_test))
      print(ev.computeF1(class_pred,y_test,"micro"))
      print(ev.computeF1(class_pred,y_test,"macro"))
    print("5. Experiment: FINISHED")
    print ("6. Experiment: LSTM - MULTICLASS- OVERSAMPLING")
    model= ModelType.CNN
    mode= Classification.MULTICLASS
    print ("WITHOUT EXTENSION" )
    dim = 9000
    pd_CNN = preprocessData(mode)
    extended_data_CNN = pd_CNN.extendData(length = dim)
    balanced_data_CNN = pd_CNN.balanceData(extended_data_CNN,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor) #Strategy.OVERUNDERSAMPLE
    spectograms,dimension = pd_CNN.generateSpectograms(balanced_data_CNN[0])
    X_train, X_test, y_train, y_test = pd_CNN.splitData((spectograms,balanced_data_CNN[1]))
    model = CNNModel(dimension,mode)
    history = model.train(X_train.reshape((-1, *dimension)), X_test.reshape((len(y_test),*dimension)), y_train, y_test,epochs,batch_size)
    y_pred = model.predict(X_test.reshape((-1, *dimension)), batch_size)
    class_pred = model.predictitionToClass(y_pred) 
    ev = Evaluation(mode)
    print(f"Dimension = {dim}")
    print(ev.computeF1(class_pred,y_test))
    print(ev.computeF1(class_pred,y_test,"micro"))
    print(ev.computeF1(class_pred,y_test,"macro"))
    print("6. Experiment: FINISHED")

    over_sampling_factor_list = [0.1,0.2,0.25,0.3]
    under_sampling_factor_list = [0.3,0.4,0.5,0.6]
    i = 6
    for over_sampling_factor in over_sampling_factor_list:
        for under_sampling_factor in under_sampling_factor_list:
            print("over: {over_sampling_factor} - under: {under_sampling_factor}")
            i=i+1
            print ("{i}. Experiment: LSTM - MULTICLASS - UNDEROVERSAMPLING")
            model= ModelType.LSTM
            mode= Classification.MULTICLASS
            print ("PLAY WITH EXTENSION" )
            for dim in [6000,9000,12000,15000,18000]:
                pd = preprocessData(mode)
                filtered_data = pd.resizeData(length = dim)
                balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
                X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
                model = LSTMModel(dimension,mode)
                history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size,verbose=0)
                y_pred = model.predict(X_test, batch_size)
                class_pred = model.predictitionToClass(y_pred) 
                ev = Evaluation(mode)
                print(f"Dimension = {dim}")
                print(ev.computeF1(class_pred,y_test))
                print(ev.computeF1(class_pred,y_test,"macro"))
                print(ev.computeF1(class_pred,y_test,"micro"))
            print("{i}. Experiment: FINISHED")
            i=i+1
            print ("{i}. Experiment: LSTM - MULTICLASS- UNDEROVERSAMPLING")
            model= ModelType.LSTM
            mode= Classification.MULTICLASS
            print ("WITHOUT EXTENSION" )
            dim = 9000
            pd = preprocessData(mode)
            filtered_data = pd.filterData(dim)
            balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
            X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
            model = LSTMModel(dimension,mode)
            history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size)
            y_pred = model.predict(X_test, batch_size)
            class_pred = model.predictitionToClass(y_pred) 
            ev = Evaluation(mode)
            print(f"Dimension = {dim}")
            print(ev.computeF1(class_pred,y_test))
            print(ev.computeF1(class_pred,y_test,"macro"))
            print(ev.computeF1(class_pred,y_test,"micro"))
            print("{i}. Experiment: FINISHED")
            i=i+1
            print ("{i}. Experiment: LSTM - BINARY - UNDEROVERSAMPLING")
            model= ModelType.LSTM
            mode= Classification.BINARY
            print ("PLAY WITH EXTENSION" )
            for dim in [6000,9000,12000,15000,18000]:
                pd = preprocessData(mode)
                filtered_data = pd.resizeData(length = dim)
                balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
                X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
                model = LSTMModel(dimension,mode)
                history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size,verbose=0)
                y_pred = model.predict(X_test, batch_size)
                class_pred = model.predictitionToClass(y_pred) 
                ev = Evaluation(mode)
                print(f"Dimension = {dim}")
                print(ev.computeF1(class_pred,y_test))
                print(ev.computeF1(class_pred,y_test,"macro"))
                print(ev.computeF1(class_pred,y_test,"micro"))
            print("{i}. Experiment: FINISHED")
            i=i+1
            print ("{i}. Experiment: LSTM - BINARY- UNDEROVERSAMPLING")
            model= ModelType.LSTM
            mode= Classification.BINARY
            print ("WITHOUT EXTENSION" )
            dim = 9000
            pd = preprocessData(mode)
            filtered_data = pd.filterData(dim)
            balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
            X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
            model = LSTMModel(dimension,mode)
            history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size)
            y_pred = model.predict(X_test, batch_size)
            class_pred = model.predictitionToClass(y_pred) 
            ev = Evaluation(mode)
            print(f"Dimension = {dim}")
            print(ev.computeF1(class_pred,y_test))
            print(ev.computeF1(class_pred,y_test,"macro"))
            print(ev.computeF1(class_pred,y_test,"micro"))
            print("{i}. Experiment: FINISHED")
            i=i+1
            print ("{i}. Experiment: CNN - MULTICLASS - UNDEROVERSAMPLING")
            model= ModelType.CNN
            mode= Classification.MULTICLASS
            print ("PLAY WITH EXTENSION" )
            for dim in [6000,9000,12000,15000,18000]:
                pd_CNN = preprocessData(mode)
                extended_data_CNN = pd_CNN.extendData(length = dim)
                balanced_data_CNN = pd_CNN.balanceData(extended_data_CNN,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor) #Strategy.OVERUNDERSAMPLE
                spectograms,dimension = pd_CNN.generateSpectograms(balanced_data_CNN[0])
                X_train, X_test, y_train, y_test = pd_CNN.splitData((spectograms,balanced_data_CNN[1]))
                model = CNNModel(dimension,mode)
                history = model.train(X_train.reshape((-1, *dimension)), X_test.reshape((len(y_test),*dimension)), y_train, y_test,epochs,batch_size)
                y_pred = model.predict(X_test.reshape((-1, *dimension)), batch_size)
                class_pred = model.predictitionToClass(y_pred) 
                ev = Evaluation(mode)
                print(f"Dimension = {dim}")
                print(ev.computeF1(class_pred,y_test))
                print(ev.computeF1(class_pred,y_test,"micro"))
                print(ev.computeF1(class_pred,y_test,"macro"))
            print("{i}. Experiment: FINISHED")
            i=i+1
            print ("{i}. Experiment: LSTM - MULTICLASS- UNDEROVERSAMPLING")
            model= ModelType.CNN
            mode= Classification.MULTICLASS
            print ("WITHOUT EXTENSION" )
            dim = 9000
            pd_CNN = preprocessData(mode)
            extended_data_CNN = pd_CNN.extendData(length = dim)
            balanced_data_CNN = pd_CNN.balanceData(extended_data_CNN,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor) #Strategy.OVERUNDERSAMPLE
            spectograms,dimension = pd_CNN.generateSpectograms(balanced_data_CNN[0])
            X_train, X_test, y_train, y_test = pd_CNN.splitData((spectograms,balanced_data_CNN[1]))
            model = CNNModel(dimension,mode)
            history = model.train(X_train.reshape((-1, *dimension)), X_test.reshape((len(y_test),*dimension)), y_train, y_test,epochs,batch_size)
            y_pred = model.predict(X_test.reshape((-1, *dimension)), batch_size)
            class_pred = model.predictitionToClass(y_pred) 
            ev = Evaluation(mode)
            print(f"Dimension = {dim}")
            print(ev.computeF1(class_pred,y_test))
            print(ev.computeF1(class_pred,y_test,"micro"))
            print(ev.computeF1(class_pred,y_test,"macro"))
            print("{i} Experiment: FINISHED")


    f.close()
    f2.close()

if __name__ == '__main__':
    print("BEGIN EXPERIMENTS")
    exp = Experiment()
    exp.experiment()
    print("FINSHED EXPERIMENTS")