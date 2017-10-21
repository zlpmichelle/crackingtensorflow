base=10
height=5
area=1/2*(base*height)
print("Area of our triangle is : ", area)

file = open("/Users/lipingzhang/Desktop/program/pycharm/seq2seq/MNIST_data/0622_train_features.csv","r")
lines = []
with file as myFile:
    for line in file:
        feat = []
        line = line.split(',')
        for i in range(0, len(line)):
            feat.append(float(line[i]))
        lines.append(feat.tolist())

