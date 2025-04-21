import matplotlib.pyplot as plt

#Läser från results.txt och extraherar resultaten
try:
    with open('results.txt', 'r') as f:
        content = f.readlines()

    #Extraherar testnoggrannheter från filen
    accuracies = []
    for line in content:
        if "Test Accuracy" in line:
            accuracy = float(line.split(":")[1].strip())  
            accuracies.append(accuracy)

    # Visualiserar testnoggrannheterna för modeller med olika antal lager
    layers = [1, 2, 3, 4, 5, 6]
    plt.plot(layers, accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')

    plt.title("Model Accuracy vs Number of Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.xticks(layers)
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Error: results.txt not found! Please run train.py first to generate the file.")
