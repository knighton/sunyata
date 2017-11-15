from sunyata.dataset.cifar import load_cifar


for classes in [10, 20, 100]:
    train, val, class_names = load_cifar(classes=classes)
    print(train[0].shape, train[1].shape)
