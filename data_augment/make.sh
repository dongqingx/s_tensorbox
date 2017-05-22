g++ -c image_augmentor.cpp image_augmentor.h `pkg-config --cflags --libs opencv`
g++ -c main.cpp
g++ -o main image_augmentor.o main.o  `pkg-config --cflags --libs opencv`
rm main.o image_augmentor.o image_augmentor.h.gch
g++ -o libimageaugmentor.so -std=c++11 -shared -fPIC interface.cpp `pkg-config --cflags --libs opencv`
