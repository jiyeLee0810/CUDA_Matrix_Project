#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

void read3DInputFile (string testDirectory, int &width, int &height, int &depth, int &kernelSize, float* &inputMatrix, float* &kernelMatrix) {
    int index = 0;
    string inputFile = "sample/" + testDirectory + "/input.txt";
    string kernelFile = "sample/" + testDirectory + "/kernel.txt";
    
    fstream Ifile(inputFile);
    if (!Ifile.is_open()) {
        return;
    }
    Ifile >> width >> height >> depth;
    inputMatrix = new float[width*height*depth];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0;  k < depth; k++) {
                index = i * (height * depth) + j * depth + k; 
                Ifile >> inputMatrix[index];
            }
        }
    }
    Ifile.close();

    fstream Kfile(kernelFile);
    if (!Kfile.is_open()) {
        return;
    }
    Kfile >> kernelSize;
    kernelMatrix = new float[kernelSize*kernelSize*kernelSize];
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            for (int k = 0;  k < kernelSize; k++) {
                index = i * (kernelSize * kernelSize) + j * kernelSize + k; 
                Kfile >> kernelMatrix[index];
            }
        }
    }
    Kfile.close();
}
