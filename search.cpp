#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

float calculateDistance(const vector<float>& vec1, const float* vec2_ptr, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2_ptr[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int main() {
    int DIM = 1280;

    // 1. Load the Single Query Vector
    ifstream query_file("query.bin", ios::binary);
    if (!query_file) return 1;
    vector<float> query_vector(DIM);
    query_file.read(reinterpret_cast<char*>(query_vector.data()), DIM * sizeof(float));
    query_file.close();

    // 2. Load the Entire Database
    ifstream db_file("database.bin", ios::binary | ios::ate);
    if (!db_file) return 1;
    streamsize size = db_file.tellg();
    db_file.seekg(0, ios::beg);
    
    int num_images = size / (DIM * sizeof(float));
    vector<float> database(num_images * DIM);
    db_file.read(reinterpret_cast<char*>(database.data()), size);
    db_file.close();

    // 3. Calculate Distances
    vector<pair<float, int>> distances;
    for (int i = 0; i < num_images; i++) {
        float dist = calculateDistance(query_vector, &database[i * DIM], DIM);
        distances.push_back({dist, i});
    }

    // 4. Sort and Output the Top 5
    sort(distances.begin(), distances.end());
    for (int i = 0; i < min(5, num_images); i++) {
        cout << distances[i].second << endl; 
    }

    return 0;
}