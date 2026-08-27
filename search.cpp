#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// Calculate Euclidean Distance
float calculateDistance(const vector<float>& vec1, const vector<float>& vec2) {
    float sum = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum); // Return the square root of the sum
}

int main() {
    // Open the binary file the Python script created
    ifstream file("embeddings.bin", ios::binary);
    if (!file) {
        cout << "Error: Could not open embeddings.bin. Did you run extract.py?" << endl;
        return 1;
    }

    //  Load the 1,280 numbers directly into RAM
    vector<float> query_vector(1280);
    file.read(reinterpret_cast<char*>(query_vector.data()), 1280 * sizeof(float));
    file.close();

    cout << "Successfully loaded query fingerprint (1280 dimensions)." << endl;

    // Create a fake "database" of images to test against
    // (In a real system, we would load thousands of these from a database)
    vector<float> database_image_1(1280, 0.1f); // Simulating an image (filled with 0.1)
    vector<float> database_image_2(1280, 0.9f); // Simulating a different image (filled with 0.9)

    // Find the closest match
    float dist1 = calculateDistance(query_vector, database_image_1);
    float dist2 = calculateDistance(query_vector, database_image_2);

    cout << "Distance to Database Image 1: " << dist1 << endl;
    cout << "Distance to Database Image 2: " << dist2 << endl;

    if (dist1 < dist2) {
        cout << "Result: Database Image 1 is the closest match!" << endl;
    } else {
        cout << "Result: Database Image 2 is the closest match!" << endl;
    }

    return 0;
}