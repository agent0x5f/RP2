#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "io.h"

using namespace std;
using namespace cv;

class ClasificadorKNN {
private:
    vector<Mat> datos;
    vector<int> etiquetas;
    int k;

public:
    ClasificadorKNN(int k_vecinos = 3) : k(k_vecinos) {}

    void entrenar(const vector<Mat>& datos_entrenamiento, const vector<int>& etiquetas_entrenamiento) {
        datos = datos_entrenamiento;
        etiquetas = etiquetas_entrenamiento;
    }

    int clasificarDato(const Mat& x) const {
        if (datos.empty()) return -1;

        vector<pair<double, int>> distancias;
        distancias.reserve(datos.size());

        for (size_t i = 0; i < datos.size(); ++i) {
            double dist = norm(x - datos[i], NORM_L2);
            distancias.push_back({dist, etiquetas[i]});
        }

        int k_efectivo = min(k, static_cast<int>(distancias.size()));

        partial_sort(distancias.begin(), distancias.begin() + k_efectivo, distancias.end(),
                          [](const pair<double, int>& a, const pair<double, int>& b) {
                              return a.first < b.first;
                          });

        map<int, int> conteoVotos;
        int mejorClase = -1;
        int maxVotos = 0;

        for (int i = 0; i < k_efectivo; ++i) {
            int clase = distancias[i].second;
            conteoVotos[clase]++;

            if (conteoVotos[clase] > maxVotos) {
                maxVotos = conteoVotos[clase];
                mejorClase = clase;
            }
        }

        return mejorClase;
    }
};

int main() {
    DatosArchivo datosArchivo = procesarArchivo("input.txt");
    ofstream archivoKNN("output_knn.txt");

    if (datosArchivo.caracteristicas.empty()) {
        registrarMensaje(archivoKNN, "No se cargaron datos validos desde input.txt\n");
        return -1;
    }

    reportarConfiguracionKNN(archivoKNN, datosArchivo.k_vecinos);
    ClasificadorKNN knn(datosArchivo.k_vecinos);
    knn.entrenar(datosArchivo.caracteristicas, datosArchivo.etiquetas);
    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = knn.clasificarDato(datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoKNN, datosArchivo.caracteristicas[i], pred);
    }

    archivoKNN.close();

    return 0;
}