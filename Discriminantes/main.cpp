#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <opencv2/opencv.hpp>
#include "io.h"

using namespace std;
using namespace cv;

Mat entrenarDiscriminanteLineal(const vector<Mat>& datos, const vector<int>& etiquetas, double mu = 1.0, int maxIter = 1000) {
    if (datos.empty()) return Mat();

    int numCaracteristicas = datos[0].rows;
    Mat w = Mat::zeros(numCaracteristicas + 1, 1, CV_32F);

    bool hayErrores = true;
    int iter = 0;

    while (hayErrores && iter < maxIter) {
        hayErrores = false;

        for (size_t i = 0; i < datos.size(); ++i) {
            Mat x_ampliado = Mat::ones(numCaracteristicas + 1, 1, CV_32F);
            datos[i].copyTo(x_ampliado(Rect(0, 0, 1, numCaracteristicas)));

            double fd = w.dot(x_ampliado);

            if (etiquetas[i] == 1 && fd <= 0) {
                w += mu * x_ampliado;
                hayErrores = true;
            } else if (etiquetas[i] == 2 && fd >= 0) {
                w -= mu * x_ampliado;
                hayErrores = true;
            }
        }
        iter++;
    }
    return w;
}

int clasificarDato(const Mat& w, const Mat& x) {
    int numCaracteristicas = x.rows;
    Mat x_ampliado = Mat::ones(numCaracteristicas + 1, 1, CV_32F);
    x.copyTo(x_ampliado(Rect(0, 0, 1, numCaracteristicas)));

    return (w.dot(x_ampliado) > 0) ? 1 : 2;
}

map<int, Mat> entrenarDiscriminanteDistancia(const vector<Mat>& datos, const vector<int>& etiquetas) {
    map<int, Mat> sumas;
    map<int, int> conteos;
    map<int, Mat> centroides;

    if (datos.empty()) return centroides;

    int numCaracteristicas = datos[0].rows;

    for (size_t i = 0; i < datos.size(); ++i) {
        int clase = etiquetas[i];
        if (sumas.find(clase) == sumas.end()) {
            sumas[clase] = Mat::zeros(numCaracteristicas, 1, CV_32F);
            conteos[clase] = 0;
        }
        sumas[clase] += datos[i];
        conteos[clase]++;
    }

    for (const auto& par : sumas) {
        int clase = par.first;
        centroides[clase] = par.second / static_cast<float>(conteos[clase]);
    }
    return centroides;
}

int clasificarDatoDistancia(const map<int, Mat>& centroides, const Mat& x) {
    int mejorClase = -1;
    double maxFd = -numeric_limits<double>::infinity();

    for (const auto& par : centroides) {
        int clase = par.first;
        Mat z_i = par.second;

        double fd_i = x.dot(z_i) - 0.5 * z_i.dot(z_i);

        if (fd_i > maxFd) {
            maxFd = fd_i;
            mejorClase = clase;
        }
    }
    return mejorClase;
}

int main() {
    DatosArchivo datosArchivo = procesarArchivo("input.txt");

    ofstream archivoRegion("output_region.txt");
    ofstream archivoDistancia("output_distancia.txt");

    if (datosArchivo.caracteristicas.empty()) {
        registrarMensaje(archivoRegion, "No se cargaron datos validos desde input.txt\n");
        registrarMensaje(archivoDistancia, "No se cargaron datos validos desde input.txt\n");
        return -1;
    }

    reportarConfiguracion(archivoRegion, datosArchivo.umbral);

    // Por regiones
    Mat w = entrenarDiscriminanteLineal(datosArchivo.caracteristicas, datosArchivo.etiquetas, datosArchivo.umbral, datosArchivo.max_iter);
    reportarModeloLineal(archivoRegion, w);

    registrarMensaje(archivoRegion, "Prueba de clasificacion (Regiones) sobre set de entrenamiento:\n");
    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = clasificarDato(w, datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoRegion, datosArchivo.caracteristicas[i], pred);
    }
    registrarMensaje(archivoRegion, "\n");

    // Por distancia
    map<int, Mat> centroides = entrenarDiscriminanteDistancia(datosArchivo.caracteristicas, datosArchivo.etiquetas);
    reportarCentroides(archivoDistancia, centroides);

    registrarMensaje(archivoDistancia, "Prueba de clasificacion (Distancia) sobre set de entrenamiento:\n");
    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = clasificarDatoDistancia(centroides, datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoDistancia, datosArchivo.caracteristicas[i], pred);
    }
    registrarMensaje(archivoDistancia, "\n");

    archivoRegion.close();
    archivoDistancia.close();

    return 0;
}