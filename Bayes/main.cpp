#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include "io.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace cv;

struct ModeloBayesDistancia {
    Mat media;
    Mat covarianzaInversa;
};

struct ModeloNaiveBayes {
    double probabilidadPriori;
    Mat medias;
    Mat varianzas;
};

map<int, ModeloBayesDistancia> entrenarBayesDistancia(const vector<Mat>& datos, const vector<int>& etiquetas) {
    map<int, vector<Mat>> datosPorClase;
    for (size_t i = 0; i < datos.size(); ++i) {
        datosPorClase[etiquetas[i]].push_back(datos[i]);
    }

    map<int, ModeloBayesDistancia> modelos;

    for (const auto& par : datosPorClase) {
        int clase = par.first;
        const vector<Mat>& muestras = par.second;
        int n = muestras.size();
        int numCaracteristicas = muestras[0].rows;

        Mat media = Mat::zeros(numCaracteristicas, 1, CV_32F);
        for (const auto& x : muestras) {
            media += x;
        }
        media /= static_cast<float>(n);

        Mat covarianza = Mat::zeros(numCaracteristicas, numCaracteristicas, CV_32F);
        if (n > 1) {
            for (const auto& x : muestras) {
                Mat diff = x - media;
                covarianza += diff * diff.t();
            }
            covarianza /= static_cast<float>(n - 1);
        } else {
            covarianza = Mat::eye(numCaracteristicas, numCaracteristicas, CV_32F);
        }

        Mat covarianzaInversa;
        invert(covarianza, covarianzaInversa, DECOMP_SVD);

        modelos[clase] = {media, covarianzaInversa};
    }

    return modelos;
}

int clasificarDatoBayesDistancia(const map<int, ModeloBayesDistancia>& modelos, const Mat& x) {
    int mejorClase = -1;
    double maxFd = -numeric_limits<double>::infinity();

    for (const auto& par : modelos) {
        int clase = par.first;
        Mat m = par.second.media;
        Mat C_inv = par.second.covarianzaInversa;

        Mat termino1 = x.t() * C_inv * m;
        Mat termino2 = m.t() * C_inv * m;

        double fd = termino1.at<float>(0, 0) - 0.5 * termino2.at<float>(0, 0);

        if (fd > maxFd) {
            maxFd = fd;
            mejorClase = clase;
        }
    }

    return mejorClase;
}

map<int, ModeloNaiveBayes> entrenarNaiveBayes(const vector<Mat>& datos, const vector<int>& etiquetas) {
    map<int, vector<Mat>> datosPorClase;
    for (size_t i = 0; i < datos.size(); ++i) {
        datosPorClase[etiquetas[i]].push_back(datos[i]);
    }

    map<int, ModeloNaiveBayes> modelos;
    int totalDatos = datos.size();
    if (totalDatos == 0) return modelos;

    int numCaracteristicas = datos[0].rows;

    for (const auto& par : datosPorClase) {
        int clase = par.first;
        const vector<Mat>& muestras = par.second;
        int n = muestras.size();

        double priori = static_cast<double>(n) / totalDatos;

        Mat medias = Mat::zeros(numCaracteristicas, 1, CV_32F);
        for (const auto& x : muestras) {
            medias += x;
        }
        medias /= static_cast<float>(n);

        Mat varianzas = Mat::zeros(numCaracteristicas, 1, CV_32F);
        if (n > 1) {
            for (const auto& x : muestras) {
                Mat diff = x - medias;
                Mat diffCuadrado;
                pow(diff, 2, diffCuadrado);
                varianzas += diffCuadrado;
            }
            varianzas /= static_cast<float>(n - 1);
        } else {
            varianzas = Mat::ones(numCaracteristicas, 1, CV_32F) * 1e-6;
        }

        varianzas += 1e-9;

        modelos[clase] = {priori, medias, varianzas};
    }

    return modelos;
}

int clasificarDatoNaiveBayes(const map<int, ModeloNaiveBayes>& modelos, const Mat& x) {
    int mejorClase = -1;
    double maxProbLog = -numeric_limits<double>::infinity();

    for (const auto& par : modelos) {
        int clase = par.first;
        double priori = par.second.probabilidadPriori;
        Mat medias = par.second.medias;
        Mat varianzas = par.second.varianzas;

        double logProbPosteriori = log(priori);

        for (int j = 0; j < x.rows; ++j) {
            double xj = x.at<float>(j, 0);
            double media = medias.at<float>(j, 0);
            double var = varianzas.at<float>(j, 0);

            double exponente = -pow(xj - media, 2) / (2 * var);
            double verosimilitud = (1.0 / sqrt(2 * M_PI * var)) * exp(exponente);

            if (verosimilitud > 0) {
                logProbPosteriori += log(verosimilitud);
            } else {
                logProbPosteriori += -1e9;
            }
        }

        if (logProbPosteriori > maxProbLog) {
            maxProbLog = logProbPosteriori;
            mejorClase = clase;
        }
    }

    return mejorClase;
}

int main() {
    DatosArchivo datosArchivo = procesarArchivo("input2.txt");

    ofstream archivoBayesDist("output_bayes_dist.txt");
    ofstream archivoBayesProb("output_bayes_prob.txt");

    if (datosArchivo.caracteristicas.empty()) {
        registrarMensaje(archivoBayesDist, "No se cargaron datos validos desde input.txt\n");
        registrarMensaje(archivoBayesProb, "No se cargaron datos validos desde input.txt\n");
        return -1;
    }

    map<int, ModeloBayesDistancia> modelosDistancia = entrenarBayesDistancia(datosArchivo.caracteristicas, datosArchivo.etiquetas);

    for (const auto& par : modelosDistancia) {
        reportarModeloBayesDistancia(archivoBayesDist, par.first, par.second.media, par.second.covarianzaInversa);
    }

    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = clasificarDatoBayesDistancia(modelosDistancia, datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoBayesDist, datosArchivo.caracteristicas[i], pred);
    }

    map<int, ModeloNaiveBayes> modelosProbabilidad = entrenarNaiveBayes(datosArchivo.caracteristicas, datosArchivo.etiquetas);

    for (const auto& par : modelosProbabilidad) {
        reportarModeloBayesProbabilidad(archivoBayesProb, par.first, par.second.probabilidadPriori, par.second.medias, par.second.varianzas);
    }

    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = clasificarDatoNaiveBayes(modelosProbabilidad, datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoBayesProb, datosArchivo.caracteristicas[i], pred);
    }

    archivoBayesDist.close();
    archivoBayesProb.close();

    return 0;
}