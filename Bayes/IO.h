#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>

struct DatosArchivo {
    double umbral = 1.0;
    int max_iter = 1000;
    int k_vecinos = 3;
    int maxima_profundidad = 10;
    std::vector<cv::Mat> caracteristicas;
    std::vector<int> etiquetas;
};

DatosArchivo procesarArchivo(const std::string& ruta);
void registrarMensaje(std::ofstream& archivo, const std::string& mensaje);
void reportarConfiguracion(std::ofstream& archivo, double umbral);
void reportarConfiguracionArbol(std::ofstream& archivo, int maxima_profundidad);
void reportarConfiguracionKNN(std::ofstream& archivo, int k);
void reportarModeloLineal(std::ofstream& archivo, const cv::Mat& w);
void reportarCentroides(std::ofstream& archivo, const std::map<int, cv::Mat>& centroides);
void reportarPrediccion(std::ofstream& archivo, const cv::Mat& dato, int prediccion);
void reportarModeloBayesDistancia(std::ofstream& archivo, int clase, const cv::Mat& media, const cv::Mat& covInversa);
void reportarModeloBayesProbabilidad(std::ofstream& archivo, int clase, double priori, const cv::Mat& medias, const cv::Mat& varianzas);

#endif // IO_H