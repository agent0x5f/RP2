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
    std::vector<cv::Mat> caracteristicas;
    std::vector<int> etiquetas;
};

DatosArchivo procesarArchivo(const std::string& ruta);

void registrarMensaje(std::ofstream& archivo, const std::string& mensaje);
void reportarConfiguracion(std::ofstream& archivo, double umbral);
void reportarModeloLineal(std::ofstream& archivo, const cv::Mat& w);
void reportarCentroides(std::ofstream& archivo, const std::map<int, cv::Mat>& centroides);
void reportarPrediccion(std::ofstream& archivo, const cv::Mat& dato, int prediccion);

#endif // IO_H