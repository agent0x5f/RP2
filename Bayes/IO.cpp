#include "io.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

DatosArchivo procesarArchivo(const string& ruta) {
    DatosArchivo datos;
    ifstream archivo(ruta);

    if (!archivo.is_open()) {
        cerr << "Error: No se pudo abrir " << ruta << "\n";
        return datos;
    }

    string linea;
    while (getline(archivo, linea)) {
        if (linea.empty() || linea == "\r" || linea == "\n") continue;

        if (linea[0] == '@') {
            try {
                size_t posDospuntos = linea.find(":");
                if (posDospuntos != string::npos) {
                    string valorStr = linea.substr(posDospuntos + 1);
                    if (linea.find("@umbral:") != string::npos) datos.umbral = stod(valorStr);
                    else if (linea.find("@max_iter:") != string::npos) datos.max_iter = stoi(valorStr);
                    else if (linea.find("@k_vecinos:") != string::npos) datos.k_vecinos = stoi(valorStr);
                    else if (linea.find("@max_profundidad:") != string::npos) datos.maxima_profundidad = stoi(valorStr);
                }
            } catch (const exception& e) {
                cerr << "Error al procesar parametro: " << linea << " - " << e.what() << "\n";
            }
        } else {
            stringstream ss(linea);
            string token;
            vector<float> fila;

            while (getline(ss, token, ',')) {
                fila.push_back(stof(token));
            }

            if (fila.size() > 1) {
                int etiqueta = static_cast<int>(fila.back());
                fila.pop_back();

                Mat x(fila.size(), 1, CV_32F);
                for (size_t i = 0; i < fila.size(); ++i) {
                    x.at<float>(i, 0) = fila[i];
                }

                datos.caracteristicas.push_back(x);
                datos.etiquetas.push_back(etiqueta);
            }
        }
    }
    return datos;
}

void registrarMensaje(ofstream& archivo, const string& mensaje) {
    cout << mensaje;
    if (archivo.is_open()) {
        archivo << mensaje;
    }
}

void reportarConfiguracionArbol(ofstream& archivo, int maxima_profundidad) {
    stringstream ss;
    ss << "Parametro maxima_profundidad configurado a: " << maxima_profundidad << "\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarConfiguracionKNN(ofstream& archivo, int k) {
    stringstream ss;
    ss << "Parametro k_vecinos configurado a: " << k << "\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarConfiguracion(ofstream& archivo, double umbral) {
    stringstream ss;
    ss << "Parametro mu (umbral) configurado a: " << fixed << setprecision(2) << umbral << "\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarModeloLineal(ofstream& archivo, const Mat& w) {
    stringstream ss;
    ss << "--- Entrenando Discriminante Lineal por Regiones ---\n";
    ss << "Vector de pesos w obtenido: [";
    for (int i = 0; i < w.rows; ++i) {
        ss << fixed << setprecision(2) << w.at<float>(i, 0);
        if (i < w.rows - 1) ss << ", ";
    }
    ss << "]\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarCentroides(ofstream& archivo, const map<int, Mat>& centroides) {
    stringstream ss;
    ss << "--- Entrenando Discriminante por Distancia ---\n";
    for (const auto& par : centroides) {
        ss << "Centroide Clase " << par.first << ": [";
        for (int i = 0; i < par.second.rows; ++i) {
            ss << fixed << setprecision(2) << par.second.at<float>(i, 0);
            if (i < par.second.rows - 1) ss << ", ";
        }
        ss << "]\n";
    }
    ss << "\n";
    registrarMensaje(archivo, ss.str());
}

void reportarPrediccion(ofstream& archivo, const Mat& dato, int prediccion) {
    stringstream ss;
    ss << "[";
    for (int i = 0; i < dato.rows; ++i) {
        ss << fixed << setprecision(2) << dato.at<float>(i, 0);
        if (i < dato.rows - 1) ss << ", ";
    }
    ss << "] -> " << prediccion << "\n";
    registrarMensaje(archivo, ss.str());
}

void reportarModeloBayesDistancia(ofstream& archivo, int clase, const Mat& media, const Mat& covInversa) {
    stringstream ss;
    ss << "Clase " << clase << " - Vector de medias: [";
    for (int i = 0; i < media.rows; ++i) {
        ss << fixed << setprecision(2) << media.at<float>(i, 0);
        if (i < media.rows - 1) ss << ", ";
    }
    ss << "]\n";

    ss << "Clase " << clase << " - Matriz de covarianza inversa:\n";
    for (int i = 0; i < covInversa.rows; ++i) {
        ss << "  [";
        for (int j = 0; j < covInversa.cols; ++j) {
            ss << fixed << setprecision(2) << covInversa.at<float>(i, j);
            if (j < covInversa.cols - 1) ss << ", ";
        }
        ss << "]\n";
    }
    ss << "\n";
    registrarMensaje(archivo, ss.str());
}

void reportarModeloBayesProbabilidad(ofstream& archivo, int clase, double priori, const Mat& medias, const Mat& varianzas) {
    stringstream ss;
    ss << "Clase " << clase << " - Probabilidad inicial: " << fixed << setprecision(4) << priori << "\n";

    ss << "Clase " << clase << " - Vector de medias: [";
    for (int i = 0; i < medias.rows; ++i) {
        ss << fixed << setprecision(2) << medias.at<float>(i, 0);
        if (i < medias.rows - 1) ss << ", ";
    }
    ss << "]\n";

    ss << "Clase " << clase << " - Vector de varianzas: [";
    for (int i = 0; i < varianzas.rows; ++i) {
        ss << fixed << setprecision(2) << varianzas.at<float>(i, 0);
        if (i < varianzas.rows - 1) ss << ", ";
    }
    ss << "]\n\n";
    registrarMensaje(archivo, ss.str());
}