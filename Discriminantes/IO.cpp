#include "io.h"
#include <iostream>
#include <sstream>

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

void reportarConfiguracion(ofstream& archivo, double umbral) {
    stringstream ss;
    ss << "Parametro mu (umbral): " << umbral << "\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarModeloLineal(ofstream& archivo, const Mat& w) {
    stringstream ss;
    ss << "--- Entrenando discriminante lineal por regiones ---\n";
    ss << "Vector de pesos w obtenido:\n" << w << "\n\n";
    registrarMensaje(archivo, ss.str());
}

void reportarCentroides(ofstream& archivo, const map<int, Mat>& centroides) {
    stringstream ss;
    ss << "--- Entrenando discriminante por distancia ---\n";
    for (const auto& par : centroides) {
        ss << "Centroide de clase " << par.first << ":\n" << par.second << "\n";
    }
    ss << "\n";
    registrarMensaje(archivo, ss.str());
}

void reportarPrediccion(ofstream& archivo, const Mat& dato, int prediccion) {
    stringstream ss;
    ss<< dato.t() << "-> Clase predicha: " << prediccion << "\n";
    registrarMensaje(archivo, ss.str());
}