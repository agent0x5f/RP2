#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "io.h"

using namespace std;
using namespace cv;

struct NodoArbol {
    int indiceCaracteristica;
    double valorUmbral;
    int etiquetaHoja;
    bool esHoja;
    NodoArbol* izquierdo;
    NodoArbol* derecho;

    NodoArbol() : indiceCaracteristica(-1), valorUmbral(0.0), etiquetaHoja(-1), esHoja(false), izquierdo(nullptr), derecho(nullptr) {}
};

void liberarArbol(NodoArbol* nodo) {
    if (!nodo) return;
    liberarArbol(nodo->izquierdo);
    liberarArbol(nodo->derecho);
    delete nodo;
}

map<int, double> calcularProbabilidades(const vector<int>& etiquetas) {
    map<int, int> conteos;
    for (int etiqueta : etiquetas) {
        conteos[etiqueta]++;
    }

    map<int, double> probabilidades;
    double total = static_cast<double>(etiquetas.size());
    for (const auto& par : conteos) {
        probabilidades[par.first] = par.second / total;
    }
    return probabilidades;
}

double calcularEntropia(const vector<int>& etiquetas) {
    if (etiquetas.empty()) return 0.0;

    map<int, double> probabilidades = calcularProbabilidades(etiquetas);
    double entropia = 0.0;

    for (const auto& par : probabilidades) {
        if (par.second > 0) {
            entropia -= par.second * log2(par.second);
        }
    }
    return entropia;
}

double calcularGananciaInformacion(const vector<int>& etiquetasPadre, const vector<int>& etiquetasHijoIzquierdo, const vector<int>& etiquetasHijoDerecho) {
    double entropiaPadre = calcularEntropia(etiquetasPadre);

    double pesoIzquierdo = static_cast<double>(etiquetasHijoIzquierdo.size()) / etiquetasPadre.size();
    double pesoDerecho = static_cast<double>(etiquetasHijoDerecho.size()) / etiquetasPadre.size();

    double entropiaHijos = (pesoIzquierdo * calcularEntropia(etiquetasHijoIzquierdo)) +
                           (pesoDerecho * calcularEntropia(etiquetasHijoDerecho));

    return entropiaPadre - entropiaHijos;
}

int obtenerClaseMayoritaria(const vector<int>& etiquetas) {
    map<int, int> conteos;
    int maxCuenta = 0;
    int claseMayoritaria = -1;
    for (int etiqueta : etiquetas) {
        conteos[etiqueta]++;
        if (conteos[etiqueta] > maxCuenta) {
            maxCuenta = conteos[etiqueta];
            claseMayoritaria = etiqueta;
        }
    }
    return claseMayoritaria;
}

void dividirDatos(const vector<Mat>& datos, const vector<int>& etiquetas, int indiceCaracteristica, double umbral, vector<Mat>& datosIzquierda, vector<int>& etiquetasIzquierda, vector<Mat>& datosDerecha, vector<int>& etiquetasDerecha) {
    for (size_t i = 0; i < datos.size(); ++i) {
        if (datos[i].at<float>(indiceCaracteristica, 0) <= umbral) {
            datosIzquierda.push_back(datos[i]);
            etiquetasIzquierda.push_back(etiquetas[i]);
        } else {
            datosDerecha.push_back(datos[i]);
            etiquetasDerecha.push_back(etiquetas[i]);
        }
    }
}

NodoArbol* construirArbol(const vector<Mat>& datos, const vector<int>& etiquetas, int profundidadActual, int maximaProfundidad) {
    NodoArbol* nodo = new NodoArbol();

    if (datos.empty()) {
        nodo->esHoja = true;
        return nodo;
    }

    int claseMayoritaria = obtenerClaseMayoritaria(etiquetas);
    bool todosMismaClase = true;
    for (size_t i = 1; i < etiquetas.size(); ++i) {
        if (etiquetas[i] != etiquetas[0]) {
            todosMismaClase = false;
            break;
        }
    }

    if (todosMismaClase || profundidadActual >= maximaProfundidad) {
        nodo->esHoja = true;
        nodo->etiquetaHoja = claseMayoritaria;
        return nodo;
    }

    int numCaracteristicas = datos[0].rows;
    double mejorGanancia = 0.0;
    int mejorCaracteristica = -1;
    double mejorUmbral = 0.0;
    vector<Mat> mejorDatosIzquierda, mejorDatosDerecha;
    vector<int> mejorEtiquetasIzquierda, mejorEtiquetasDerecha;

    for (int i = 0; i < numCaracteristicas; ++i) {
        for (size_t j = 0; j < datos.size(); ++j) {
            double umbralActual = datos[j].at<float>(i, 0);

            vector<Mat> datosIzquierda, datosDerecha;
            vector<int> etiquetasIzquierda, etiquetasDerecha;

            dividirDatos(datos, etiquetas, i, umbralActual, datosIzquierda, etiquetasIzquierda, datosDerecha, etiquetasDerecha);

            if (datosIzquierda.empty() || datosDerecha.empty()) continue;

            double ganancia = calcularGananciaInformacion(etiquetas, etiquetasIzquierda, etiquetasDerecha);

            if (ganancia > mejorGanancia) {
                mejorGanancia = ganancia;
                mejorCaracteristica = i;
                mejorUmbral = umbralActual;
                mejorDatosIzquierda = move(datosIzquierda);
                mejorEtiquetasIzquierda = move(etiquetasIzquierda);
                mejorDatosDerecha = move(datosDerecha);
                mejorEtiquetasDerecha = move(etiquetasDerecha);
            }
        }
    }

    if (mejorGanancia == 0.0) {
        nodo->esHoja = true;
        nodo->etiquetaHoja = claseMayoritaria;
        return nodo;
    }

    nodo->indiceCaracteristica = mejorCaracteristica;
    nodo->valorUmbral = mejorUmbral;
    nodo->izquierdo = construirArbol(mejorDatosIzquierda, mejorEtiquetasIzquierda, profundidadActual + 1, maximaProfundidad);
    nodo->derecho = construirArbol(mejorDatosDerecha, mejorEtiquetasDerecha, profundidadActual + 1, maximaProfundidad);

    return nodo;
}

int clasificarDatoArbol(NodoArbol* raiz, const Mat& dato) {
    if (raiz->esHoja) {
        return raiz->etiquetaHoja;
    }

    if (dato.at<float>(raiz->indiceCaracteristica, 0) <= raiz->valorUmbral) {
        return clasificarDatoArbol(raiz->izquierdo, dato);
    } else {
        return clasificarDatoArbol(raiz->derecho, dato);
    }
}

int exportarNodosDOT(ofstream& archivo, NodoArbol* nodo, int& contadorID) {
    if (!nodo) return -1;

    int idActual = contadorID++;

    if (nodo->esHoja) {
        archivo << "    node" << idActual << " [label=\"Clase " << nodo->etiquetaHoja << "\", shape=ellipse, style=filled, fillcolor=lightblue];\n";
    } else {
        archivo << "    node" << idActual << " [label=\"X[" << nodo->indiceCaracteristica << "] <= " << nodo->valorUmbral << "\", shape=box];\n";
    }

    if (!nodo->esHoja) {
        int idIzquierdo = exportarNodosDOT(archivo, nodo->izquierdo, contadorID);
        int idDerecho = exportarNodosDOT(archivo, nodo->derecho, contadorID);

        if (idIzquierdo != -1) {
            archivo << "    node" << idActual << " -> node" << idIzquierdo << " [label=\" Si\"];\n";
        }
        if (idDerecho != -1) {
            archivo << "    node" << idActual << " -> node" << idDerecho << " [label=\" No\"];\n";
        }
    }

    return idActual;
}

void trazarEstructuraArbolDOT(ofstream& archivo, NodoArbol* raiz) {
    archivo << "digraph ArbolDecision {\n";
    int contadorID = 0;
    exportarNodosDOT(archivo, raiz, contadorID);

    archivo << "}\n";
}

int main() {
    DatosArchivo datosArchivo = procesarArchivo("input2.txt");
    ofstream archivoArbol("output_arbol.txt");
    ofstream archivoDOT("output_dot.txt");

    if (datosArchivo.caracteristicas.empty()) {
        registrarMensaje(archivoArbol, "No se cargaron datos validos desde input.txt\n");
        return -1;
    }

    reportarConfiguracionArbol(archivoArbol, datosArchivo.maxima_profundidad);

    NodoArbol* raiz = construirArbol(datosArchivo.caracteristicas, datosArchivo.etiquetas, 0, datosArchivo.maxima_profundidad);

    if (archivoDOT.is_open()) {
        trazarEstructuraArbolDOT(archivoDOT, raiz);
    }

    for (size_t i = 0; i < datosArchivo.caracteristicas.size(); ++i) {
        int pred = clasificarDatoArbol(raiz, datosArchivo.caracteristicas[i]);
        reportarPrediccion(archivoArbol, datosArchivo.caracteristicas[i], pred);
    }

    liberarArbol(raiz);
    archivoArbol.close();
    archivoDOT.close();

    return 0;
}