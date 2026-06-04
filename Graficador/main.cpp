#include <opencv2/opencv.hpp>
#include <string>

void dibujarArista(cv::Mat& imagen, cv::Point origen, cv::Point destino, const std::string& etiqueta) {
    cv::line(imagen, origen, destino, cv::Scalar(0, 0, 0), 2);

    cv::Point punto_medio((origen.x + destino.x) / 2, (origen.y + destino.y) / 2 - 10);
    cv::putText(imagen, etiqueta, punto_medio, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 2);
}

void dibujarNodo(cv::Mat& imagen, cv::Point centro, const std::string& texto, bool es_hoja) {
    int grosor = 1;
    int linea_base = 0;
    cv::Size tam_texto = cv::getTextSize(texto, cv::FONT_HERSHEY_SIMPLEX, 0.5, grosor, &linea_base);

    if (es_hoja) {
        cv::Scalar color_azul_claro(230, 216, 173);
        cv::ellipse(imagen, centro, cv::Size(tam_texto.width / 2 + 20, tam_texto.height + 15),
                    0, 0, 360, color_azul_claro, cv::FILLED);
        cv::ellipse(imagen, centro, cv::Size(tam_texto.width / 2 + 20, tam_texto.height + 15),
                    0, 0, 360, cv::Scalar(0, 0, 0), 2);
    } else {
        cv::Rect caja(centro.x - tam_texto.width / 2 - 15, centro.y - tam_texto.height / 2 - 10,
                      tam_texto.width + 30, tam_texto.height + 20);
        cv::rectangle(imagen, caja, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::rectangle(imagen, caja, cv::Scalar(0, 0, 0), 2);
    }

    cv::Point origen_texto(centro.x - tam_texto.width / 2, centro.y + tam_texto.height / 2);
    cv::putText(imagen, texto, origen_texto, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), grosor);
}

int main() {
    cv::Mat lienzo(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::Point pt_nodo0(400, 100);
    cv::Point pt_nodo1(250, 250);
    cv::Point pt_nodo4(550, 250);
    cv::Point pt_nodo2(150, 400);
    cv::Point pt_nodo3(350, 400);

    dibujarArista(lienzo, pt_nodo0, pt_nodo1, "Si");
    dibujarArista(lienzo, pt_nodo0, pt_nodo4, "No");
    dibujarArista(lienzo, pt_nodo1, pt_nodo2, "Si");
    dibujarArista(lienzo, pt_nodo1, pt_nodo3, "No");

    dibujarNodo(lienzo, pt_nodo0, "X[0] <= 5.8", false);
    dibujarNodo(lienzo, pt_nodo1, "X[0] <= 2.5", false);
    dibujarNodo(lienzo, pt_nodo4, "Clase 3", true);
    dibujarNodo(lienzo, pt_nodo2, "Clase 1", true);
    dibujarNodo(lienzo, pt_nodo3, "Clase 2", true);

    std::string archivo_salida = "output_grafico.jpg";
    cv::imwrite(archivo_salida, lienzo);

    return 0;
}