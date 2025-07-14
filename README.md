# Transferencia de Timbre

Este repositorio contiene el código utilizado en el TFG "Transferencia de Timbre en Audio con IA Generativa para Datos no Paralelos", realizado por Óscar Marín Flores.

# Referencias

Este proyecto utiliza y adapta partes del código de los siguientes repositorios:

- [hifi-gan](https://github.com/jik876/hifi-gan): Para el entrenamiento de la red HiFi-GAN y la síntesis de audio.
- [tt-vae-gan](https://github.com/RussellSB/tt-vae-gan): Para el entrenamiento del modelo VAE-GAN y la conversión de los audios.
- [autovc](https://github.com/auspicious3000/autovc): Para el entrenamiento de AUTOVC y la conversión de los audios.

# Organización

En el repositorio se encuentran las siguientes carpetas:
- `autovc`: Contiene el código y los modelos para la transferencia de timbre utilizando AUTOVC.
- `vae_gan`: Contiene el código y los modelos para la transferencia de timbre utilizando VAE-GAN.
- `hifi_gan`: Contiene el código y los modelos para la síntesis de audio utilizando HiFi-GAN.
- `audios`: Contiene ejemplos de audios originales y convertidos.

Además, se incluyen los siguientes archivos:
- `README.md`: Este archivo, que proporciona una descripción general.
- `evaluation.ipynb`: Jupyter Notebook que contiene la definición de las métricas de evaluación, desde el que se ejecutaron las evaluaciones. Además, contiene algunas otras celdas con funciones generales.

Aunque se incluye el código fuente de cada modelo, durante el desarrollo del proyecto se hizo uso de Jupyter Notebooks para facilitar la experimentación y visualización de resultados. Los Notebooks se encuentran en las carpetas correspondientes a cada modelo. En estos, se redefinieron gran parte de las funciones del código fuente para adaptarlas a la ejecución en Notebooks, permitiendo así una mejor interacción.

## Ejemplos de audio

- [Audio original](audios/originales/f1_scales_belt_i.wav)
- [Audio generado](audios/convertidos/exp1/G2_f1_scales_belt_i_mel.wav)