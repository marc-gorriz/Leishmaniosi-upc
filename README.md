# Leishmaniasis Parasite Segmentation and Classification

| ![Marc Górriz][MarcGorriz-photo]  |  ![Albert Aparicio][AlbertAparicio-photo] | ![Elisa Sayrol][ElisaSayrol-photo]  | ![Verónica Vilaplana][VeronicaVilaplana-photo]  |
|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Albert Aparicio][AlbertAparicio-web] | [Elisa Sayrol][ElisaSayrol-web] | [Verónica Vilaplana][VeronicaVilaplana-web] |

[MarcGorriz-web]: https://www.linkedin.com/in/marc-górriz-blanch-74501a123/
[AlbertAparicio-web]: https://www.linkedin.com/in/albert-aparicio-isarn-3977038b/
[ElisaSayrol-web]: https://imatge.upc.edu/web/people/elisa-sayrol
[VeronicaVilaplana-web]: https://imatge.upc.edu/web/people/veronica-vilaplana



[MarcGorriz-photo]: https://raw.githubusercontent.com/marc-gorriz/Leishmaniosi-upc/master/authors/MarcGorriz.jpg
[AlbertAparicio-photo]: https://raw.githubusercontent.com/marc-gorriz/Leishmaniosi-upc/master/authors/AlbertAparicio.jpeg
[ElisaSayrol-photo]: https://raw.githubusercontent.com/marc-gorriz/Leishmaniosi-upc/master/authors/ElisaSayrol.jpg
[VeronicaVilaplana-photo]: https://raw.githubusercontent.com/marc-gorriz/Leishmaniosi-upc/master/authors/VeronicaVilaplana.jpg

## Abstract

Leishmaniasis is considered a neglected disease that causes thousands of deaths annually in some tropical and subtropical countries. There are various techniques to diagnose leishmaniasis of which manual microscopy is considered to be the gold standard. There is a need for the development of automatic techniques that are able to detect parasites in a robust and unsupervised manner. In this paper we present a procedure for automatizing the detection process based on a deep learning approach. We train a U-net model that successfully segments leismania parasites and classifies them into promastigotes, amastigotes and adhered parasites.


![system-fig]

[system-fig]: https://raw.githubusercontent.com/marc-gorriz/Leishmaniosi-upc/master/img/system_diagram.png

---

## How to use

### Dependencies

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.4.2.

```
pip install -r https://github.com/marc-gorriz/Leishmaniosi-upc/blob/master/requeriments.txt
```

### Prepare the data
Make sure that ```raw``` dir is located in the root of this project.
Also, the tree of ```raw``` dir must be like:
```
-raw
 |
 ---- img
 |    |
 |    ---- img1.jpg
 |    |
 |    ---- …
 |
 ---- labels
      |
      ---- img1.png
      |
      ---- …
```
* Run ```python src/data.py --img_path [raw path] --balance``` to randomly balance the raw data into train, test and validation sets. The script will create ```data``` path in the project root locating the ```data/balance``` path within in.

* Run ```python src/data.py --data_path data --patches``` to generate the training patches with the right size and overlap, check ```src/data.py``` script.

* Then, run ```python src/data.py --data_path data --parasite_score``` to generate ```data/parasite_score.npy``` file.

### Launch an experiment
* Make a new configuration file based on ```config/config_exp1.py``` and save it into ```config``` directory.
Make sure to launch the process over a GPU. The project models were trained on a NVIDIA GTX Titan X taking around 11 GB of RAM  memory.
