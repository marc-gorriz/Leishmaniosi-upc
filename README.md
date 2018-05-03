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
* Run ```python src/data.py --img_path [raw path] --data_path [data path] --balance``` to randomly balance the raw data into train, test and validation sets. The script will create ```[data path]``` locating the ```[data path]/balance/``` directoy within in.

* Run ```python src/data.py --data_path [data path] --patches``` to generate the training patches with the right size and overlap, check ```src/data.py``` script.

* Then, run ```python src/data.py --data_path [data path] --parasite_score``` to generate ```[data path]/parasite_score.npy``` file.

### Launch an experiment
* Make a new configuration file based on ```config/config_exp1.py``` and save it into the ```config``` directory.
Make sure to launch all the processes over GPU. On this project there was used an NVIDIA GTX Titan X.

* To train a new model, run  ```python main.py --config_path config/[config file].py --action train```.
* To test the model, run ```python main.py --config_path config/[config file].py --action test```.

## Acknowledgements

We would like to especially thank Albert Gil Moreno from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/marc-gorriz/Leishmaniosi-upc/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:elisa.sayrol@upc.edu>.
