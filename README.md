[![licensebuttons
by-nc-sa-white](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# *RSV4H&Q* - River surface velocity for Water depth and Dischagre

*RSV4H&Q* is a work developed to use deep learning for river discharge inversion using surface velocity distribution. It is based on pytorch.

- Physics-informed datasets: Generated via OpenFOAM with varied channel widths (1-10 m), water depths (0.3-10 m), and discharges, embedding real river flow mechanisms.
- Core optimizations: Data gradient enhancement, GELU activation function, and bank division/cross-validation to exclude abnormal cross-sections.
- Strong validation: R^2 > 0.99 for simulated data and MSE < 0.36 (m^3/s)^2 for real-river (Swindale Beck) predictions, bypassing the need for bathymetric surveys.

## Dependencies

The easiest way to install all dependencies is to use the
\"requirements.txt\" file and 
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html):

``` bash
$ conda create -n RSV4HQ python=3.12
$ conda activate RSV4HQ
$ pip install -r requirements.txt 
```

which creates an environment named \"RSV4HQ\". You can change the name to
suit your need.

## Data

The training and validation data were generated using hydraulic models such as OpenFOAM and HEC-RAS. The complete dataset required for training and validation is provided in the `Data` folder.

An example case of the rectangular channel model used to generate the training data can be found in the `FOAM` folder and is intended to be run with OpenFOAM-8. 

For the validation data based on real river scenarios, the computational approach follows the methodology implemented in the [Masafu et al. (2022)](http://dx.doi.org/10.5525/gla.researchdata.1325) package.

Note: These resources are only needed if you wish to reproduce the data generation process using the same methodology.

### Multi-scale Rectangular Flow Datasets

The `datasets` folder contains three datasets of multiscale rectangular flow fields generated using OpenFOAM by batch calculations. These four datasets are divided for training purposes, with the difficulty level progressing from easy to difficult and their details are shown in the table below.

<div align="center">

| Datasets |Basedat| V (m/s) | H (m) | B (m) | Case Numbers | Discrete Points |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset1 | RiverCalder500 | 0.1 ~ 3 | 0.3 ~ 5 | 5 | 500 | 200 |
| Dataset2 | case240913_5K | 0.1 ~ 3 | 0.3 ~ 5 | 1 ~ 10 | 5000 | 200 |
| Dataset3 | Random5000 | 0.1 ~ 3 | H < B | 1 ~ 10 | 2500 | 1000 |
| Dataset4 | Random5000 | 0.1 ~ 3 | 0.3 ~ 10 | 1 ~ 10 | 5000 | 1000 |

</div>

The data is preprocessed and imported into the training task via `Functions`. There are three functions in our study:

- `Function`: raw data is fed directly into the model  
- `Functionpro`: data is processed with gradient enhancement  
- `Fuctionpro_parameter_constraints`: enhanced data is filtered within a certain range  

If you want to change the data input method, simply switch the `Function` to achieve it. For information on how to use this, please refer to the code in `dataload.ipynb`.

### Real-River (Swindale Beck)

The calculated V.tif and H.tif results for a real river are provided, which can be obtained using HEC-RAS software, for verification purposes. We provide the data processing methods before inputting the data into our model, including extracting the river segment calculation area, extracting the river centerline, extracting cross-sectional velocity and water depth data along the vertical direction, and other data analyses, which can be found in `tifread.ipynb`.

## Train and Test

We provide all the neural network models tested in the article, which are located in `Model.py` within the `Model_Base` folder. In the Multi-scale Rectangular Flow Datasets folder, we provide the training method, parameters, and visualization results for our most important ResNet model.

The Real_River folder demonstrates the validation of our model on real rivers. The model used here has been structurally adjusted for data size adaptability. We have added a half-training-prediction strategy (clearly marked modification locations in the `code`) to address the issue of asymmetric underlying surfaces in real rivers, while also enabling re-validation of the prediction results. If you are only concerned about the results, you can directly check the model usage in pre, run our trained model, and view the results. If you are more interested in the training process, you can find our training strategies and training parameters in `train.ipynb`.

Note: The real river dataset package is extracted from the complete channel calculation results provided in data-river. The specific extraction method can be found in the code under that folder.

## Conclusion

River discharge measurement is critical for water resource management, yet conventional methods are constrained by reliance on extensive in-situ or bathymetric data—limiting applicability in data-scarce regions. This work aligns with your journal’s focus on hydrology and data-driven modeling, offering a scalable solution for small-to-medium rivers in data-scarce regions.

## License

Creative Commons Attribution-NonCommercial 4.0 International License

## Author

| Yuncheng Xu, Ph.D.
| Associate Professor

| College of Water Resources and Civil Engineering
| China Agricultural University

E-mail: ycxu@cau.edu.cn; ycxu1990@gmail.com

Web: <http://xuyuncheng.com>

| Tianyi Xu
| Graduate Research Assistant

| College of Water Resources and Civil Engineering
| China Agricultural University

E-mail: xuty@cau.edu.cn

## Contributors and contributor agreement

If you are interested in our research, you are especially welcome to join our discussions. We will also continue testing and optimizing the model to adapt to a wider range of rivers or focus on more refined subsurface inversion.

### The list of contributors:

- (To be added)

### Contributor agreement

First of all, thanks for your interest in contributing to *RSV4H&Q*.
Collectively, we can make *RSV4H&Q* more powerful, better, and easier to
use.

Because of legal reasons and like many successful open source projects,
contributors have to sign a \"Contributor License Agreement\" to grant
their rights to \"Us\". See details of the agreement on GitHub. The
signing of the agreement is automatic when a pull request is issued.

If you are just a user of *RSV4H&Q*, the contributor agreement is
irrelevant.
