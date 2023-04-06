# Machine-learning modelling of Infrasound Transmission-Loss (MITL)

## Summary
MITL generates infrasound ground Transmission Loss (TL) profiles over 1000 km based on realistic and range-dependent atmospheric models. MITL is based on a Convolutional Neural Network (CNN) architecture to segment 2d wind profiles (range vs altitude) and produce realistic TL estimates.
<p align="center">
<img width="447" alt="image" src="https://user-images.githubusercontent.com/6717390/230470859-6b097a03-076b-47db-87fd-91e46ce8514c.png">
</p>
Figure: (from Brissaud et al, 2023) TL predicted by PE simulations (red), LP12 (green), and ML model (blue) for a wind model with a (a) tropospheric duct, (b) stratospheric duct, and (c) thermospheric duct. (a-c) top, effective soundspeed profiles used for PE predictions.

## Requirements
- Python3.7
- pandas
- obspy
- sklearn
- multiprocessing
- seaborn
- tensorflow

## Usage
See Python notebook "run_ML_attenuation.ipynb"

## Data
You can collect the synthetic dataset and the deep learning model here: 10.6084/m9.figshare.22572577

## Paper 
http://dx.doi.org/10.1093/gji/ggac307
Modelling the spatial distribution of infrasound attenuation (or transmission loss, TL) is key to understanding and interpreting microbarometer data and observations. Such predictions enable the reliable assessment of infrasound source characteristics such as ground pressure levels associated with earthquakes, man-made or volcanic explosion properties, and ocean-generated microbarom wavefields. However, the computational cost inherent in full-waveform modelling tools, such as Parabolic Equation (PE) codes, often prevents the exploration of a large parameter space, i.e., variations in wind models, source frequency, and source location, when deriving reliable estimates of source or atmospheric properties – in particular for real-time and near-real-time applications. Therefore, many studies rely on analytical regression-based heuristic TL equations that neglect complex vertical wind variations and the range-dependent variation in the atmospheric properties. This introduces significant uncertainties in the predicted TL. In the current contribution, we propose a deep learning approach trained on a large set of simulated wavefields generated using PE simulations and realistic atmospheric winds to predict infrasound ground-level amplitudes up to 1000 km from a ground-based source. Realistic range dependent atmospheric winds are constructed by combining ERA5, NRLMSISE-00, and HWM-14 atmospheric models, and small-scale gravity-wave perturbations computed using the Gardner model. Given a set of wind profiles as input, our new modelling framework provides a fast (0.05 s runtime) and reliable (∼5 dB error on average, compared to PE simulations) estimate of the infrasound TL.

## Citation
Brissaud, Q., Näsholm, S. P., Turquet, A., & Le Pichon, A. (2023). Predicting infrasound transmission loss using deep learning. Geophysical Journal International, 232(1), 274-286.
```
@article{brissaud2023predicting,
  title={Predicting infrasound transmission loss using deep learning},
  author={Brissaud, Quentin and N{\"a}sholm, Sven Peter and Turquet, Antoine and Le Pichon, Alexis},
  journal={Geophysical Journal International},
  volume={232},
  number={1},
  pages={274--286},
  year={2023},
  publisher={Oxford University Press}
}
```
