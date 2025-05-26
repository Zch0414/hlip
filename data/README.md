## CT-RATE

Downloading the CT-RATE dataset with:
```bash
python download_valid.py
python download_train.py
```
Downloading the training split may take ~2 days. Once complete, process the dataset with:
```bash
python process.py --num-cpus 8 --data valid --root-dir '/download/ct_rate/dataset/' --save-dir '/data/ct_rate/'
python process.py --num-cpus 8 --data train --root-dir '/download/ct_rate/dataset/' --save-dir '/data/ct_rate/'
```
Data type and spacing should not be critical concerns based on our analysis. One may consider the following commands to reduce the dataset size:
```bash
--save-astype; --spacing
```
All necessary files have already been provided, some of which are provided by [fVLM](https://github.com/alibaba-damo-academy/fvlm). 
There is no need to run the other code; it is included solely as a reference to illustrate how the files were generated.

**reference:**
```bib
@misc{hamamci2024foundation,
  title={Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography}, 
  author={Ibrahim Ethem Hamamci and Sezgin Er and Furkan Almas and Ayse Gulnihan Simsek and Sevval Nil Esirgun and Irem Dogan and Muhammed Furkan Dasdelen and Omer Faruk Durugol and Bastian Wittmann and Tamaz Amiranashvili and Enis Simsar and Mehmet Simsar and Emine Bensu Erdemir and Abdullah Alanbay and Anjany Sekuboyina and Berkan Lafci and Christian Bluethgen and Mehmet Kemal Ozdemir and Bjoern Menze},
  year={2024},
  eprint={2403.17834},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2403.17834}, 
}
```

## Rad-ChestCT

Downloading the Rad-ChestCT dataset with:
```bash
python download.py
```
Once complete, process the dataset with:
```bash
python process.py --num-cpus 8 --data valid --root-dir '/download/rad_chestct/' --save-dir '/data/rad_chestct/'
```
All necessary files have already been provided. There is no need to run the other code; it is included solely as a reference to illustrate how the files were generated.

**reference:**
```bib
@article{draelos2021machine,
  title={Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes},
  author={Draelos, Rachel Lea and Dov, David and Mazurowski, Maciej A and Lo, Joseph Y and Henao, Ricardo and Rubin, Geoffrey D and Carin, Lawrence},
  journal={Medical image analysis},
  volume={67},
  pages={101857},
  year={2021},
  publisher={Elsevier}
}
```
