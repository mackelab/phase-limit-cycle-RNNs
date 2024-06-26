### Code for Trained recurrent neural networks develop  phase-locked limit cycles in a working memory task.

[click here to go to the paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011852)


First install the conda environment: 
```conda env create -f phase_env_ubu.yml```, if you are on Ubuntu, or 
```conda env create -f phase_env_mac.yml``` if you are on MacOS, then 
activate it.


You can train an RNN models by running ```python rnn_scripts/run_training.py```
All the paper figures can be recreated with the notebooks inside the generate_figures folder


To create the 3D plots you need to allow the [Mayavi](https://mayavi.readthedocs.io/en/latest/) plug-in:
```$ jupyter nbextension install --py mayavi --user```
```$ jupyter nbextension enable --py mayavi --user```


Paper written by: Matthijs Pals, Jakob Macke and Omri Barak, code written by: Matthijs Pals
