# dmp_lib
The __dmp_lib__ library offers several Dynamic Movement Primitive (DMP) implementations for robot learning from demonstrations:
- *Standard DMPs*
- *Bimanual DMPs*
- *Target-Referred DMPs*


## Install ##

The package can be installed by running

```
pip install -e .
```

## Repository Contents ##

The repository contains two folders: __dmp_lib__ and __example__.

The __dmp_lib__ folder contains all the modules implementing DMPs.

The __example__ folder contains example scripts (run them from inside it).

The complete library documentation is available at: [https://fabio-amadio.github.io/dmp_lib/](https://fabio-amadio.github.io/dmp_lib/).

## Citing
If you use this package for any academic work, please cite our original [paper](https://ieeexplore.ieee.org/document/10000233).
```bibtex
@inproceedings{amadio2022target,
  author={Amadio, Fabio and Laghi, Marco and Raiano, Luigi and Rollo, Federico and Zunino, Andrea and Raiola, Gennaro and Ajoudani, Arash},
  booktitle={2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)}, 
  title={Target-Referred DMPs for Learning Bimanual Tasks from Shared-Autonomy Telemanipulation}, 
  year={2022},
  volume={},
  number={},
  pages={496-503},
  doi={10.1109/Humanoids53995.2022.10000233}
}
```
