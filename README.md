# Opt-SSM 🤖 🚀
*A toolbox for learning optimal projections onto spectral submanifolds*

This codebase accompanies the paper:  
**_Taming High-Dimensional Dynamics: Learning Optimal Projections onto Spectral Submanifolds_**

![Figure Description](./data/assets/foliation.png)

## 📦 Installation

It is recommended to work in a virtual environment.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Opt-SSM:

```bash
pip install -r requirements.txt
```
✅ If you have all the packages with the correct versions, you're good to go!


## 🧠 Usage
Example usage of the OptSSM class:

```python
from utils.ssm import OptSSM

opt_ssm = OptSSM(aut_trajs_obs=aut_trajs_obs,
                 t_split=3.5,
                 SSMDim=5,
                 SSMOrder=2,
                 ROMOrder=2,
                 N_delay=3,
                 ts=ts,
                 verbose=True)
```
ℹ️ Want to see a complete example? Check out one of the notebooks, e.g. the [slow-fast example](./sim_slow-fast.ipynb).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* We thank the authors of the package [SSMLearnPy](https://github.com/haller-group/SSMLearnPy) for inspiration
* Slow-fast and simplified fluid benchmarks are adapted from *Learning nonlinear projections for reduced-order modeling of dynamical systems using constrained autoencoders* ([Chaos, 2023](https://pubs.aip.org/aip/cha/article/33/11/113130/2923554))

## 📚 Citation
```
@article{buurmeijere2025optssm,
      author = {Buurmeijer, Hugo and Pabon, Luis and Alora, John Irvin and Roshan, Kaundinya and Haller, George and Pavone, Marco},
      title = {Taming High-Dimensional Dynamics: Learning Optimal Projections onto Spectral Submanifolds},
      year = {2025},
      journal = {arXiv preprint arXiv:2504.03157},
      note = {Submitted to IEEE Conference on Decision and Control (CDC), Rio de Janeiro, 2025},
      }
```
