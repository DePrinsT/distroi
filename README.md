<p align='center'>
  <br/>
  <img src="./docs/logo/distroi_logo.png" width="240" height="64" alt= >
  <br/>
</p>

**`DISTROI`**, or **D**isk **I**nner **ST**ructure **R**econnaissance through **O**ptical **I**nteferometry, is a pure Python
package intended to compute optical interferometry (OI) observables  from radiative transfer (RT) model images and compare
them to observations from modern OI observatories (VLTI, CHARA, ...).

# Dependencies

`DISTROI` depends on a few other Python packages. The main dependencies are listed in the requirements.txt file supplied
with the source code. In addition,  the distroiEnv.yml file can be used to create a conda environment with all
dependencies installed via 'conda env create -f distroiEnv.yml'.

# Examples

After installing all the dependencies, run the single_disk_model.py script in the 'examples' folder. This will
run OI observable calculation for an example model of the dusty circumbinary disk surroudning the IRAS 08544-4431
system. This model is calculated using the [MCFOST radiative transfer code](https://ipag.osug.fr/~pintec/mcfost/docs/html/overview.html).
`DISTROI` is applied to four separate datasets, covering the H, K, L and N-band, and shows the resulting plots.

# Documentation

The API documentation of `DISTROI` is accessible on its own [Read the Docs page](https://distroi.readthedocs.io/en/latest/). This API documentation is generated automatically from the source code docstring using [Sphinx](https://www.sphinx-doc.org/en/master/).

# Use and collaboration

If you wish to use `DISTROI` in your own research, including if there are missing features you would need or if you wish to apply it to outputs of your own RT codes, don't hesitate to contact [Toon De Prins](https://deprinst.github.io/).

# Acknowledgement

If `DISTROI` proves useful in your own publications, we kindly ask you to mention a link to the [GitHub repository](https://github.com/DePrinsT/distroi) in the footnotes of your publications.

# Issues and contact

If you face any issues when installing or using `DISTROI`, report them at the project's [GitHub issues page](https://github.com/DePrinsT/distroi/issues). For any further help, please contact [Toon De Prins](https://deprinst.github.io/).

# Developers and contributors

**Devlopers**

- [Toon De Prins](https://deprinst.github.io/)

# License

DISTROI is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by  the Free Software Foundation, either version 3.0 of the License, or (at your option) any later version.
DISTROI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with DISTROI. If not, see
<https://www.gnu.org/licenses/>.

---

# NOTE

This project is under active development. Its current structure is bound to change significantly in the near future

---
